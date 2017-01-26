"""
file iterator for image semantic segmentation
"""
import os
import time
from PIL import Image

import numpy as np
import numpy.random as npr

from mxnet.io import DataBatch, DataIter
from mxnet.ndarray import array

from util.io import BatchFetcherGroup
from util.sampler import FixedSampler, RandomSampler
from util.util import get_interp_method, load_image_with_cache


def parse_split_file(dataset, split, data_root=''):
    split_filename = 'issegm/data/{}/{}.lst'.format(dataset, split)
    image_list = []
    label_list = []
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[0]))
            label_list.append(os.path.join(data_root, fields[1]))
    return image_list, label_list

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)

class FileIter(DataIter):
    """FileIter object for image semantic segmentation.
    Parameters
    ----------

    dataset : string
        dataset
    split : string
        data split
        the list file of images and labels, whose each line is in the format:
        image_path \t label_path
    data_root : string
        the root data directory
    data_name : string
        the data name used in the network input
    label_name : string
        the label name used in SoftmaxOutput
    sampler: str
        how to shuffle the samples per epoch
    has_gt: bool
        if there are ground truth labels
    batch_images : int
        the number of images per batch
    meta : dict
        dataset specifications
    
    prefetch_threads: int
        the number of prefetchers
    prefetcher_type: string
        the type of prefechers, e.g., process/thread
    """
    def __init__(self,
                 dataset,
                 split,
                 data_root,
                 data_name = 'data',
                 label_name = 'softmax_label',
                 sampler = 'fixed',
                 has_gt = True,
                 batch_images = 1,
                 meta = None,
                 ####
                 # transforming by the old fasion reader
                 rgb_mean = None, # (R, G, B)
                 feat_stride = 32,
                 label_stride = 32,
                 label_steps = 1,
                 origin_size = None,
                 crop_size = 0,
                 scale_rate_range = None,
                 crops_per_image = 1,
                 # or the new functional reader
                 transformer = None,
                 # image only pre-processing such as rgb scale and meanstd
                 transformer_image = None,
                 ####
                 prefetch_threads = 1,
                 prefetcher_type = 'thread',):
        super(FileIter, self).__init__()
        assert crop_size > 0
        
        self._meta = meta
        self._data_name = data_name
        self._label_name = label_name
        self._has_gt = has_gt
        self._batch_images = batch_images
        self._feat_stride = feat_stride
        self._label_stride = label_stride
        self._label_steps = label_steps
        self._origin_size = origin_size
        self._crop_size = make_divisible(crop_size, self._feat_stride)
        self._crops_per_image = crops_per_image
        #
        self._data_mean = None if rgb_mean is None else rgb_mean.reshape((1, 1, 3))
        self._scale_rate_range = (1.0, 1.0) if scale_rate_range is None else scale_rate_range
        #
        self._transformer = transformer
        self._transformer_image = transformer_image
        self._reader = self._read if self._transformer is None else self._read_transformer
        
        self._ignore_label = 255

        self._image_list, self._label_list = parse_split_file(dataset, split, data_root)
        self._perm_len = len(self._image_list)
        if sampler == 'fixed':
            sampler = FixedSampler(self._perm_len)
        elif sampler == 'random':
            sampler = RandomSampler(self._perm_len)
        
        assert self._label_steps == 1
        assert self._crops_per_image == 1
        self.batch_size = self._batch_images
        
        self._cache = {} if self._meta['cache_images'] else None
        
        self._fetcher = BatchFetcherGroup(self,
                                          sampler,
                                          batch_images,
                                          prefetch_threads,
                                          prefetch_threads*2,
                                          prefetcher_type)
        
        if crop_size > 0:
            crop_h = crop_w = self._crop_size
        else:
            rim = load_image_with_cache(self._image_list[0], self._cache)
            crop_h = make_divisible(rim.size[1], self._feat_stride)
            crop_w = make_divisible(rim.size[0], self._feat_stride)
        self._data = list({self._data_name: np.zeros((1, 3, crop_h, crop_w), np.single)}.items())
        self._label = list({self._label_name: np.zeros((1, crop_h * crop_w / self._label_stride**2), np.single)}.items())

    def read(self, db_inds):
        return self._reader(db_inds)
    
    def _read_transformer(self, db_inds):
        output_list = []
        output_shape = [0, 0]
        for db_ind in db_inds:
            # load an image
            rim = load_image_with_cache(self._image_list[db_ind], self._cache).convert('RGB')
            data = np.array(rim, np.uint8)
            # load the label
            if self._has_gt:
                rlabel = load_image_with_cache(self._label_list[db_ind], self._cache)
                label = np.array(rlabel, np.uint8)
            else:
                label = self._ignore_label * np.ones(data.shape[:2], np.uint8)
            # jitter
            if self._transformer is not None:
                data, label = self._transformer(data, label)
            lsy = lsx = self._label_stride / 2
            label = label[lsy::self._label_stride, lsx::self._label_stride]
            output_list.append((data, label))
            output_shape = np.maximum(output_shape, data.shape[:2])
        
        output_shape = [make_divisible(_, self._feat_stride) for _ in output_shape]
        output = [np.zeros((self.batch_size, 3, output_shape[0], output_shape[1]), np.single),
                  self._ignore_label * np.ones((self.batch_size, output_shape[0]/self._label_stride, output_shape[1]/self._label_stride), np.single),]
        for i in xrange(len(output_list)):
            imh, imw = output_list[i][0].shape[:2]
            output[0][i][:, :imh, :imw] = output_list[i][0].transpose(2, 0, 1)
            output[1][i][:imh, :imw] = output_list[i][1]
        output[1] = output[1].reshape((self.batch_size, -1))
            
        return tuple(output)
            
    def _read(self, db_inds):
        max_h, max_w = self._meta['max_shape']
        label_2_id = self._meta['label_2_id']
        
        target_size_range = [int(_*self._origin_size) for _ in self._scale_rate_range]
        min_rate = 1.*target_size_range[0] / max(max_h, max_w)
        target_crop_size = self._crop_size
        max_crop_size = int(target_crop_size / min_rate)
        label_size = target_crop_size // self._label_stride
        assert label_size * self._label_stride == target_crop_size
        label_per_image = label_size**2
        locs_per_crop = self._label_steps ** 2
        output = []
        for _ in xrange(locs_per_crop * self._crops_per_image):
            output_data = np.zeros((self._batch_images, 3, target_crop_size, target_crop_size), np.single)
            output_label = np.zeros((self._batch_images, label_per_image), np.single)
            output.append([output_data, output_label])
        for i,db_ind in enumerate(db_inds):
            # load an image
            im = np.array(load_image_with_cache(self._image_list[db_ind], self._cache).convert('RGB'))
            h, w = im.shape[:2]
            assert h <= max_h and w <= max_w
            # randomize the following cropping
            target_size = npr.randint(target_size_range[0], target_size_range[1] + 1)
            rate = 1.*target_size / max(h, w)
            crop_size = int(target_crop_size / rate)
            label_stride = self._label_stride / rate
            # make sure there is a all-zero border
            d0 = max(1, int(label_stride // 2))
            # allow shifting within the grid between the used adjacent labels
            d1 = max(0, int(label_stride - d0))
            # prepare the image
            nim_size = max(max_crop_size, max_h, max_w) + d1 + d0
            nim = np.zeros((nim_size, nim_size, 3), np.single)
            nim += self._data_mean
            nim[d0:d0+h, d0:d0+w, :] = im
            # label
            nlabel = self._ignore_label * np.ones((nim_size, nim_size), np.uint8)
            label = np.array(load_image_with_cache(self._label_list[db_ind], self._cache))
            if label_2_id is not None:
                label = label_2_id[label]
            nlabel[d0:d0+h, d0:d0+w] = label
            # crop
            real_label_stride = label_stride / self._label_steps
            sy = npr.randint(0, max(1, real_label_stride, d0 + h - crop_size + 1), self._crops_per_image)
            sx = npr.randint(0, max(1, real_label_stride, d0 + w - crop_size + 1), self._crops_per_image)
            dyx = np.arange(0, label_stride, real_label_stride).astype(np.int32)[:self._label_steps].tolist()
            dy = dyx * self._label_steps
            dx = sum([[_] * self._label_steps for _ in dyx], [])
            for k in xrange(self._crops_per_image):
                do_flipping = npr.randint(2) == 0
                for j in xrange(locs_per_crop):
                    # cropping & resizing image
                    tim = nim[sy[k]+dy[j]:sy[k]+dy[j]+crop_size, sx[k]+dx[j]:sx[k]+dx[j]+crop_size, :].astype(np.uint8)
                    assert tim.shape[0] == tim.shape[1] == crop_size
                    interp_method = get_interp_method(crop_size, crop_size, target_crop_size, target_crop_size)
                    rim = Image.fromarray(tim).resize((target_crop_size,target_crop_size), interp_method)
                    rim = np.array(rim)
                    # cropping & resizing label
                    tlabel = nlabel[sy[k]+dy[j]:sy[k]+dy[j]+crop_size, sx[k]+dx[j]:sx[k]+dx[j]+crop_size]
                    assert tlabel.shape[0] == tlabel.shape[1] == crop_size
                    rlabel = Image.fromarray(tlabel).resize((target_crop_size,target_crop_size), Image.NEAREST)
                    lsy = self._label_stride / 2
                    lsx = self._label_stride / 2
                    rlabel = np.array(rlabel)[lsy : target_crop_size : self._label_stride, lsx : target_crop_size : self._label_stride]
                    # flipping
                    if do_flipping:
                        rim = rim[:, ::-1, :]
                        rlabel = rlabel[:, ::-1]
                    # transformers
                    if self._transformer_image is not None:
                        rim = self._transformer_image(rim)
                    else:
                        rim -= self._data_mean
                    # assign
                    output[k*locs_per_crop+j][0][i,:] = rim.transpose(2,0,1)
                    output[k*locs_per_crop+j][1][i,:] = rlabel.flatten()
        return output
    
    @property
    def batch_images(self):
        return self._batch_images
    
    @property
    def batches_per_epoch(self):
        return self._perm_len // self._batch_images

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self._data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self._label]

    def reset(self):
        self._fetcher.reset()

    def next(self):
        if self._fetcher.iter_next():
            tic = time.time()
            data_batch = self._fetcher.get()
            print 'Waited for {} seconds'.format(time.time() - tic)
        else:
            raise StopIteration
        
        return DataBatch(data=[array(data_batch[0])], label=[array(data_batch[1])])

    def debug(self):
        for i in xrange(self._perm_len):
            self.read([i])
            print 'Done {}/{}'.format(i+1, self._perm_len)

