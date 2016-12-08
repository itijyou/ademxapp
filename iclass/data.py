# pylint: skip-file
"""
file iterator for image classification
"""
import os
import time
from PIL import Image

import numpy as np

from mxnet.io import DataBatch, DataIter
from mxnet.ndarray import array

from util.io import BatchFetcherGroup
#from util.sampler import BalancedSampler_OneClassPerImage as BalancedSampler
from util.sampler import FixedSampler, RandomSampler
from util.util import as_list


def parse_split_file(dataset, split, data_root):
    split_filename = 'iclass/data/{}/{}.lst'.format(dataset, split)
    image_list = []
    label_list = []
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[2]))
            label_list.append(int(fields[1]))
    return image_list, label_list

class FileIter(DataIter):
    """FileIter object for image classification.
    Parameters
    ----------

    dataset : string
        dataset
    split : string
        data split
        the list file of images and labels, whose each line is in the format:
        image_id(0 indexed) \t image_label \t image_file_path
    data_root : string
        the root data directory
    data_name : string
        the data name used in the network input
    label_name : string
        the label name used in SoftmaxOutput
    sampler: obj
        how to shuffle the samples per epoch
    has_gt: bool
        if there are ground truth labels
    batch_images : int
        the number of images per batch
    transformer : object
        the transformer for data augmentation
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
                 batch_images = 256,
                 transformer = None,
                 prefetch_threads = 1,
                 prefetcher_type = 'thread',):
        super(FileIter, self).__init__()
        
        self._data_name = data_name
        self._label_name = label_name
        self._has_gt = has_gt
        self._batch_images = batch_images
        self._transformer = transformer
        
        self._image_list, self._label_list = parse_split_file(dataset, split, data_root)
        self._perm_len = len(self._image_list)
        if sampler == 'fixed':
            sampler = FixedSampler(self._perm_len)
        elif sampler == 'random':
            sampler = RandomSampler(self._perm_len)
        
        data_batch = self.read([0])
        self.batch_size = self._batch_images * data_batch[1].shape[0]
        self._data = list({self._data_name: data_batch[0]}.items())
        self._label = list({self._label_name: data_batch[1]}.items())
        
        self._fetcher = BatchFetcherGroup(self,
                                          sampler,
                                          batch_images,
                                          prefetch_threads,
                                          prefetch_threads*2,
                                          prefetcher_type)

    def read(self, db_inds):
        outputs = [[], [],]
        for db_ind in db_inds:
            # load an image
            rim = Image.open(self._image_list[db_ind]).convert('RGB')
            data = np.array(rim, np.uint8)
            # jitter
            if self._transformer is not None:
                data = self._transformer(data)
            data_list = as_list(data)
            for datum in data_list:
                outputs[0].append(datum.transpose(2, 0, 1)[np.newaxis])
            if self._has_gt:
                outputs[1].append([self._label_list[db_ind]] * len(data_list))
        for i, output in enumerate(outputs):
            outputs[i] = np.concatenate(output, axis=0)
        return tuple(outputs)
    
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
    
    def draw_sample(self, data_batch, meanstd, rgb_scale):
        import pylab as pl
        for i in xrange(data_batch.data[0].shape[0]):
            im = data_batch.data[0][i].asnumpy().transpose(1, 2, 0)
            im = im * meanstd[0] + meanstd[1]
            im *= rgb_scale
            im = np.maximum(0, np.minimum(255, im))
            pl.imshow(im.astype(np.uint8))
            pl.show()

