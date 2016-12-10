# pylint: skip-file
import argparse
import cPickle
import os
import os.path as osp
import re
import sys
import time
from functools import partial
from PIL import Image
from multiprocessing import Pool

import numpy as np

import mxnet as mx

from util import transformer as ts
from util import util

from data import make_divisible, parse_split_file


def parse_model_label(args):
    assert args.model is not None
    fields = [_.strip() for _ in osp.basename(args.model).split('_')]
    # parse fields
    i = 0
    num_fields = len(fields)
    # database
    dataset = fields[i] if args.dataset is None else args.dataset
    i += 1
    # network structure
    assert fields[i].startswith('rn')
    net_type = re.compile('rn[a-z]*').findall(fields[i])[0]
    net_name = fields[i][len(net_type):].strip('-')
    i += 1
    # number of classes
    assert fields[i].startswith('cls')
    classes = int(fields[i][len('cls'):])
    i += 1
    # feature resolution
    feat_stride = 32
    if i < num_fields and fields[i].startswith('s'):
        feat_stride = int(fields[i][len('s'):])
        i += 1

    model_specs = {
        # model
        'net_type': net_type,
        'net_name': net_name,
        'classes': classes,
        'feat_stride': feat_stride,
        # data
        'dataset': dataset,
    }
    return model_specs


def parse_args():
    parser = argparse.ArgumentParser(description='Tune FCRNs from ResNets.')
    parser.add_argument('--gpus', default='0',
                        help='The devices to use, e.g. 0,1,2,3')
    parser.add_argument('--dataset', default=None,
                        help='The dataset to use, e.g. cityscapes, voc.')
    parser.add_argument('--split', default='train',
                        help='The split to use, e.g. train, trainval.')
    parser.add_argument('--data-root', dest='data_root',
                        help='The root data dir.',
                        default=None, type=str)
    parser.add_argument('--output', default=None,
                        help='The output dir.')
    parser.add_argument('--model', default=None,
                        help='The unique label of this model.')
    parser.add_argument('--weights', default='models/resnet50-0001.params',
                        help='The path of a pretrained model (mxnet format).')
    parser.add_argument('--base-lr', dest='base_lr',
                        help='The lr to start from.',
                        default=None, type=float)
    parser.add_argument('--from-epoch', dest='from_epoch',
                        help='The epoch to start from.',
                        default=None, type=int)
    parser.add_argument('--stop-epoch', dest='stop_epoch',
                        help='The index of epoch to stop.',
                        default=None, type=int)
    parser.add_argument('--to-epoch', dest='to_epoch',
                        help='The number of epochs to run.',
                        default=None, type=int)
    parser.add_argument('--phase',
                        help='Phase of this call, e.g., train/val.',
                        default='train', type=str)
    # for testing
    parser.add_argument('--test-scales', dest='test_scales',
                        help='Lengths of the longer side to resize an image into, e.g., 224,256.',
                        default=None, type=str)
    parser.add_argument('--test-flipping', dest='test_flipping', 
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument('--test-steps', dest='test_steps',
                        help='The number of steps to take, for predictions at a higher resolution.',
                        default=1, type=int)
    parser.add_argument('--save-predictions', dest='save_predictions', 
                        help='If save the predicted score maps.',
                        default=False, action='store_true')
    parser.add_argument('--no-save-results', dest='save_results', 
                        help='If save the predicted pixel-wise labels.',
                        default=True, action='store_false')
    #
    parser.add_argument('--kvstore', dest='kvstore',
                        help='The type of kvstore, e.g., local/device.',
                        default='device', type=str)
    parser.add_argument('--prefetch-threads', dest='prefetch_threads',
                        help='The number of threads to fetch data.',
                        default=1, type=int)
    parser.add_argument('--prefetcher', dest='prefetcher',
                        help='The type of prefetercher, e.g., process/thread.',
                        default='thread', type=str)
    parser.add_argument('--cache-images', dest='cache_images', 
                        help='If cache images, e.g., 0/1',
                        default=None, type=int)
    parser.add_argument('--log-file', dest='log_file',
                        default=None, type=str)
    parser.add_argument('--check-start', dest='check_start',
                        help='The first epoch to snapshot.',
                        default=1, type=int)
    parser.add_argument('--check-step', dest='check_step',
                        help='The steps between adjacent snapshots.',
                        default=4, type=int)
    parser.add_argument('--debug',
                        help='True means logging debug info.',
                        default=False, action='store_true')
    parser.add_argument('--backward-do-mirror', dest='backward_do_mirror',
                        help='True means less gpu memory usage.',
                        default=False, action='store_true')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.debug:
        os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

    if args.backward_do_mirror:
        os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'
    
    if args.output is None:
        if args.phase == 'val':
            args.output = osp.dirname(args.weights)
        else:
            args.output = '../output'
    
    if args.weights is not None:
        #
        if args.model is None:
            assert '_ep-' in args.weights
            parts = osp.basename(args.weights).split('_ep-')
            args.model = '_'.join(parts[:-1])
        #
        if args.from_epoch is None:
            parts = osp.splitext(osp.basename(args.weights))[0].split('-')
            args.from_model = osp.join(osp.dirname(args.weights), '-'.join(parts[:-1]))
            args.from_epoch = int(parts[-1])
    
    if args.from_epoch is None:
        args.from_epoch = 0
    
    if args.log_file is None:
        if args.phase == 'train':
            args.log_file = '{}.log'.format(args.model)
        elif args.phase == 'val':
            suffix = ''
            if args.split != 'val':
                suffix = '_{}'.format(args.split)
            args.log_file = '{}{}.log'.format(osp.splitext(osp.basename(args.weights))[0], suffix)
        else:
            raise NotImplementedError('Unknown phase: {}'.format(args.phase))
    
    model_specs = parse_model_label(args)
    if args.data_root is None:
        args.data_root = osp.join('../data', model_specs['dataset'])
    
    return args, model_specs


def get_dataset_specs(args, model_specs):
    dataset = model_specs['dataset']
    meta = {}
    meta_path = osp.join('issegm/data', dataset, 'meta.pkl')
    if osp.isfile(meta_path):
        with open(meta_path) as f:
            meta = cPickle.load(f)
    
    label_2_id = None
    id_2_label = None
    cmap = None
    cmap_path = 'data/shared/cmap.pkl'
    ident_size = False
    cache_images = args.phase == 'train'
    mx_workspace = 1650
    if dataset == 'ade20k':
        num_classes = model_specs.get('classes', 150)
        label_2_id = np.arange(-1, 150)
        label_2_id[0] = 255
        id_2_label = np.arange(1, 256+1)
        id_2_label[255] = 0
        valid_labels = range(1, 150+1)
        #
        if args.split == 'test':
            cmap_path = None
        #
        max_shape = np.array((2100, 2100))
        if model_specs.get('balanced', False) and args.split == 'trainval':
            meta['image_classes']['trainval'] = meta['image_classes']['train'] + meta['image_classes']['val']    
    elif dataset == 'cityscapes':
        sys.path.insert(0, '../cityscapesScripts/cityscapesscripts/helpers')
        from labels import id2label, trainId2label
        #
        num_classes = model_specs.get('classes', 19)
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        valid_labels = sorted(set(id_2_label.ravel()))
        #
        cmap = np.zeros((256,3), dtype=np.uint8)
        for i in id2label.keys():
            cmap[i] = id2label[i].color
        #
        ident_size = True
        #
        max_shape = np.array((1024, 2048))
        #
        if args.split in ('train+', 'trainval+'):
            cache_images = False
        #
        if args.phase in ('val',):
            mx_workspace = 8000
    elif dataset == 'coco':
        sys.path.insert(0, osp.join(args.data_root, 'PythonAPI'))
        from pycocotools.coco import COCO
        coco = COCO(osp.join(args.data_root, 'annotations', 'instances_minival2014.json'))
        #
        id_2_label = np.array([0] + sorted(coco.getCatIds()))
        assert len(id_2_label) == 81
        valid_labels = id_2_label.tolist()
        num_classes = model_specs.get('classes', 81)
        label_2_id = 255 * np.ones((256,))
        for i, l in enumerate(id_2_label):
            label_2_id[l] = i
        #
        max_shape = np.array((640, 640))
    elif dataset == 'pascal-context':
        num_classes = model_specs.get('classes', 60)
        valid_labels = range(num_classes)
        #
        max_shape = np.array((500, 500))
    elif dataset == 'voc':
        num_classes = model_specs.get('classes', 21)
        valid_labels = range(num_classes)
        #
        if args.split in ('train++',):
            max_shape = np.array((640, 640))
        else:
            max_shape = np.array((500, 500))
    else:
        raise NotImplementedError('Unknow dataset: {}'.format(dataset))
    
    if cmap is None and cmap_path is not None:
        if osp.isfile(cmap_path):
            with open(cmap_path) as f:
                cmap = cPickle.load(f)
    
    meta['label_2_id'] = label_2_id
    meta['id_2_label'] = id_2_label
    meta['valid_labels'] = valid_labels
    meta['cmap'] = cmap
    meta['ident_size'] = ident_size
    meta['max_shape'] = meta.get('max_shape', max_shape)
    meta['cache_images'] = args.cache_images if args.cache_images is not None else cache_images
    meta['mx_workspace'] = mx_workspace
    return meta


def _get_scalemeanstd():
    if model_specs['net_type'] == 'rn':
        return -1, np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3)), None
    if model_specs['net_type'] in ('rna',):
        return (1.0/255,
            np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
            np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
    return None, None, None

def _get_transformer_post():
    scale, mean_, std_ = _get_scalemeanstd()
    transformers = []
    if scale > 0:
        transformers.append(ts.ColorScale(np.single(scale)))
    transformers.append(ts.ColorNormalize(mean_, std_))
    return transformers

def _get_module(margs, dargs, net=None):
    if net is None:
        # the following lines show how to create symbols for our networks
        if model_specs['net_type'] == 'rna':
            from util.symbol.symbol import cfg as symcfg
            if model_specs['net_name'] == 'a1':
                symcfg['workspace'] = dargs.mx_workspace
                symcfg['use_global_stats'] = True
                from util.symbol.resnet_v2 import fcrna_model_a1
                net = fcrna_model_a1(margs.classes, margs.feat_stride)
        if net is None:
            raise NotImplementedError('Unknown network: {}'.format(vars(margs)))
    contexts = [mx.gpu(int(_)) for _ in args.gpus.split(',')]
    mod = mx.mod.Module(net, context=contexts)
    return mod

def _make_dirs(path):
    if not osp.isdir(path):
        os.makedirs(path)


def _interp_preds_as_impl(num_classes, im_size, pred_stride, imh, imw, pred):
    imh0, imw0 = im_size
    pred = pred.astype(np.single, copy=False)
    input_h, input_w = pred.shape[0] * pred_stride, pred.shape[1] * pred_stride
    assert pred_stride >= 1.
    this_interp_pred = np.array(Image.fromarray(pred).resize((input_w, input_h), Image.CUBIC))
    if imh0 == imh:
        interp_pred = this_interp_pred[:imh, :imw]
    else:
        interp_method = util.get_interp_method(imh, imw, imh0, imw0)
        interp_pred = np.array(Image.fromarray(this_interp_pred[:imh, :imw]).resize((imw0, imh0), interp_method))
    return interp_pred

def interp_preds_as(im_size, net_preds, pred_stride, imh, imw, threads=4):
    num_classes = net_preds.shape[0]
    worker = partial(_interp_preds_as_impl, num_classes, im_size, pred_stride, imh, imw)
    if threads == 1:
        ret = [worker(_) for _ in net_preds]
    else:
        pool = Pool(threads)
        ret = pool.map(worker, net_preds)
        pool.close()
    return np.array(ret)

class ScoreUpdater(object):
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels
        
        self._confs = np.zeros((c_num, c_num, x_num))
        self._pixels = np.zeros((c_num, x_num))
        self._logger = logger
        self._label = label
        self._info = info
        
    @property
    def info(self):
        return self._info
    
    def reset(self):
        self._start = time.time()
        self._computed = np.zeros((self._pixels.shape[1],))
        self._confs[:] = 0
        self._pixels[:] = 0
    
    @staticmethod
    def calc_updates(valid_labels, pred_label, label):
        num_classes = len(valid_labels)
        
        pred_flags = [set(np.where((pred_label == _).ravel())[0]) for _ in valid_labels]
        class_flags = [set(np.where((label == _).ravel())[0]) for _ in valid_labels]
        
        conf = [len(class_flags[j].intersection(pred_flags[k])) for j in xrange(num_classes) for k in xrange(num_classes)]
        pixel = [len(class_flags[j]) for j in xrange(num_classes)]
        return np.single(conf).reshape((num_classes, num_classes)), np.single(pixel)
    
    def do_updates(self, conf, pixel, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._confs[:, :, i] = conf
        self._pixels[:, i] = pixel
    
    def update(self, pred_label, label, i, computed=True):
        conf, pixel = ScoreUpdater.calc_updates(self._valid_labels, pred_label, label)
        self.do_updates(conf, pixel, i, computed)
        self.scores(i)
        
    def scores(self, i=None, logger=None):
        confs = self._confs
        pixels = self._pixels
        
        num_classes = pixels.shape[0]
        x_num = pixels.shape[1]
        
        class_pixels = pixels.sum(1)
        class_pixels += class_pixels == 0
        scores = confs[xrange(num_classes), xrange(num_classes), :].sum(1)
        acc = scores.sum() / pixels.sum()
        cls_accs = scores / class_pixels
        class_preds = confs.sum(0).sum(1)
        ious = scores / (class_pixels + class_preds - scores)
        
        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1.*self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i+1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}pixel acc: {:.2f}%, mean acc: {:.2f}%, mean iou: {:.2f}%'.\
                format(name, acc*100, cls_accs.mean()*100, ious.mean()*100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(cls_accs*100))
                logger.info('\n{}'.format(ious*100))
        
        return acc, cls_accs, ious
    
    def overall_scores(self, logger=None):
        acc, cls_accs, ious = self.scores(None, logger)
        return acc, cls_accs.mean(), ious.mean()

#@profile
def _val_impl(args, model_specs, logger):
    assert args.prefetch_threads == 1
    assert args.weights is not None
    
    margs = argparse.Namespace(**model_specs)
    dargs = argparse.Namespace(**get_dataset_specs(args, model_specs))
    
    image_list, label_list = parse_split_file(margs.dataset, args.split)
    _, net_args, net_auxs = util.load_params(args.from_model, args.from_epoch)
    net = None
    mod = _get_module(margs, dargs, net)
    has_gt = args.split in ('train', 'val',)
    
    crop_sizes = sorted([int(_) for _ in args.test_scales.split(',')])[::-1]
    # TODO: multi-scale testing
    assert len(crop_sizes) == 1, 'multi-scale testing not implemented'
    label_stride = margs.feat_stride
    crop_size = crop_sizes[0]
    
    save_dir = osp.join(args.output, osp.splitext(args.log_file)[0])
    _make_dirs(save_dir)
    
    x_num = len(image_list)
    
    do_forward = True
    if do_forward:
        batch = None
        transformers = [ts.Scale(crop_size, Image.CUBIC, False)]
        transformers += _get_transformer_post()
        transformer = ts.Compose(transformers)
    
    scorer = ScoreUpdater(dargs.valid_labels, margs.classes, x_num, logger)
    scorer.reset()
    start = time.time()
    done_count = 0
    for i in xrange(x_num):
        sample_name = osp.splitext(osp.basename(image_list[i]))[0]
        
        # skip computed images
        if args.save_predictions:
            pred_save_path = osp.join(save_dir, 'predictions', '{}.h5'.format(sample_name))
            if osp.isfile(pred_save_path):
                logger.info('Skipped {} {}/{}'.format(sample_name, i+1, x_num))
                continue
            
        im_path = osp.join(args.data_root, image_list[i])
        rim = np.array(Image.open(im_path).convert('RGB'), np.uint8)
        
        if do_forward:
            im = transformer(rim)
            imh, imw = im.shape[:2]
            
            # init
            if batch is None:
                if dargs.ident_size:
                    input_h = make_divisible(imh, margs.feat_stride)
                    input_w = make_divisible(imw, margs.feat_stride)
                else:
                    input_h = input_w = make_divisible(crop_size, margs.feat_stride)
                label_h, label_w = input_h / label_stride, input_w / label_stride
                test_steps = args.test_steps
                pred_stride = label_stride / test_steps
                pred_h, pred_w = label_h * test_steps, label_w * test_steps
        
                input_data = np.zeros((1, 3, input_h, input_w), np.single)
                input_label = 255 * np.ones((1, label_h * label_w), np.single)
                dataiter = mx.io.NDArrayIter(input_data, input_label)
                batch = dataiter.next()
                mod.bind(dataiter.provide_data, dataiter.provide_label, for_training=False, force_rebind=True)
                if not mod.params_initialized:
                    mod.init_params(arg_params=net_args, aux_params=net_auxs)
            
            nim = np.zeros((3, imh+label_stride, imw+label_stride), np.single)
            sy = sx = label_stride // 2
            nim[:, sy:sy+imh, sx:sx+imw] = im.transpose(2, 0, 1)
            
            net_preds = np.zeros((margs.classes, pred_h, pred_w), np.single)
            sy = sx = pred_stride // 2 + np.arange(test_steps) * pred_stride
            for ix in xrange(test_steps):
                for iy in xrange(test_steps):
                    input_data = np.zeros((1, 3, input_h, input_w), np.single)
                    input_data[0, :, :imh, :imw] = nim[:, sy[iy]:sy[iy]+imh, sx[ix]:sx[ix]+imw]
                    batch.data[0] = mx.nd.array(input_data)
                    mod.forward(batch, is_train=False)
                    this_call_preds = mod.get_outputs()[0].asnumpy()[0]
                    if args.test_flipping:
                        batch.data[0] = mx.nd.array(input_data[:, :, :, ::-1])
                        mod.forward(batch, is_train=False)
                        this_call_preds = 0.5 * (this_call_preds + mod.get_outputs()[0].asnumpy()[0][:, :, ::-1])
                    net_preds[:, iy:iy+pred_h:test_steps, ix:ix+pred_w:test_steps] = this_call_preds

        # save predicted probabilities
        if args.save_predictions:
            _make_dirs(osp.dirname(pred_save_path))
            tmp = (rim.shape[:2], net_preds.astype(np.float16), pred_stride, imh, imw)
            util.h5py_save(pred_save_path, *tmp)
        
        if args.save_results:
            # compute pixel-wise predictions
            interp_preds = interp_preds_as(rim.shape[:2], net_preds, pred_stride, imh, imw)
            pred_label = interp_preds.argmax(0)
            if dargs.id_2_label is not None:
                pred_label = dargs.id_2_label[pred_label]
            
            # save predicted labels into an image
            out_path = osp.join(save_dir, '{}.png'.format(sample_name))
            im_to_save = Image.fromarray(pred_label.astype(np.uint8))
            if dargs.cmap is not None:
                im_to_save.putpalette(dargs.cmap.ravel())
            im_to_save.save(out_path)
        else:
            assert not has_gt
        
        done_count += 1
        if not has_gt:
            logger.info('Done {}/{} with speed: {:.2f}/s'.format(i+1, x_num, 1.*done_count / (time.time() - start)))
            continue
        
        label_path = osp.join(args.data_root, label_list[i])
        label = np.array(Image.open(label_path), np.uint8)
        
        # save correctly labeled pixels into an image
        out_path = osp.join(save_dir, 'correct', '{}.png'.format(sample_name))
        _make_dirs(osp.dirname(out_path))
        invalid_mask = np.logical_not(np.in1d(label, dargs.valid_labels)).reshape(label.shape)
        Image.fromarray((invalid_mask*255 + (label == pred_label)*127).astype(np.uint8)).save(out_path)
        
        scorer.update(pred_label, label, i)
    logger.info('Done in %.2f s.', time.time() - start)

if __name__ == "__main__":
    util.cfg['choose_interpolation_method'] = True
    
    args, model_specs = parse_args()
    
    if len(args.output) > 0:
        _make_dirs(args.output)
    
    logger = util.set_logger(args.output, args.log_file, args.debug)
    logger.info('start with arguments %s', args)
    logger.info('and model specs %s', model_specs)
    
    if args.phase == 'train':
        NotImplementedError('Unknown phase: {}'.format(args.phase))
        #_train_impl(args, model_specs, logger)
    elif args.phase == 'val':
        _val_impl(args, model_specs, logger)
    else:
        raise NotImplementedError('Unknown phase: {}'.format(args.phase))

