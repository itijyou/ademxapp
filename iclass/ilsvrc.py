# pylint: skip-file
import argparse
import cPickle
import itertools
import os
import os.path as osp
import re
import sys
import time

import numpy as np

import mxnet as mx

from util import mxutil
from util import transformer as ts
from util import util
from util.initializer import TorchXavier_Linear
from util.lr_scheduler import FixedScheduler, LinearScheduler
from util.optimizer import TorchNesterov

from data import FileIter, parse_split_file


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
    
    # images per batch
    if args.batch_images is not None:
        batch_images = args.batch_images
    # input size
    if args.crop_size is not None:
        crop_size = args.crop_size
    elif args.test_scales is not None:
        crop_size = int(args.test_scales.split(',')[0])
    # learning rate
    lr_params = {
        'type': 'fixed',
        'base': 0.1,
        'args': None,
    }
    if args.base_lr is not None:
        lr_params['base'] = args.base_lr
    # linear
    if args.lr_type in ('linear',):
        lr_params['type'] = args.lr_type
    elif args.lr_type == 'step':
        lr_params['args'] = {'step': [int(_) for _ in args.lr_steps.split(',')],
                             'factor': 0.1}
    
    model_specs = {
        # model
        'lr_params': lr_params,
        'net_type': net_type,
        'net_name': net_name,
        'classes': classes,
        'feat_stride': feat_stride,
        # data
        'dataset': dataset,
        'batch_images': batch_images,
        'crop_size': crop_size,
    }
    return model_specs

def parse_args():
    parser = argparse.ArgumentParser(description='Train/test ResNets.')
    parser.add_argument('--gpus', default='0',
                        help='The devices to use, e.g. 0,1,2,3')
    parser.add_argument('--dataset', default=None,
                        help='The dataset to use, e.g. ilsvrc-cls.')
    parser.add_argument('--split', default='train',
                        help='The split to use, e.g. train/val/trainval.')
    parser.add_argument('--data-root', dest='data_root',
                        help='The root data dir.',
                        default=None, type=str)
    parser.add_argument('--output', default=None,
                        help='The output dir.')
    parser.add_argument('--model', default=None,
                        help='The unique label of this model.')
    parser.add_argument('--batch-images', dest='batch_images',
                        help='The number of images per batch.',
                        default=None, type=int)
    parser.add_argument('--crop-size', dest='crop_size',
                        help='The size of network input during training.',
                        default=None, type=int)
    parser.add_argument('--weights', default=None,
                        help='The path of a pretrained model.')
    #
    parser.add_argument('--lr-type', dest='lr_type',
                        help='The learning rate scheduler, e.g., fixed(default)/step/linear',
                        default=None, type=str)
    parser.add_argument('--base-lr', dest='base_lr',
                        help='The lr to start from.',
                        default=None, type=float)
    parser.add_argument('--lr-steps', dest='lr_steps',
                        help='The steps when to reduce lr.',
                        default=None, type=str)
    #
    parser.add_argument('--from-epoch', dest='from_epoch',
                        help='The epoch to start from.',
                        default=None, type=int)
    parser.add_argument('--stop-epoch', dest='stop_epoch',
                        help='The index of epoch to stop.',
                        default=None, type=int)
    parser.add_argument('--to-epoch', dest='to_epoch',
                        help='The number of epochs to run.',
                        default=None, type=int)
    #
    parser.add_argument('--phase',
                        help='Phase of this call, e.g., train/val.',
                        default='train', type=str)
    # for testing
    parser.add_argument('--test-scales', dest='test_scales',
                        help='Lengths of the shorter side to resize an image into, e.g., 224,256.',
                        default=None, type=str)
    parser.add_argument('--test-flipping', dest='test_flipping', 
                        help='If average predictions of an original and its flipped images.',
                        default=False, action='store_true')
    parser.add_argument('--test-3crops', dest='test_3crops', 
                        help='If average predictions of three crops from an image.',
                        default=False, action='store_true')
    #
    parser.add_argument('--kvstore', dest='kvstore',
                        help='The type of kvstore, e.g., local/device.',
                        default='device', type=str)
    parser.add_argument('--prefetch-threads', dest='prefetch_threads',
                        help='The number of threads to fetch data.',
                        default=1, type=int)
    parser.add_argument('--prefetcher', dest='prefetcher',
                        help='Type of prefetercher, e.g., process/thread.',
                        default='thread', type=str)
    parser.add_argument('--log-file', dest='log_file',
                        default=None, type=str)
    parser.add_argument('--debug',
                        help='True for logging debug info.',
                        default=False, action='store_true')
    parser.add_argument('--backward-do-mirror', dest='backward_do_mirror',
                        help='True for less gpu memory usage.',
                        default=False, action='store_true')
    # for testing (our released old) pre-trained models (Model A and/or Model A1)
    parser.add_argument('--no-choose-interp-method', dest='choose_interpolation_method',
                        help='True to adaptively choose an interpolation method.',
                        default=True, action='store_false')
    parser.add_argument('--pool-top-infer-style', dest='pool_top_infer_style',
                        help='Specify another convention, e.g., caffe.',
                        default=None, type=str)

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
            args.output = os.path.dirname(args.weights)
        else:
            args.output = 'output'
    
    if args.weights is not None:
        #
        if args.model is None:
            assert '_ep-' in args.weights
            parts = os.path.basename(args.weights).split('_ep-')
            args.model = '_'.join(parts[:-1])
        #
        if args.phase == 'train':
            if args.from_epoch is None:
                assert '_ep-' in args.weights
                parts = os.path.basename(args.weights).split('_ep-')
                assert len(parts) == 2
                from_model = parts[0]
                if from_model == args.model:
                    parts = os.path.splitext(os.path.basename(args.weights))[0].split('-')
                    args.from_epoch = int(parts[-1])
    
    if args.model is None:
        raise NotImplementedError('Missing argument: args.model')
    
    if args.from_epoch is None:
        args.from_epoch = 0
        
    if args.log_file is None:
        if args.phase == 'train':
            args.log_file = '{}.log'.format(args.model)
        elif args.phase == 'val':
            suffix = ''
            if args.split != 'val':
                suffix = '_{}'.format(args.split)
            args.log_file = '{}{}.log'.format(os.path.splitext(os.path.basename(args.weights))[0], suffix)
        else:
            raise NotImplementedError('Unknown phase: {}'.format(args.phase))
    
    model_specs = parse_model_label(args)
    if args.data_root is None:
        args.data_root = os.path.join('data', model_specs['dataset'])
        
    return args, model_specs


def _get_metric():
    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(mx.metric.CrossEntropy())
    eval_metric.add(mx.metric.Accuracy())
    eval_metric.add(mx.metric.TopKAccuracy(top_k=5))
    return eval_metric

def _get_scalemeanstd():
    if model_specs['net_type'] == 'rn':
        return -1, np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3)), None
    if model_specs['net_type'] in ('rna',):
        return (1.0/255,
            np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
            np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
    return None, None, None

def _get_module(args, model_specs, net=None):
    if net is None:
        # the following lines show how to create symbols for our networks
        if model_specs['net_type'] == 'rna':
            from util.symbol.symbol import cfg as symcfg
            symcfg['pool_top_infer_style'] = args.pool_top_infer_style
            if args.phase == 'val':
                symcfg['bn_use_global_stats'] = True
            if model_specs['net_name'] == 'a':
                #----
                # Now use args.pool_top_infer_style to do the trick.
                # Use 'caffe' when testing our pre-trained Model A,
                # otherwise, use `None'.
                #
                # Model A has down-sampling operations by pooling,
                # and it was trained using an old version of MXNet,
                # so the following line is required to reproduce our result,
                # by directly using our trained model.
                #symcfg['pool_top_infer_style'] == 'caffe'
                # When u train a new model from scratch,
                # leave this field as `None'.
                # When u tune our pre-trained model,
                # it also should be better to use `None'.
                # However, this is not empirically evaluated.
                #symcfg['pool_top_infer_style'] = None
                #----
                from util.symbol.resnet_v2 import rna_model_a
                net = rna_model_a(model_specs['classes'], model_specs['feat_stride'])
            elif model_specs['net_name'] == 'a1':
                # Model A1 has no down-sampling operation by pooling
                from util.symbol.resnet_v2 import rna_model_a1
                net = rna_model_a1(model_specs['classes'])
        if net is None:
            raise NotImplementedError('Unknown network: {}'.format(model_specs))
    contexts = [mx.gpu(int(_)) for _ in args.gpus.split(',')]
    mod = mx.mod.Module(net, logger=logger, context=contexts)
    return mod


def _train_impl(args, model_specs, logger):
    # dataiter
    scale_, mean_, std_ = _get_scalemeanstd()
    assert scale_ == 1./255
    pca = (np.array([0.2175, 0.0188, 0.0045]),
           np.array([[-0.5675, 0.7192, 0.4009],
                     [-0.5808, -0.0045, -0.814],
                     [-0.5836, -0.6948, 0.4203]]))
    crop_size = model_specs['crop_size']
    transformer = ts.Compose([ts.RandomSizedCrop(crop_size),
                              ts.ColorScale(np.single(1./255)),
                              ts.ColorJitter(crop_size, 0.4, 0.4, 0.4),
                              ts.Lighting(0.1, pca[0], pca[1]),
                              ts.Bound(lower=0., upper=1.),
                              ts.HorizontalFlip(),
                              ts.ColorNormalize(mean_, std_),])
    if model_specs['dataset'] == 'ilsvrc-cls':
        dataiter = FileIter(dataset=model_specs['dataset'],
                            split=args.split,
                            data_root=args.data_root,
                            sampler='random',
                            batch_images=model_specs['batch_images'],
                            transformer=transformer,
                            prefetch_threads=args.prefetch_threads,
                            prefetcher_type=args.prefetcher,)
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(model_specs['dataset']))
    dataiter.reset()
    # optimizer
    assert args.to_epoch is not None
    if args.stop_epoch is not None:
        assert args.stop_epoch > args.from_epoch and args.stop_epoch <= args.to_epoch
    else:
        args.stop_epoch = args.to_epoch
    from_iter = args.from_epoch * dataiter.batches_per_epoch
    to_iter = args.to_epoch * dataiter.batches_per_epoch
    lr_params = model_specs['lr_params']
    base_lr = lr_params['base']
    if lr_params['type'] == 'fixed':
        scheduler = FixedScheduler()
    elif lr_params['type'] == 'step':
        left_step = []
        for step in lr_params['args']['step']:
            if from_iter > step:
                base_lr *= lr_params['args']['factor']
                continue
            left_step.append(step - from_iter)
        model_specs['lr_params']['step'] = left_step
        scheduler = mx.lr_scheduler.MultiFactorScheduler(**lr_params['args'])
    elif lr_params['type'] == 'linear':
        scheduler = LinearScheduler(updates=to_iter+1, frequency=50,
                                    stop_lr=1e-6, offset=from_iter)
    optimizer_params = {
        'learning_rate': base_lr,
        'momentum': 0.9,
        'wd': 0.0001,
        'lr_scheduler': scheduler,
    }
    # initializer
    net_args = None
    net_auxs = None
    if args.weights is not None:
        net_args, net_auxs = mxutil.load_params_from_file(args.weights)
    initializer = mx.init.Mixed(
        ['linear.*', '.*',],
        [TorchXavier_Linear(rnd_type='uniform', factor_type='in', magnitude=1),
         mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2),]
    )
    # fit
    to_model = os.path.join(args.output, '{}_ep'.format(args.model))
    mod = _get_module(args, model_specs)
    mod.fit(
        dataiter,
        eval_metric=_get_metric(),
        batch_end_callback=mx.callback.Speedometer(dataiter.batch_size, 1),
        epoch_end_callback=mx.callback.do_checkpoint(to_model),
        kvstore=args.kvstore,
        optimizer='TorchNesterov',
        optimizer_params=optimizer_params,
        initializer=initializer,
        arg_params=net_args,
        aux_params=net_auxs,
        allow_missing=args.from_epoch == 0,
        begin_epoch=args.from_epoch,
        num_epoch=args.stop_epoch,
    )


#@profile
def _val_impl(args, model_specs, logger):
    assert args.prefetch_threads == 1
    assert args.weights is not None
    net_args, net_auxs = mxutil.load_params_from_file(args.weights)
    mod = _get_module(args, model_specs)
    has_gt = args.split in ('train', 'val',)
    scale_, mean_, std_ = _get_scalemeanstd()
    if args.test_scales is None:
        crop_sizes = [model_specs['crop_size']]
    else:
        crop_sizes = sorted([int(_) for _ in args.test_scales.split(',')])[::-1]
    
    batch_images = args.batch_images
    
    if has_gt:
        gt_labels = np.array(parse_split_file(model_specs['dataset'], args.split, args.data_root)[1])
    save_dir = os.path.join(args.output, os.path.splitext(args.log_file)[0])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    preds = []
    for crop_size in crop_sizes:
        save_path = os.path.join(save_dir, 'preds_sz{}'.format(crop_size))
        if os.path.isfile(save_path):
            logger.info('File %s exists, skipped crop size %d', save_path, crop_size)
            with open(save_path) as f:
                preds.append(cPickle.load(f))
            continue
        ts_list = [ts.Scale(crop_size),
                   ts.ThreeCrops(crop_size) if args.test_3crops else ts.CenterCrop(crop_size),]
        if scale_ > 0:
            ts_list.append(ts.ListInput(ts.ColorScale(np.single(scale_))))
        ts_list += [ts.ListInput(ts.ColorNormalize(mean_, std_))]
        transformer = ts.Compose(ts_list)
        dataiter = FileIter(dataset=model_specs['dataset'],
                            split=args.split,
                            data_root=args.data_root,
                            sampler='fixed',
                            has_gt=has_gt,
                            batch_images=batch_images,
                            transformer=transformer,
                            prefetch_threads=args.prefetch_threads,
                            prefetcher_type=args.prefetcher,)
        dataiter.reset()
        mod.bind(dataiter.provide_data, dataiter.provide_label, for_training=False, force_rebind=True)
        if not mod.params_initialized:
            mod.init_params(arg_params=net_args, aux_params=net_auxs)
        this_call_preds = []
        start = time.time()
        counter = [0, 0]
        for nbatch, batch in enumerate(dataiter):
            mod.forward(batch, is_train=False)
            outputs = mod.get_outputs()[0].asnumpy()
            outputs = outputs.reshape((batch_images, -1, model_specs['classes'])).mean(1)
            this_call_preds.append(outputs)
            if args.test_flipping:
                batch.data[0] = mx.nd.flip(batch.data[0], axis=3)
                mod.forward(batch, is_train=False)
                outputs = mod.get_outputs()[0].asnumpy()
                outputs = outputs.reshape((batch_images, -1, model_specs['classes'])).mean(1)
                this_call_preds[-1] = (this_call_preds[-1] + outputs) / 2
            score_str = ''
            if has_gt:
                counter[0] += batch_images
                counter[1] += (this_call_preds[-1].argmax(1) == gt_labels[nbatch*batch_images : (nbatch+1)*batch_images]).sum()
                score_str = ', Top1 {:.4f}%'.format(100.0*counter[1] / counter[0])
            logger.info('Crop size {}, done {}/{} at speed: {:.2f}/s{}'.\
                format(crop_size, nbatch+1, dataiter.batches_per_epoch, 1.*(nbatch+1)*batch_images / (time.time()-start), score_str))
        logger.info('Done crop size {} in {:.4f}s.'.format(crop_size, time.time() - start))
        this_call_preds = np.vstack(this_call_preds)
        with open(save_path, 'wb') as f:
            cPickle.dump(this_call_preds, f)
        preds.append(this_call_preds)
    for num_sizes in set((1, len(crop_sizes),)):
        for this_pred_inds in itertools.combinations(xrange(len(crop_sizes)), num_sizes):
            this_pred = np.mean([preds[_] for _ in this_pred_inds], axis=0)
            this_pred_label = this_pred.argsort(1)[:, -1 - np.arange(5)]
            logger.info('Done testing crop_size %s', [crop_sizes[_] for _ in this_pred_inds])
            if has_gt:
                top1 = 100. * (this_pred_label[:, 0] == gt_labels).sum() / gt_labels.size
                top5 = 100. * sum(map(lambda x, y: y in x.tolist(), this_pred_label, gt_labels)) / gt_labels.size
                logger.info('Top1 %.4f%%, Top5 %.4f%%', top1, top5)
            else:
                # TODO: Save predictions for submission
                raise NotImplementedError('Save predictions for submission')


if __name__ == '__main__':
    args, model_specs = parse_args()
    
    if len(args.output) > 0 and not osp.isdir(args.output):
        os.makedirs(args.output)
    
    util.cfg['choose_interpolation_method'] = args.choose_interpolation_method
    
    logger = util.set_logger(args.output, args.log_file, args.debug)
    logger.info('Run with arguments: %s', args)
    logger.info('and model specs: %s', model_specs)
    
    if args.phase == 'train':
        _train_impl(args, model_specs, logger)
    elif args.phase == 'val':
        _val_impl(args, model_specs, logger)
    else:
        raise NotImplementedError('Unknown phase: {}'.format(args.phase))

