import collections

import mxnet as mx

from symbol import relu, dropout, conv, bn, pool, fc, softmax_out
from resnet_v1 import conv_stage as conv_state_v1


# conv stage for residual block v2
# bn -> relu -> conv
def conv_stage(data, names, filters, kernel=3, stride=1, dilate=1, pad=-1,
               groups=1, no_bias=True, dropout_rate=0.):
    i = 0
    bn1 = bn(data, names[i])
    i += 1
    relu1 = relu(bn1, names[i])
    i += 1
    if dropout_rate > 0.:
        dropout1 = dropout(relu1, names[i], p=dropout_rate)
    i += 1
    top = conv(dropout1 if dropout_rate > 0. else relu1, names[i],
               filters, kernel=kernel,
               stride=stride,
               dilate=dilate,
               pad=pad,
               groups=groups,
               no_bias=no_bias)
    return relu1, top

def res_conv_stage(data, name, filters, kernel=3, stride=1, dilate=1, pad=-1,
                   groups=1, no_bias=True, dropout_rate=0.):
    names = ['bn{}'.format(name),
             'res{}_relu'.format(name),
             'res{}_do'.format(name),
             'res{}'.format(name),]
    return conv_stage(data, names, filters, kernel, stride, dilate, pad,
                      groups, no_bias, dropout_rate)

# residual block v2
# only apply down-sampling at the first stage (if there is any)
def res_block(data, name, filters, kernel=3, stride=1, dilate=1, inc_dilate=False,
              identity_map=True, shortcut_kernel=1, dropout_rate=0.):
    if not isinstance(filters, list):
        filters = [filters, filters]
    if not isinstance(kernel, list):
        kernel = [kernel] * len(filters)
    assert len(filters) == len(kernel)
    
    dilate_factor = 1
    if inc_dilate:
        assert stride > 1
        dilate_factor = stride
        stride = 1
    
    block, branch2 = res_conv_stage(data, '{}_branch2a'.format(name),
                                    filters[0],
                                    kernel=kernel[0],
                                    stride=stride,
                                    dilate=dilate)
    for i in xrange(1, len(filters)):
        this_dropout_rate = dropout_rate if i == len(filters)-1 else 0.
        _, branch2 = res_conv_stage(branch2, '{}_branch2b{}'.format(name, i),
                                    filters[i],
                                    kernel=kernel[i],
                                    dilate=dilate*dilate_factor,
                                    dropout_rate=this_dropout_rate)
    if identity_map:
        return data + branch2
    else:
        assert shortcut_kernel in (1, 2)
        branch1 = conv(block, 'res{}_branch1'.format(name),
                       filters[-1],
                       kernel=shortcut_kernel,
                       stride=stride,
                       dilate=dilate,
                       pad=-1 if shortcut_kernel % 2 else 0,
                       no_bias=True)
        return branch1 + branch2


def rna_feat(conv1_layers, level_blocks):
    '''RNA features'''
    
    def _widen(filters, max_filters, min_filters=-1):
        filters = min(max_filters, int(filters * level_blocks[level].width))
        if min_filters > 0:
            filters = max(min_filters, filters)
        return filters
    
    # TODO: these may only work for b33
    def _ds_by_pl(data, name, level, dilate):
        assert level > 0
        level_block = level_blocks[level]
        if not level_block.downsample == 'p':
            return data, dilate
        pool_stride = stride = 2
        inc_dilate = level_block.dilate
        pad = -1
        if inc_dilate:
            pool_stride = 1
            pad = dilate
        print 'Pooling stride: {}, dilate: {}, pad: {}'.format(pool_stride, dilate, pad)
        top = pool(data, name, stride=pool_stride, dilate=dilate, pad=pad, pool_type=pool_type)
        if inc_dilate:
            dilate *= stride
        return top, dilate
    def _ds_by_cv(data, name, filters, level, dilate, kernel=3, dropout_rate=0.):
        assert level > 1
        level_block = level_blocks[level-1]
        if not level_block.downsample == 'c':
            print 'First block on level {}, dilate: {}'.format(level, dilate)
            top = res_block(data, name, filters, kernel=kernel,
                            dilate=dilate, identity_map=False,
                            dropout_rate=dropout_rate)
        else:
            stride = 2
            inc_dilate = level_block.dilate
            print 'First block on level {}, stride: {}, dilate: {}'.format(level, stride, dilate)
            top = res_block(data, name, filters, kernel=kernel, stride=stride,
                            dilate=dilate, inc_dilate=inc_dilate, identity_map=False,
                            dropout_rate=dropout_rate)
            if inc_dilate:
                dilate *= stride
        return top, dilate
    
    dilate = 1
    pool_type = 'max'
    crop_size = 224
    
    data = mx.sym.Variable('data')
    # 224^2 3
    
    level = 0; print 'Level {}'.format(level)
    conv0 = data
    # 224^2 3
    print conv0.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    level = 1; print 'Level {}'.format(level)
    filters = _widen(64, 1024)
    res1 = conv(conv0, 'conv1a',
                filters,
                kernel=conv1_layers[0].kernel,
                stride=1,
                no_bias=True)
    for i, conv1_layer in enumerate(conv1_layers[1:]):
        names = ['conv1b{}_bn'.format(i),
                 'conv1b{}_relu'.format(i),
                 'conv1b{}_do'.format(i),
                 'conv1b{}'.format(i),]
        res1 = conv_stage(res1, names,
                          filters,
                          kernel=conv1_layer.kernel,
                          stride=1,
                          no_bias=True)
    for i in xrange(level_blocks[level].num):
        res1 = res_block(res1, '1b{}'.format(i+1), filters)
    # pass dilate from now on
    assert dilate == 1
    res1, dilate = _ds_by_pl(res1, 'pool1', level, dilate)
    # 112^2 64
    print res1.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    level = 2; print 'Level {}'.format(level)
    filters = _widen(128, 1024)
    res2, dilate = _ds_by_cv(res1, '2a', filters, level, dilate)
    for i in xrange(1, level_blocks[level].num):
        res2 = res_block(res2, '2b{}'.format(i), filters, dilate=dilate)
    res2, dilate = _ds_by_pl(res2, 'pool2', level, dilate)
    # 56^2 128
    print res2.infer_shape(data=(64, 3, crop_size, crop_size))[1]

    level = 3; print 'Level {}'.format(level)
    filters = _widen(256, 1024)
    res3, dilate = _ds_by_cv(res2, '3a', filters, level, dilate)
    for i in xrange(1, level_blocks[level].num):
        res3 = res_block(res3, '3b{}'.format(i), filters, dilate=dilate)
    res3, dilate = _ds_by_pl(res3, 'pool3', level, dilate)
    # 28^2 256
    print res3.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    level = 4; print 'Level {}'.format(level)
    filters = _widen(512, 1024)
    res4, dilate = _ds_by_cv(res3, '4a', filters, level, dilate)
    for i in xrange(1, level_blocks[level].num):
        res4 = res_block(res4, '4b{}'.format(i), filters, dilate=dilate)
    res4, dilate = _ds_by_pl(res4, 'pool4', level, dilate)
    # 14^2 512
    print res4.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    level = 5; print 'Level {}'.format(level)
    # wd = 0.5 : (512, 512)
    # wd = 1.0 : (512, 1024)
    # wd = 2.0 : (1024, 1024)
    filters_l1 = _widen(512, 1024, filters)
    filters = [filters_l1, _widen(1024, 1024, filters_l1)]
    res5, dilate = _ds_by_cv(res4, '5a', filters, level, dilate)
    for i in xrange(1, level_blocks[level].num):
        res5 = res_block(res5, '5b{}'.format(i), filters, dilate=dilate)
    res5, dilate = _ds_by_pl(res5, 'pool5', level, dilate)
    # 7^2 1024
    print res5.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    level = 6; print 'Level {}'.format(level)
    res6, dilate = _ds_by_cv(res5, '6a', [512, 1024, 2048],
                             level, dilate, kernel=[1, 3, 1],
                             dropout_rate=level_blocks[level].dropout)
    # 7^2 2048
    print res6.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    level = 7; print 'Level {}'.format(level)
    res7 = res_block(res6, '7a', [1024, 2048, 4096], kernel=[1, 3, 1],
                     dilate=dilate, identity_map=False,
                     dropout_rate=level_blocks[level].dropout)
    # 7^2 4096
    print res7.infer_shape(data=(64, 3, crop_size, crop_size))[1]
    
    bn7 = bn(res7, 'bn7')
    relu7 = relu(bn7, 'relu7')
    return relu7


def rn_top(feat, fc_name, classes):
    pool7 = pool(feat, 'pool7', pool_type='avg', global_pool=True)
    scores = fc(pool7, fc_name, classes)
    return softmax_out(scores)


def fcn_top(feat, classifier, fc_name):
    top = feat
    for j, layer in enumerate(classifier[:-1]):
        # This naming (conv6) is derived from the ResNets (with five levels),
        # which is not accurate for our networks (with seven levels).
        top = conv_state_v1(top, 'conv6{}'.format(chr(j+97)),
                            layer.channels,
                            kernel=layer.kernel,
                            dilate=layer.dilate,
                            dropout_rate=0.)
    layer = classifier[-1]
    scores = conv(top, fc_name,
                  layer.channels,
                  kernel=layer.kernel,
                  dilate=layer.dilate)
    return softmax_out(scores, multi_output=True)


ConvStage = collections.namedtuple('ConvStage',
                                   ['channels', 'kernel', 'dilate'])

LevelBlock = collections.namedtuple('LevelBlock',
                                    ['type', 'num', 'width',
                                     'downsample', 'dilate',
                                     'dropout'])


def rna_feat_a(inv_resolution=32):
    assert inv_resolution in (16, 32)
    '''RNA features Model A'''
    conv1_layers = [ConvStage(1000, 3, 1),]
    if inv_resolution == 32:
        level_blocks = [None,
                        LevelBlock('b33', 0, 1., 'p', False, 0.),
                        LevelBlock('b33', 3, 1., 'p', False, 0.),
                        LevelBlock('b33', 3, 1., 'p', False, 0.),
                        LevelBlock('b33', 6, 1., 'p', False, 0.),
                        LevelBlock('b33', 3, 1., 'p', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),]
    elif inv_resolution == 16:
        level_blocks = [None,
                        LevelBlock('b33', 0, 1., 'p', False, 0.),
                        LevelBlock('b33', 3, 1., 'p', False, 0.),
                        LevelBlock('b33', 3, 1., 'p', False, 0.),
                        LevelBlock('b33', 6, 1., 'p', False, 0.),
                        LevelBlock('b33', 3, 1., 'p', True, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),]
    return rna_feat(conv1_layers, level_blocks)

def rna_model_a(classes, inv_resolution=32):
    '''RNA Model A'''
    feat = rna_feat_a(inv_resolution)
    return rn_top(feat, 'linear{}'.format(classes), classes)


def rna_feat_a1(inv_resolution=32):
    assert inv_resolution in (8, 16, 32)
    '''RNA features Model A1'''
    conv1_layers = [ConvStage(1000, 3, 1),]
    if inv_resolution == 32:
        level_blocks = [None,
                        LevelBlock('b33', 0, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b33', 6, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),]
    elif inv_resolution == 16:
        level_blocks = [None,
                        LevelBlock('b33', 0, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b33', 6, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', True, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),]
    elif inv_resolution == 8:
        level_blocks = [None,
                        LevelBlock('b33', 0, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b33', 3, 1., 'c', False, 0.),
                        LevelBlock('b33', 6, 1., 'c', True, 0.),
                        LevelBlock('b33', 3, 1., 'c', True, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),
                        LevelBlock('b131', 1, 1., 'n', False, 0.),]
    return rna_feat(conv1_layers, level_blocks)

def rna_model_a1(classes):
    '''RNA Model A1'''
    feat = rna_feat_a1()
    return rn_top(feat, 'linear{}'.format(classes), classes)

def fcrna_model_a1(classes, inv_resolution=8):
    '''FCRNA Model A1'''
    feat = rna_feat_a1(inv_resolution)
    classifier = [ConvStage(512, 3, 12),
                  ConvStage(classes, 3, 12),]
    return fcn_top(feat, classifier, 'linear{}'.format(classes))

