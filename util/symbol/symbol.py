"""
handy wraps of mxnet symbols
"""
import mxnet as mx


cfg = {}


def _attr_scope_lr(lr_type, lr_owner):
    assert lr_type in ('alex', 'alex10', 'torch')
    # weight (lr_mult, wd_mult); bias;
    # 1, 1; 2, 0;
    if lr_type == 'alex':
        if lr_owner == 'weight':
            return mx.AttrScope()
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='2.', wd_mult='0.')
        else:
            assert False
    # 10, 1; 20, 0;
    if lr_type == 'alex10':
        if lr_owner == 'weight':
            return mx.AttrScope(lr_mult='10.', wd_mult='1.')
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='20.', wd_mult='0.')
        else:
            assert False
    # 0, 0; 0, 0;
    # so apply this to both
    if lr_type == 'fixed':
        assert lr_owner in ('weight', 'bias')
        return mx.AttrScope(lr_mult='0.', wd_mult='0.')
    # 1, 1; 1, 1;
    # so do nothing
    return mx.AttrScope()


def relu(data, name):
    return mx.sym.Activation(data, name=name, act_type='relu')

def lrelu(data, name, slope=0.25):
    return mx.sym.LeakyReLU(data, name=name, act_type='leaky', slope=slope)

def dropout(data, name, p=0.5):
    return mx.sym.Dropout(data, name=name, p=p)

def conv(data, name, filters, kernel=3, stride=1, dilate=1, pad=-1,
         groups=1, no_bias=False, workspace=-1):
    if kernel == 1:
        # set dilate to 1, since kernel is 1
        dilate = 1
    if pad < 0:
        assert kernel % 2 == 1, 'Specify pad for an even kernel size'
        pad = ((kernel - 1) * dilate + 1) // 2
    if workspace < 0:
        workspace = cfg.get('workspace', 512)
    lr_type = cfg.get('lr_type', 'torch')
    with _attr_scope_lr(lr_type, 'weight'):
        weight = mx.sym.Variable('{}_weight'.format(name))
    if no_bias:
        return mx.sym.Convolution(data=data, weight=weight, name=name,
                                  kernel=(kernel, kernel),
                                  stride=(stride, stride),
                                  dilate=(dilate, dilate),
                                  pad=(pad, pad),
                                  num_filter=filters,
                                  num_group=groups,
                                  workspace=workspace,
                                  no_bias=True)
    else:
        with _attr_scope_lr(lr_type, 'bias'):
            bias = mx.sym.Variable('{}_bias'.format(name))
        return mx.sym.Convolution(data=data, weight=weight, bias=bias, name=name,
                                  kernel=(kernel, kernel),
                                  stride=(stride, stride),
                                  dilate=(dilate, dilate),
                                  pad=(pad, pad),
                                  num_filter=filters,
                                  num_group=groups,
                                  workspace=workspace,
                                  no_bias=False)

def bn(data, name, eps=1.001e-5, fix_gamma=False, use_global_stats=None):
    if use_global_stats is None:
        use_global_stats = cfg.get('bn_use_global_stats', False)
    
    if fix_gamma:
        with mx.AttrScope(lr_mult='0.', wd_mult='0.'):
            gamma = mx.sym.Variable('{}_gamma'.format(name))
            beta = mx.sym.Variable('{}_beta'.format(name))
        return mx.sym.BatchNorm(data=data, gamma=gamma, beta=beta, name=name,
                                eps=eps,
                                fix_gamma=True,
                                use_global_stats=use_global_stats)
    else:
        lr_type = cfg.get('lr_type', 'torch')
        with _attr_scope_lr(lr_type, 'weight'):
            gamma = mx.sym.Variable('{}_gamma'.format(name))
        with _attr_scope_lr(lr_type, 'bias'):
            beta = mx.sym.Variable('{}_beta'.format(name))
        return mx.sym.BatchNorm(data=data, gamma=gamma, beta=beta, name=name,
                                eps=eps,
                                fix_gamma=False,
                                use_global_stats=use_global_stats)

def pool(data, name, kernel=3, stride=2, dilate=1, pad=-1, pool_type='max', global_pool=False):
    if pool_type == 'max+avg':
        branch1 = pool(data, '{}_branch1'.format(name),
                       kernel=kernel,
                       stride=stride,
                       dilate=dilate,
                       pad=pad,
                       pool_type='max')
        branch2 = pool(data, '{}_branch2'.format(name),
                       kernel=kernel,
                       stride=stride,
                       dilate=dilate,
                       pad=pad,
                       pool_type='avg')
        return branch1 + branch2
    if kernel == 1:
        assert dilate == 1
    if global_pool:
        assert dilate == 1
        assert pad < 0
        return mx.sym.Pooling(data, name=name,
                              kernel=(1, 1),
                              pool_type=pool_type,
                              global_pool=True)
    else:
        if pad < 0:
            if cfg.get('pool_top_infer_style', None) == 'caffe':
                pad = 0
            else:
                assert kernel % 2 == 1, 'Specify pad for an even kernel size'
                pad = ((kernel - 1) * dilate + 1) // 2
        if dilate == 1:
            return mx.sym.Pooling(data, name=name,
                                  kernel=(kernel, kernel),
                                  stride=(stride, stride),
                                  pad=(pad, pad),
                                  pool_type=pool_type)
        else:
            # TODO: not checked for stride > 1
            assert stride == 1
            return mx.sym.Pooling(data, name=name,
                                  kernel=(kernel, kernel),
                                  stride=(stride, stride),
                                  dilate=(dilate, dilate),
                                  pad=(pad, pad),
                                  pool_type=pool_type)

def fc(data, name, hiddens, no_bias=False):
    lr_type = cfg.get('lr_type', 'torch')
    with _attr_scope_lr(lr_type, 'weight'):
        weight = mx.sym.Variable('{}_weight'.format(name))
    if no_bias:
        return mx.sym.FullyConnected(data=data, weight=weight, name=name,
                                     num_hidden=hiddens,
                                     no_bias=True)
    else:
        with _attr_scope_lr(lr_type, 'bias'):
            bias = mx.sym.Variable('{}_bias'.format(name))
        return mx.sym.FullyConnected(data=data, weight=weight, bias=bias, name=name,
                                     num_hidden=hiddens,
                                     no_bias=False)

def softmax_out(data, grad_scale=1.0, multi_output=False):
    if multi_output:
        return mx.sym.SoftmaxOutput(data, name='softmax',
                                    grad_scale=grad_scale,
                                    use_ignore=True,
                                    ignore_label=255,
                                    multi_output=True,
                                    normalization='valid')
    else:
        return mx.sym.SoftmaxOutput(data, name='softmax',
                                    grad_scale=grad_scale,
                                    multi_output=False)

