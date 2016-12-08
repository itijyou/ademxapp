from symbol import relu, dropout, conv, bn


# conv stage for residual block v1
# conv->bn->relu
def conv_stage(data, name, filters, kernel=3, stride=1, dilate=1, pad=-1,
               groups=1, no_bias=False, has_bn=False, dropout_rate=0., has_relu=True):
    top = conv(data, name,
               filters,
               kernel=kernel,
               stride=stride,
               dilate=dilate,
               pad=pad,
               groups=groups,
               no_bias=no_bias)
    if has_bn:
        top = bn(top, name='{}_bn'.format(name))
    if dropout_rate > 0.0:
        top = dropout(top, '{}_do'.format(name), p=dropout_rate)
    if has_relu:
        top = relu(top, '{}_relu'.format(name))
    return top

def res_conv_stage(data, name, filters, kernel=3, stride=1, dilate=1, pad=-1,
               groups=1, no_bias=True, has_relu=True, dropout_rate=0.):
    top = conv(data, 'res{}'.format(name),
               filters,
               kernel=kernel,
               stride=stride,
               dilate=dilate,
               pad=pad,
               groups=groups,
               no_bias=no_bias)
    top = bn(data=top, name='bn{}'.format(name))
    if dropout_rate > 0.0:
        top = dropout(top, 'res{}_do'.format(name), p=dropout_rate)
    if has_relu:
        top = relu(top, 'res{}_relu'.format(name))
    return top

# residual block v1
# only apply down-sampling at the first stage (if there is any)
def bottleneck_block(data, name, filters, stride=1, dilate=1,
                     identity_map=True, shortcut_kernel=1, dropout_rate=0.):
    branch2a = res_conv_stage(data, '{}_branch2a'.format(name),
                              filters,
                              kernel=1,
                              stride=stride)
    branch2b = res_conv_stage(branch2a, '{}_branch2b'.format(name),
                              filters,
                              kernel=3,
                              dilate=dilate)
    branch2c = res_conv_stage(branch2b, '{}_branch2c'.format(name),
                              filters*4,
                              kernel=1,
                              has_relu=False,
                              dropout_rate=dropout_rate)
    if identity_map:
        return relu(data + branch2c, 'res{}_relu'.format(name))
    else:
        assert shortcut_kernel in (1, 2)
        branch1 = res_conv_stage(data, '{}_branch1'.format(name),
                                 filters*4,
                                 kernel=shortcut_kernel,
                                 stride=stride,
                                 dilate=dilate,
                                 pad=-1 if shortcut_kernel % 2 else 0,
                                 has_relu=False)
        return relu(branch1 + branch2c, 'res{}_relu'.format(name))

