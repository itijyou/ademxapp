"""
extended initialization helper
"""
import numpy as np

import mxnet as mx
from mxnet import random


class TorchXavier_Linear(mx.initializer.Xavier):
    """Initialize the weight with Xavier or similar initialization scheme
    for the top-most linear layer.

    Parameters
    ----------
    rnd_type: str, optional
        Use ```gaussian``` or ```uniform``` to init

    factor_type: str, optional
        Use ```avg```, ```in```, or ```out``` to init

    magnitude: float, optional
        scale of random number range
    """
    def __init__(self, rnd_type='uniform', factor_type='avg', magnitude=3):
        super(TorchXavier_Linear, self).__init__(rnd_type, factor_type, magnitude)
        self._layers = {}

    def _init_impl(self, scale, out):
        if self.rnd_type == 'uniform':
            random.uniform(-scale, scale, out=out)
        elif self.rnd_type == 'gaussian':
            random.normal(0, scale, out=out)
        else:
            raise ValueError('Unknown random type')

    def _init_bias(self, name, arr):
        layer_name = name[:-len('_bias')]
        if layer_name in self._layers:
            scale = self._layers[layer_name]['scale']
            self._init_impl(scale, arr)
        else:
            self._layers[layer_name] = {'arr': arr}

    def _init_weight(self, name, arr):
        layer_name = name[:-len('_weight')]
        shape = arr.shape
        hw_scale = 1.
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.
        if self.factor_type == 'avg':
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == 'in':
            factor = fan_in
        elif self.factor_type == 'out':
            factor = fan_out
        else:
            raise ValueError('Incorrect factor type')
        scale = np.sqrt(self.magnitude / factor)
        self._init_impl(scale, arr)
        
        if layer_name not in self._layers:
            self._layers[layer_name] = {'scale': scale}
        else:
            assert 'scale' not in self._layers[layer_name]
            self._layers[layer_name]['scale'] = scale
            self._init_impl(scale, self._layers[layer_name]['arr'])

