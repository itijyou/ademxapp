"""
utilities for convenience
"""
import os.path as osp

import mxnet as mx


def load_params(prefix, epoch):
    """Load model checkpoint from file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - parameters will be loaded from ``prefix-epoch.params``.
    """
    path = '%s-%04d.params' % (prefix, epoch)
    arg_params, aux_params = load_params_from_file(path)
    symbol = None
    symbol_path = '%s-symbol.json' % prefix
    if osp.isfile(symbol_path):
        symbol = mx.sym.load(symbol_path)
    return symbol, arg_params, aux_params

def load_params_from_file(path):
    """Load model checkpoint from file."""

    save_dict = mx.nd.load(path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

