"""
softmax with bootstrapping
"""
import heapq
import logging
from functools import partial

import numpy as np

import mxnet as mx


class OhemSoftmax(mx.operator.CustomOp):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, margin=-1., **kwargs):
        super(OhemSoftmax, self).__init__()
        self._ignore_label = int(ignore_label)
        self._thresh = float(thresh)
        self._min_kept = int(min_kept)
        self._margin = float(margin)
    
    def forward(self, is_train, req, in_data, out_data, aux):
        in_shape = list(in_data[0].shape)
        comp_shape = [in_shape[1]] + [in_shape[0]] + in_shape[2:]
        num_classes = in_shape[1]
        
        x = np.rollaxis(in_data[0].asnumpy().astype(np.double), 1).reshape((num_classes, -1))
        y = np.exp(x - x.max(axis=0).reshape((1, -1)))
        y /= y.sum(axis=0).reshape((1, -1))
        y = np.rollaxis(y.reshape(comp_shape), 1)
        self.assign(out_data[0], req[0], mx.nd.array(y.astype(np.single)))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        in_shape = list(in_data[0].shape)
        comp_shape = [in_shape[1]] + [in_shape[0]] + in_shape[2:]
        num_classes = in_shape[1]
        
        input_label = in_data[1].asnumpy().ravel().astype(np.int32)
        input_prob = np.rollaxis(out_data[0].asnumpy(), 1).reshape((num_classes, -1))
        
        valid_flag = input_label != self._ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self._min_kept >= num_valid:
            logging.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self._thresh
            if self._min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self._min_kept) - 1 ]
                if pred[threshold_index] > self._thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            pred = pred[kept_flag]
            valid_inds = valid_inds[kept_flag]
            label = label[kept_flag]
            margin_threshold = self._margin
            if self._margin > 0:
                top2 = np.array(map(partial(heapq.nlargest, 2), input_prob[:, valid_inds].T))
                margin = pred - top2[:, 1]
                if self._min_kept > 0:
                    index = margin.argsort()
                    margin_index = index[ min(len(index), self._min_kept) - 1 ]
                    if margin[margin_index] > self._margin:
                        margin_threshold = margin[margin_index]
                kept_flag = margin <= margin_threshold
                valid_inds = valid_inds[kept_flag]
                label = label[kept_flag]
            logging.info('Labels: {} {} {}'.format(len(label), threshold, margin_threshold))
        
        y = np.zeros_like(input_prob)
        y[:, valid_inds] = input_prob[:, valid_inds]
        y[label, valid_inds] += -1.
        y *= 1. / max(1, len(label))
        y = np.rollaxis(y.reshape(comp_shape), 1)
        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register('ohem_softmax')
class OhemSoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(OhemSoftmaxProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs.copy()
    
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return OhemSoftmax(**self._kwargs)

