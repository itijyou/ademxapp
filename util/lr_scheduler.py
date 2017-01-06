"""
extended learning rate scheduler,
which adaptive changes the learning rate based on the progress
"""
import logging

import mxnet as mx


class FixedScheduler(mx.lr_scheduler.LRScheduler):
    def __call__(self, num_update):
        return self.base_lr


class LinearScheduler(mx.lr_scheduler.LRScheduler):
    """Reduce learning rate linearly

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * (1 - n/iters)

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, updates, frequency=0, stop_lr=-1., offset=0):
        super(LinearScheduler, self).__init__()
        if updates < 1:
            raise ValueError('Schedule required max number of updates to be greater than 1 round')
        self._updates = updates
        self._frequency = frequency
        self._stop_lr = stop_lr
        self._offset = offset
        
        self._pre_updates = -1

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        now_update = self._offset + num_update
        if now_update > self._updates:
            if self._pre_updates != num_update:
                print 'Exceeds the number of updates, {} > {}'.format(now_update, self._updates)
                self._pre_updates = num_update
            now_update = self._updates
        lr = self.base_lr * (1 - float(now_update) / self._updates)
        if self._stop_lr > 0. and lr < self._stop_lr:
            lr = self._stop_lr
        if self._frequency > 0 and num_update % self._frequency == 0 and self._pre_updates != num_update:
            logging.info('Update[%d]: Current learning rate is %0.5e',
                         num_update, lr)
            self._pre_updates = num_update
        return lr

