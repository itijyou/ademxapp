"""
extended optimization algorithms
"""

import mxnet as mx

@mx.optimizer.Optimizer.register
class TorchNesterov(mx.optimizer.SGD):

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom *= self.momentum
            grad += wd * weight
            mom += grad
            grad += self.momentum * mom
            weight += -lr * grad
        else:
            assert self.momentum == 0.0
            weight += -lr * (grad + self.wd * weight)

