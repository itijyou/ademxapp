"""
training data samplers
"""
import numpy as np
import numpy.random as npr


class FixedSampler(object):
    def __init__(self, perm_len):
        assert perm_len > 0
        self._perm_len = perm_len
    
    def __call__(self, perm_len=None):
        perm_len = self._perm_len if perm_len is None else perm_len
        return np.arange(perm_len)

class RandomSampler(object):
    def __init__(self, perm_len):
        assert perm_len > 0
        self._perm_len = perm_len
    
    def __call__(self, perm_len=None):
        perm_len = self._perm_len if perm_len is None else perm_len
        return npr.permutation(perm_len)

class BalancedSampler_OneClassPerImage(object):
    def __init__(self, perm_len, num_c, x2c):
        assert perm_len > 0
        self._perm_len = perm_len
        self._num_c = num_c
        self._x2c = np.array(x2c, np.int32)
        
        self._c2x = []
        for i in xrange(self._num_c):
            self._c2x.append(np.where(self._x2c == i)[0])
        self._cur_c = -1
        self._cls = npr.permutation(self._num_c).tolist()
        self._cur_x = [-1] * self._num_c
        for i in xrange(self._num_c):
            npr.shuffle(self._c2x[i])
    
    def _next_c(self):
        self._cur_c += 1
        if self._cur_c == self._num_c:
            npr.shuffle(self._cls)
            self._cur_c = 0
        return self._cls[self._cur_c]
    
    def _next_x(self, ind_c):
        self._cur_x[ind_c] += 1
        if self._cur_x[ind_c] == len(self._c2x[ind_c]):
            npr.shuffle(self._c2x[ind_c])
            self._cur_x[ind_c] = 0
        return self._c2x[ind_c][self._cur_x[ind_c]]
    
    def _next(self):
        return self._next_x(self._next_c())
    
    def __call__(self, perm_len=None):
        perm_len = self._perm_len if perm_len is None else perm_len
        return np.array([self._next() for _ in xrange(perm_len)])

