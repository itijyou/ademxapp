"""
extened data io
"""
from multiprocessing import Process, Queue
from multiprocessing.dummy import Process as Thread, Queue as ThreadQueue

import numpy as np

from util import as_list


class BatchFetcherGroup(object):
    def __init__(self,
                 dataiter,
                 sampler,
                 batch_size,
                 threads,
                 queue_size,
                 fetcher_type='thread'):
        self._dataiter = dataiter
        self._sampler = sampler
        self._batch_size = batch_size
        self._threads = threads
        
        creators = {'process': [Queue, BatchFetcherProcess],
                    'thread': [ThreadQueue, BatchFetcherThread],}[fetcher_type]
        self._queue = creators[0](queue_size)
        self._fetcher_thread_creator = creators[1]
        
        self._procs = []
        self._running = False
        
    def reset(self):
        assert not self._running
        for proc in self._procs:
            proc.join()
        self._procs = []
        self._cursor = -self._batch_size
        
        perm = self._sampler()
        self._perm_len = len(perm)
        
        threads = self._threads
        num_batches = len(perm) // self._batch_size
        batch_splits = [1. * (num_batches - _) / threads for _ in xrange(threads)]
        batch_splits = np.ceil(batch_splits).astype(np.int32) * self._batch_size
        for tid in xrange(self._threads):
            left = batch_splits[:tid].sum()
            thread_perm = perm[left:left+batch_splits[tid]].reshape((-1, self._batch_size))
            fetcher = BatchFetcher(self._queue,
                                   self._dataiter,
                                   thread_perm)
            proc = self._fetcher_thread_creator(fetcher)
            proc.daemon = True
            proc.start()
            self._procs.append(proc)
        self._running = True
    
    def iter_next(self):
        self._cursor += self._batch_size
        self._running = self._cursor + self._batch_size <= self._perm_len
        return self._running
    
    def get(self):
        return self._queue.get()
        
class BatchFetcher(object):
    def __init__(self, queue, dataiter, perm):
        self._queue = queue
        self._dataiter = dataiter
        self._perm = perm
        self._perm_len = len(perm)
        self._cursor = -1
        
    def iter_next(self):
        self._cursor += 1
        return self._cursor + 1 <= self._perm_len
    
    def run(self):
        while self.iter_next():
            db_inds = self._perm[self._cursor]
            for datum in as_list(self._dataiter.read(db_inds)):
                self._queue.put(datum)

class BatchFetcherProcess(Process):
    def __init__(self, fetcher):
        super(BatchFetcherProcess, self).__init__()
        self._fetcher = fetcher

    def run(self):
        self._fetcher.run()
        
class BatchFetcherThread(Thread):
    def __init__(self, fetcher):
        super(BatchFetcherThread, self).__init__()
        self._fetcher = fetcher

    def run(self):
        self._fetcher.run()

