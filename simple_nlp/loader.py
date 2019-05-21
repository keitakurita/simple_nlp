import numpy as np
import random
import math
from typing import *
import torch
from torch.utils.data import BatchSampler

class BucketBatchSampler(BatchSampler):
    """
    A sampler that buckets sequences of similar lengths together for efficient processing.

    noise: Sometimes bucketing can diminsh performance because
        the same samples continue to be batched together (this is particularly
        problematic when your target variable is related to the length of the sequence).
        This option adds random noise to the lengths of the sequences when sorting to
        make batching non-deterministic.
    """
    def __init__(self, data_source: List[Tuple[List[int], Any]], batch_size: int,
                 noise: float=0.):
        self.data_source = data_source
        self.batch_size = batch_size
        self.noise = noise
        if self.noise == 0.:
            # compute once
            self.argsort = self._compute_argsort()
        else:
            self.argsort = None # compute during iteration

    def _compute_argsort(self):
        return [i for (i, x) in
                sorted(enumerate(self.data_source),
                                 key=self._get_length)]

    @property
    def num_batches(self):
        return math.ceil(len(self.data_source) / self.batch_size)

    def _get_length(self, x: Tuple[int, List[str]]):
        # x: (i, (X, y))
        # e.g. (3, ([1, 2, 3], 0.3)
        return len(x[1][0]) + random.random() * self.noise

    def __iter__(self) -> Iterable[List[int]]:
        batch_idxs = list(range(self.num_batches))
        random.shuffle(batch_idxs)
        if self.argsort is None:
            self.argsort = self._compute_argsort()
        for idx in batch_idxs:
            yield self.argsort[idx * self.batch_size:(idx+1) * self.batch_size]

    def __len__(self):
        return len(self.data_source)

def tensor_collate(x, dtype=np.float32):
    return torch.from_numpy(np.array(x, dtype=dtype))

def pad_collate(samples: Iterable[Tuple[List[int], Any]], pad_first: bool=True,
                padding_idx: int=0, y_collate_fn=tensor_collate):
    """
    Pads sequences of different lengths into a single tensor of shape
    (batch_size, max_seq_len)
    """
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + padding_idx
    for i,s in enumerate(samples):
        if pad_first: res[i,-len(s[0]):] = torch.LongTensor(s[0])
        else:         res[i,:len(s[0]):] = torch.LongTensor(s[0])
    return res, y_collate_fn([s[1] for s in samples])
