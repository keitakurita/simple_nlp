import numpy as np
from torch.utils.data import DataLoader
import pytest
from simple_nlp import *

@pytest.fixture
def dataset():
    idxs = [
        [1, 2, 3],
        [4],
        [3, 1],
        [4, 4, 4, 4],
    ]
    return TextDataset(idxs, np.arange(len(idxs)))

def test_sampler(dataset):
    sampler = BucketBatchSampler(dataset, 1, noise=0.)
    assert sampler.argsort == [1, 2, 0, 3]

    # add noise that is too small to affect the ordering
    sampler = BucketBatchSampler(dataset, 1, noise=0.01)
    next(iter(sampler)) # compute argsort
    assert sampler.argsort == [1, 2, 0, 3]

def test_bs1(dataset):
    sampler = BucketBatchSampler(dataset, 1)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_collate)
    x, y = next(iter(loader))
    assert x.size(0) == 1
    assert y.size(0) == 1

def test_bs2(dataset):
    sampler = BucketBatchSampler(dataset, 2)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_collate)
    iter_ = iter(loader)
    x1, y1 = next(iter_)
    assert x1.size(0) == 2
    possible_seqlens = [4, 2]
    assert x1.size(1) in possible_seqlens
    possible_seqlens.remove(x1.size(1))
    x2, y2 = next(iter_)
    assert x2.size(1) in possible_seqlens
