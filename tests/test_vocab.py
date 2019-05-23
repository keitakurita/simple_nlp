import numpy as np
import pytest
from simple_nlp import *

@pytest.fixture
def texts():
    return [
        "this is a sentence",
        "a sentence this is",
        "all unique words",
        "a bunch of tokens",
        "a bunch of flowers",
    ]

def test_basic(texts):
    uniq_words = set()
    for text in texts:
        for word in text.split():
            uniq_words.add(word)

    vocab = Vocab()
    X = vocab.fit_transform(texts)
    assert len(X) == len(texts)
    assert len(vocab.stoi) == len(uniq_words)
    for x in X:
        assert 0 not in x
        assert 1 not in x

def test_min_freq(texts):
    vocab = Vocab(min_freq=2, keep_freq=True)
    X = vocab.fit_transform(texts)
    assert len(vocab.stoi) == 6
    for x in X:
        assert 0 not in x
    assert X[2] == [1, 1, 1]
    assert X[3][-1] == 1
    assert X[4][-1] == 1

def text_max_size(texts):
    vocab = Vocab(max_size=8)
    X = vocab.fit_transform(texts)
    assert len(vocab.stoi) == 6
    for x in X:
        assert 0 not in x
    assert X[2] == [1, 1, 1]
    assert X[3][-1] == 1
    assert X[4][-1] == 1

def test_prebuilt(texts):
    vocab = Vocab(prebuilt_stoi={"this": 2, "is": 3, "a": 4})
    X = vocab.fit_transform(texts)
    assert X[0] == [2, 3, 4, 1]
    assert X[1] == [4, 1, 2, 3]
    assert X[2] == [1, 1, 1]
