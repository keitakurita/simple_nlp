from typing import *
from collections import Counter

class DefaultDict(dict):
    """
    A dictionary that, when an unknown key is accessed,
    returns a default value but *does not register that key*.
    This simplifies accessing in the vocab and also ensures
    the length of the dictionary is always equivalent to the vocab size.
    This also saves memory.
    """
    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return self.default

class Vocab:
    """
    Handles the conversion of raw text to integers to be fed to a model.

    min_freq: Minimum frequency required for a word to be registered in the vocab.
    tokenizer: Function that converts a chunk of text into tokens.
        All preprocessing (e.g. lowercasing, limiting the length of the sequence)
        should be done in the tokenizer. This makes reasoning about the preprocessing
        much easier.
    keep_freq: If True, will keep the frequencies of the words (used when building vocab)
        as the `freq` attribute.
    """
    def __init__(self, min_freq: int=1,
                 max_size: Optional[int]=None,
                 tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 keep_freq: bool=False,
                 padding_index=0,
                 unk_index=1,
                 prebuilt_stoi: Dict[str, int]=None,
                 ):
        self.min_freq = min_freq
        self.max_size = max_size
        self.tokenizer = tokenizer
        self.keep_freq = keep_freq
        self.padding_index = padding_index
        self.unk_index = unk_index
        self._prebuilt = prebuilt_stoi is not None
        self.stoi = DefaultDict(unk_index)
        self.itos = {}
        if prebuilt_stoi is not None:
            for k, v in prebuilt_stoi.items():
                self.stoi[k] = v
                self.itos[v] = k

    def _textlist_to_tokenlists(self, texts: Iterable[str]):
        return [self.tokenizer(text) for text in texts]

    def _tokenlist_to_idxs(self, tokenlist: Iterable[str]):
        return [self.stoi[x] for x in tokenlist]

    def _text_to_idxs(self, text):
        return self._tokenlist_to_idxs(self.tokenizer(text))

    def fit_transform(self, texts: Iterable[str]):
        _inner_texts = self._textlist_to_tokenlists(texts)
        if not self._prebuilt:
            _freqs = Counter()
            for text in _inner_texts:
                _freqs.update(text)

            if self.keep_freq: self.freq = _freqs

            idx = 0
            for word, freq in _freqs.most_common(None):
                # stop construction of vocab when appropriate
                if self.max_size is not None and idx >= self.max_size: break
                if freq < self.min_freq: break

                # keep these reserved indices intact
                if idx == self.padding_index:
                    idx += 1
                if idx == self.unk_index:
                    idx += 1
                self.stoi[word] = idx
                self.itos[idx] = word
                assert freq >= self.min_freq
                idx += 1
        return [self._tokenlist_to_idxs(x) for x in _inner_texts]

    def transform(self, texts: Iterable[str]):
        return [self._text_to_idxs(x) for x in texts]

    def __len__(self):
        return len(self.stoi)

    @property
    def size(self):
        return len(self.stoi)
