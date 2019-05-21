from typing import *
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, x: List[List[int]], y=None):
        self.x = x
        self.y = y
        if y is not None:
            assert len(x) == len(y)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], None

    def __len__(self):
        return len(self.x)
