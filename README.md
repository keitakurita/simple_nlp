# Simple NLP
Simple NLP: A simple, minimal library for building NLP pipelines in PyTorch.

Key attributes:
- Built on top of PyTorch-native concepts (Dataset, DataLoader, Sampler)
- Simple, readable code so you can know exactly what is happening when and where
- Minimum overhead

## Features
- Converting text to integers
- Bucketing sequences of similar length together
- Dynamic padding

## Usage
```
from simple_nlp import *

vocab = Vocab(min_freq=2, tokenizer=lambda x: x.split()))
x_train = vocab.fit_transform(train_texts)
x_val = vocab.transform(val_texts)
y_train = np.array(train_labels)
y_val = np.array(val_labels)

train_dataset = TextDataset(x_train, y_train)
valid_dataset = TextDataset(x_val, y_val)

sampler = BucketBatchSampler(train_dataset, 32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
```
