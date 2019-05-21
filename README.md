# Simple NLP
Are you tired of out-of-memory errors?
Introducing Simple NLP: The (hopefully) simplest library for building NLP pipelines in PyTorch.

Key features:
- Built on top of PyTorch-native concepts (Dataset, DataLoader, Sampler)
- Simple, readable code so you can know exactly what is happening when and where
- Minimum overhead


## Usage
```
from simple_nlp import *

vocab = Vocab(min_freq=2)
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
