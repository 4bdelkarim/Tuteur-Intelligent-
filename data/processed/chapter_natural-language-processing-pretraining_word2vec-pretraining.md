---
source_url: https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec-pretraining.html
title: 15.4. Pretraining word2vec
chapter: '15'
section_number: '15.4'
date: null
extractor: d2l
source_type: web
source: chapter_natural-language-processing-pretraining_word2vec-pretraining
---

# 15.4. Pretraining word2vec

We go on to implement the skip-gram model defined in
Section 15.1. Then we will pretrain word2vec using negative
sampling on the PTB dataset. First of all, let’s obtain the data
iterator and the vocabulary for this dataset by calling the
`d2l.load_data_ptb` function, which was described in
Section 15.3

```python
import math
import torch
from torch import nn
from d2l import torch as d2l

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
 num_noise_words)
```

## 15.4.1. The Skip-Gram Model

We implement the skip-gram model by using embedding layers and batch
matrix multiplications. First, let’s review how embedding layers work.

### 15.4.1.1. Embedding Layer

As described in Section 10.7, an embedding layer maps a
token’s index to its feature vector. The weight of this layer is a
matrix whose number of rows equals to the dictionary size
(`input_dim`) and number of columns equals to the vector dimension for
each token (`output_dim`). After a word embedding model is trained,
this weight is what we need.

```python
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
 f'dtype={embed.weight.dtype})')
```

The input of an embedding layer is the index of a token (word). For any
token index $i$, its vector representation can be obtained from
the $i^\textrm{th}$ row of the weight matrix in the embedding
layer. Since the vector dimension (`output_dim`) was set to 4, the
embedding layer returns vectors with shape (2, 3, 4) for a minibatch of
token indices with shape (2, 3).

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 15.4.1.2. Defining the Forward Propagation

In the forward propagation, the input of the skip-gram model includes
the center word indices `center` of shape (batch size, 1) and the
concatenated context and noise word indices `contexts_and_negatives`
of shape (batch size, `max_len`), where `max_len` is defined in
Section 15.3.5. These two variables are
first transformed from the token indices into vectors via the embedding
layer, then their batch matrix multiplication (described in
Section 11.3.2.2) returns an output of shape (batch size, 1,
`max_len`). Each element in the output is the dot product of a center
word vector and a context or noise word vector.

```python
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
 v = embed_v(center)
 u = embed_u(contexts_and_negatives)
 pred = torch.bmm(v, u.permute(0, 2, 1))
 return pred
```

Let’s print the output shape of this `skip_gram` function for some
example inputs.

```python
skip_gram(torch.ones((2, 1), dtype=torch.long),
 torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## 15.4.2. Training

Before training the skip-gram model with negative sampling, let’s first
define its loss function.

### 15.4.2.1. Binary Cross-Entropy Loss

According to the definition of the loss function for negative sampling
in Section 15.2.1, we will use the binary
cross-entropy loss.

```python
class SigmoidBCELoss(nn.Module):
 # Binary cross-entropy loss with masking
 def __init__(self):
 super().__init__()

 def forward(self, inputs, target, mask=None):
 out = nn.functional.binary_cross_entropy_with_logits(
 inputs, target, weight=mask, reduction="none")
 return out.mean(dim=1)

loss = SigmoidBCELoss()
```

Recall our descriptions of the mask variable and the label variable in
Section 15.3.5. The following calculates
the binary cross-entropy loss for the given variables.

```python
pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Below shows how the above results are calculated (in a less efficient
way) using the sigmoid activation function in the binary cross-entropy
loss. We can consider the two outputs as two normalized losses that are
averaged over non-masked predictions.

```python
def sigmd(x):
 return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### 15.4.2.2. Initializing Model Parameters

We define two embedding layers for all the words in the vocabulary when
they are used as center words and context words, respectively. The word
vector dimension `embed_size` is set to 100.

```python
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
 embedding_dim=embed_size),
 nn.Embedding(num_embeddings=len(vocab),
 embedding_dim=embed_size))
```

### 15.4.2.3. Defining the Training Loop

The training loop is defined below. Because of the existence of padding,
the calculation of the loss function is slightly different compared to
the previous training functions.

```python
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
 def init_weights(module):
 if type(module) == nn.Embedding:
 nn.init.xavier_uniform_(module.weight)
 net.apply(init_weights)
 net = net.to(device)
 optimizer = torch.optim.Adam(net.parameters(), lr=lr)
 animator = d2l.Animator(xlabel='epoch', ylabel='loss',
 xlim=[1, num_epochs])
 # Sum of normalized losses, no. of normalized losses
 metric = d2l.Accumulator(2)
 for epoch in range(num_epochs):
 timer, num_batches = d2l.Timer(), len(data_iter)
 for i, batch in enumerate(data_iter):
 optimizer.zero_grad()
 center, context_negative, mask, label = [
 data.to(device) for data in batch]

 pred = skip_gram(center, context_negative, net[0], net[1])
 l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
 / mask.sum(axis=1) * mask.shape[1])
 l.sum().backward()
 optimizer.step()
 metric.add(l.sum(), l.numel())
 if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
 animator.add(epoch + (i + 1) / num_batches,
 (metric[0] / metric[1],))
 print(f'loss {metric[0] / metric[1]:.3f}, '
 f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Now we can train a skip-gram model using negative sampling.

```python
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 15.4.3. Applying Word Embeddings

After training the word2vec model, we can use the cosine similarity of
word vectors from the trained model to find words from the dictionary
that are most semantically similar to an input word.

```python
def get_similar_tokens(query_token, k, embed):
 W = embed.weight.data
 x = W[vocab[query_token]]
 # Compute the cosine similarity. Add 1e-9 for numerical stability
 cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
 torch.sum(x * x) + 1e-9)
 topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
 for i in topk[1:]: # Remove the input words
 print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## 15.4.4. Summary

- We can train a skip-gram model with negative sampling using embedding
 layers and the binary cross-entropy loss.
- Applications of word embeddings include finding semantically similar
 words for a given word based on the cosine similarity of word
 vectors.
