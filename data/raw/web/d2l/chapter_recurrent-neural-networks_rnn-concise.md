---
source_url: "https://d2l.ai/chapter_recurrent-neural-networks/rnn-concise.html"
title: "9.6. Concise Implementation of Recurrent Neural Networks"
chapter: "9"
section_number: "9.6"
date: null
extractor: "d2l"
---

# 9.6. Concise Implementation of Recurrent Neural Networks

Like most of our from-scratch implementations,
Section 9.5 was designed to provide insight into how
each component works. But when you are using RNNs every day or writing
production code, you will want to rely more on libraries that cut down
on both implementation time (by supplying library code for common models
and functions) and computation time (by optimizing the heck out of these
library implementations). This section will show you how to implement
the same language model more efficiently using the high-level API
provided by your deep learning framework. We begin, as before, by
loading *The Time Machine* dataset.

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

## 9.6.1. Defining the Model

We define the following class using the RNN implemented by high-level
APIs.

```python
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

Inheriting from the `RNNLMScratch` class in
Section 9.5, the following `RNNLM` class defines a
complete RNN-based language model. Note that we need to create a
separate fully connected output layer.

```python
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, hiddens):
        return self.linear(hiddens).swapaxes(0, 1)
```

## 9.6.2. Training and Predicting

Before training the model, let’s make a prediction with a model
initialized with random weights. Given that we have not trained the
network, it will generate nonsensical predictions.

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Next, we train our model, leveraging the high-level API.

```python
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

![../_images/output_rnn-concise_eff2f4_62_0.svg](../_images/output_rnn-concise_eff2f4_62_0.svg)

Compared with Section 9.5, this model achieves
comparable perplexity, but runs faster due to the optimized
implementations. As before, we can generate predicted tokens following
the specified prefix string.

```python
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

## 9.6.3. Summary

High-level APIs in deep learning frameworks provide implementations of
standard RNNs. These libraries help you to avoid wasting time
reimplementing standard models. Moreover, framework implementations are
often highly optimized, leading to significant (computational)
performance gains when compared with implementations from scratch.

## 9.6.4. Exercises

1. Can you make the RNN model overfit using the high-level APIs?
2. Implement the autoregressive model of Section 9.1 using
   an RNN.
