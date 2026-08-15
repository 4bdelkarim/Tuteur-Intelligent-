---
source_url: https://d2l.ai/chapter_recurrent-modern/bi-rnn.html
title: 10.4. Bidirectional Recurrent Neural Networks
chapter: '10'
section_number: '10.4'
date: null
extractor: d2l
source_type: web
source: chapter_recurrent-modern_bi-rnn
---

# 10.4. Bidirectional Recurrent Neural Networks

So far, our working example of a sequence learning task has been
language modeling, where we aim to predict the next token given all
previous tokens in a sequence. In this scenario, we wish only to
condition upon the leftward context, and thus the unidirectional
chaining of a standard RNN seems appropriate. However, there are many
other sequence learning tasks contexts where it is perfectly fine to
condition the prediction at every time step on both the leftward and the
rightward context. Consider, for example, part of speech detection. Why
shouldn’t we take the context in both directions into account when
assessing the part of speech associated with a given word?

Another common task—often useful as a pretraining exercise prior to
fine-tuning a model on an actual task of interest—is to mask out random
tokens in a text document and then to train a sequence model to predict
the values of the missing tokens. Note that depending on what comes
after the blank, the likely value of the missing token changes
dramatically:

- I am `___`.
- I am `___` hungry.
- I am `___` hungry, and I can eat half a pig.

In the first sentence “happy” seems to be a likely candidate. The words
“not” and “very” seem plausible in the second sentence, but “not” seems
incompatible with the third sentences.

Fortunately, a simple technique transforms any unidirectional RNN into a
bidirectional RNN (Schuster and Paliwal, 1997). We simply implement
two unidirectional RNN layers chained together in opposite directions
and acting on the same input (Fig. 10.4.1). For the first RNN
layer, the first input is $\mathbf{x}_1$ and the last input is
$\mathbf{x}_T$, but for the second RNN layer, the first input is
$\mathbf{x}_T$ and the last input is $\mathbf{x}_1$. To
produce the output of this bidirectional RNN layer, we simply
concatenate together the corresponding outputs of the two underlying
unidirectional RNN layers.

Fig. 10.4.1 Architecture of a bidirectional RNN.

Formally for any time step $t$, we consider a minibatch input
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number of examples
$=n$; number of inputs in each example $=d$) and let the
hidden layer activation function be $\phi$. In the bidirectional
architecture, the forward and backward hidden states for this time step
are $\overrightarrow{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$
and $\overleftarrow{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$,
respectively, where $h$ is the number of hidden units. The forward
and backward hidden state updates are as follows:

$$
(10.4.1)\[\begin{split}\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)} + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)} + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}\end{split}
$$

where the weights
$\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ and } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$,
and the biases
$\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$ and
$\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$ are all
the model parameters.

Next, we concatenate the forward and backward hidden states
$\overrightarrow{\mathbf{H}}_t$ and
$\overleftarrow{\mathbf{H}}_t$ to obtain the hidden state
$\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ for feeding into the
output layer. In deep bidirectional RNNs with multiple hidden layers,
such information is passed on as *input* to the next bidirectional
layer. Last, the output layer computes the output
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (number of outputs
$=q$):

$$
(10.4.2)\[\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.
$$

Here, the weight matrix
$\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$ and the
bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ are the
model parameters of the output layer. While technically, the two
directions can have different numbers of hidden units, this design
choice is seldom made in practice. We now demonstrate a simple
implementation of a bidirectional RNN.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 10.4.1. Implementation from Scratch

To implement a bidirectional RNN from scratch, we can include two
unidirectional `RNNScratch` instances with separate learnable
parameters.

```python
class BiRNNScratch(d2l.Module):
 def __init__(self, num_inputs, num_hiddens, sigma=0.01):
 super().__init__()
 self.save_hyperparameters()
 self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
 self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
 self.num_hiddens *= 2 # The output dimension will be doubled
```

States of forward and backward RNNs are updated separately, while
outputs of these two RNNs are concatenated.

```python
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
 f_H, b_H = Hs if Hs is not None else (None, None)
 f_outputs, f_H = self.f_rnn(inputs, f_H)
 b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
 outputs = [torch.cat((f, b), -1) for f, b in zip(
 f_outputs, reversed(b_outputs))]
 return outputs, (f_H, b_H)
```

## 10.4.2. Concise Implementation

Using the high-level APIs, we can implement bidirectional RNNs more
concisely. Here we take a GRU model as an example.

```python
class BiGRU(d2l.RNN):
 def __init__(self, num_inputs, num_hiddens):
 d2l.Module.__init__(self)
 self.save_hyperparameters()
 self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
 self.num_hiddens *= 2
```

## 10.4.3. Summary

In bidirectional RNNs, the hidden state for each time step is
simultaneously determined by the data prior to and after the current
time step. Bidirectional RNNs are mostly useful for sequence encoding
and the estimation of observations given bidirectional context.
Bidirectional RNNs are very costly to train due to long gradient chains.
