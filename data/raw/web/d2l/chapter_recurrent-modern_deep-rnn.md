---
source_url: "https://d2l.ai/chapter_recurrent-modern/deep-rnn.html"
title: "10.3. Deep Recurrent Neural Networks"
chapter: "10"
section_number: "10.3"
date: null
extractor: "d2l"
---

# 10.3. Deep Recurrent Neural Networks

Up until now, we have focused on defining networks consisting of a
sequence input, a single hidden RNN layer, and an output layer. Despite
having just one hidden layer between the input at any time step and the
corresponding output, there is a sense in which these networks are deep.
Inputs from the first time step can influence the outputs at the final
time step $T$ (often 100s or 1000s of steps later). These inputs
pass through $T$ applications of the recurrent layer before
reaching the final output. However, we often also wish to retain the
ability to express complex relationships between the inputs at a given
time step and the outputs at that same time step. Thus we often
construct RNNs that are deep not only in the time direction but also in
the input-to-output direction. This is precisely the notion of depth
that we have already encountered in our development of MLPs and deep
CNNs.

The standard method for building this sort of deep RNN is strikingly
simple: we stack the RNNs on top of each other. Given a sequence of
length $T$, the first RNN produces a sequence of outputs, also of
length $T$. These, in turn, constitute the inputs to the next RNN
layer. In this short section, we illustrate this design pattern and
present a simple example for how to code up such stacked RNNs. Below, in
Fig. 10.3.1, we illustrate a deep RNN with $L$ hidden
layers. Each hidden state operates on a sequential input and produces a
sequential output. Moreover, any RNN cell (white box in
Fig. 10.3.1) at each time step depends on both the same
layer’s value at the previous time step and the previous layer’s value
at the same time step.

![../_images/deep-rnn.svg](../_images/deep-rnn.svg)

Fig. 10.3.1 Architecture of a deep RNN.

Formally, suppose that we have a minibatch input
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number of examples
$=n$; number of inputs in each example $=d$) at time step
$t$. At the same time step, let the hidden state of the
$l^\textrm{th}$ hidden layer ($l=1,\ldots,L$) be
$\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$ (number of hidden
units $=h$) and the output layer variable be
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (number of outputs:
$q$). Setting $\mathbf{H}_t^{(0)} = \mathbf{X}_t$, the
hidden state of the $l^\textrm{th}$ hidden layer that uses the
activation function $\phi_l$ is calculated as follows:

$$
(10.3.1)\[\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),
$$

where the weights
$\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$ and
$\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$,
together with the bias
$\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$, are the
model parameters of the $l^\textrm{th}$ hidden layer.

At the end, the calculation of the output layer is only based on the
hidden state of the final $L^\textrm{th}$ hidden layer:

$$
(10.3.2)\[\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},
$$

where the weight
$\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ and the
bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ are the
model parameters of the output layer.

Just as with MLPs, the number of hidden layers $L$ and the number
of hidden units $h$ are hyperparameters that we can tune. Common
RNN layer widths ($h$) are in the range $(64, 2056)$, and
common depths ($L$) are in the range $(1, 8)$. In addition,
we can easily get a deep-gated RNN by replacing the hidden state
computation in (10.3.1) with that from an LSTM or a GRU.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 10.3.1. Implementation from Scratch

To implement a multilayer RNN from scratch, we can treat each layer as
an `RNNScratch` instance with its own learnable parameters.

```python
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

The multilayer forward computation simply performs forward computation
layer by layer.

```python
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = torch.stack(outputs, 0)
    return outputs, Hs
```

As an example, we train a deep GRU model on *The Time Machine* dataset
(same as in Section 9.5). To keep things simple we set
the number of layers to 2.

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                              num_hiddens=32, num_layers=2)
model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

![../_images/output_deep-rnn_d70a11_48_0.svg](../_images/output_deep-rnn_d70a11_48_0.svg)

## 10.3.2. Concise Implementation

Fortunately many of the logistical details required to implement
multiple layers of an RNN are readily available in high-level APIs. Our
concise implementation will use such built-in functionalities. The code
generalizes the one we used previously in Section 10.2, letting
us specify the number of layers explicitly rather than picking the
default of only one layer.

```python
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

The architectural decisions such as choosing hyperparameters are very
similar to those of Section 10.2. We pick the same number of
inputs and outputs as we have distinct tokens, i.e., `vocab_size`. The
number of hidden units is still 32. The only difference is that we now
select a nontrivial number of hidden layers by specifying the value of
`num_layers`.

```python
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

![../_images/output_deep-rnn_d70a11_82_0.svg](../_images/output_deep-rnn_d70a11_82_0.svg)

```python
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

## 10.3.3. Summary

In deep RNNs, the hidden state information is passed to the next time
step of the current layer and the current time step of the next layer.
There exist many different flavors of deep RNNs, such as LSTMs, GRUs, or
vanilla RNNs. Conveniently, these models are all available as parts of
the high-level APIs of deep learning frameworks. Initialization of
models requires care. Overall, deep RNNs require considerable amount of
work (such as learning rate and clipping) to ensure proper convergence.

## 10.3.4. Exercises

1. Replace the GRU by an LSTM and compare the accuracy and training
   speed.
2. Increase the training data to include multiple books. How low can you
   go on the perplexity scale?
3. Would you want to combine sources of different authors when modeling
   text? Why is this a good idea? What could go wrong?
