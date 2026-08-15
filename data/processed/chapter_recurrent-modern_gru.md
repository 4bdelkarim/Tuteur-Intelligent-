---
source_url: https://d2l.ai/chapter_recurrent-modern/gru.html
title: 10.2. Gated Recurrent Units (GRU)
chapter: '10'
section_number: '10.2'
date: null
extractor: d2l
source_type: web
source: chapter_recurrent-modern_gru
---

# 10.2. Gated Recurrent Units (GRU)

As RNNs and particularly the LSTM architecture (Section 10.1)
rapidly gained popularity during the 2010s, a number of researchers
began to experiment with simplified architectures in hopes of retaining
the key idea of incorporating an internal state and multiplicative
gating mechanisms but with the aim of speeding up computation. The gated
recurrent unit (GRU) (Cho et al., 2014)
offered a streamlined version of the LSTM memory cell that often
achieves comparable performance but with the advantage of being faster
to compute (Chung et al., 2014).

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 10.2.1. Reset Gate and Update Gate

Here, the LSTM’s three gates are replaced by two: the *reset gate* and
the *update gate*. As with LSTMs, these gates are given sigmoid
activations, forcing their values to lie in the interval $(0, 1)$.
Intuitively, the reset gate controls how much of the previous state we
might still want to remember. Likewise, an update gate would allow us to
control how much of the new state is just a copy of the old one.
Fig. 10.2.1 illustrates the inputs for both the reset and
update gates in a GRU, given the input of the current time step and the
hidden state of the previous time step. The outputs of the gates are
given by two fully connected layers with a sigmoid activation function.

Fig. 10.2.1 Computing the reset gate and the update gate in a GRU model.

Mathematically, for a given time step $t$, suppose that the input
is a minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number
of examples $=n$; number of inputs $=d$) and the hidden
state of the previous time step is
$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (number of hidden
units $=h$). Then the reset gate
$\mathbf{R}_t \in \mathbb{R}^{n \times h}$ and update gate
$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ are computed as
follows:

$$
(10.2.1)\[\begin{split}\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z}),
\end{aligned}\end{split}
$$

where
$\mathbf{W}_{\textrm{xr}}, \mathbf{W}_{\textrm{xz}} \in \mathbb{R}^{d \times h}$
and
$\mathbf{W}_{\textrm{hr}}, \mathbf{W}_{\textrm{hz}} \in \mathbb{R}^{h \times h}$
are weight parameters and
$\mathbf{b}_\textrm{r}, \mathbf{b}_\textrm{z} \in \mathbb{R}^{1 \times h}$
are bias parameters.

## 10.2.2. Candidate Hidden State

Next, we integrate the reset gate $\mathbf{R}_t$ with the regular
updating mechanism in (9.4.5), leading to the
following *candidate hidden state*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ at time step
$t$:

$$
(10.2.2)\[\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),
$$

where $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ and
$\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ are weight
parameters, $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$ is
the bias, and the symbol $\odot$ is the Hadamard (elementwise)
product operator. Here we use a tanh activation function.

The result is a *candidate*, since we still need to incorporate the
action of the update gate. Comparing with (9.4.5),
the influence of the previous states can now be reduced with the
elementwise multiplication of $\mathbf{R}_t$ and
$\mathbf{H}_{t-1}$ in (10.2.2). Whenever the entries
in the reset gate $\mathbf{R}_t$ are close to 1, we recover a
vanilla RNN such as that in (9.4.5). For all entries
of the reset gate $\mathbf{R}_t$ that are close to 0, the
candidate hidden state is the result of an MLP with $\mathbf{X}_t$
as input. Any pre-existing hidden state is thus *reset* to defaults.

Fig. 10.2.2 illustrates the computational flow after applying
the reset gate.

Fig. 10.2.2 Computing the candidate hidden state in a GRU model.

## 10.2.3. Hidden State

Finally, we need to incorporate the effect of the update gate
$\mathbf{Z}_t$. This determines the extent to which the new hidden
state $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ matches the old
state $\mathbf{H}_{t-1}$ compared with how much it resembles the
new candidate state $\tilde{\mathbf{H}}_t$. The update gate
$\mathbf{Z}_t$ can be used for this purpose, simply by taking
elementwise convex combinations of $\mathbf{H}_{t-1}$ and
$\tilde{\mathbf{H}}_t$. This leads to the final update equation
for the GRU:

$$
(10.2.3)\[\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.
$$

Whenever the update gate $\mathbf{Z}_t$ is close to 1, we simply
retain the old state. In this case the information from
$\mathbf{X}_t$ is ignored, effectively skipping time step
$t$ in the dependency chain. By contrast, whenever
$\mathbf{Z}_t$ is close to 0, the new latent state
$\mathbf{H}_t$ approaches the candidate latent state
$\tilde{\mathbf{H}}_t$. Fig. 10.2.3 shows the
computational flow after the update gate is in action.

Fig. 10.2.3 Computing the hidden state in a GRU model.

In summary, GRUs have the following two distinguishing features:

- Reset gates help capture short-term dependencies in sequences.
- Update gates help capture long-term dependencies in sequences.

## 10.2.4. Implementation from Scratch

To gain a better understanding of the GRU model, let’s implement it from
scratch.

### 10.2.4.1. Initializing Model Parameters

The first step is to initialize the model parameters. We draw the
weights from a Gaussian distribution with standard deviation to be
`sigma` and set the bias to 0. The hyperparameter `num_hiddens`
defines the number of hidden units. We instantiate all weights and
biases relating to the update gate, the reset gate, and the candidate
hidden state.

```python
class GRUScratch(d2l.Module):
 def __init__(self, num_inputs, num_hiddens, sigma=0.01):
 super().__init__()
 self.save_hyperparameters()

 init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
 triple = lambda: (init_weight(num_inputs, num_hiddens),
 init_weight(num_hiddens, num_hiddens),
 nn.Parameter(torch.zeros(num_hiddens)))
 self.W_xz, self.W_hz, self.b_z = triple() # Update gate
 self.W_xr, self.W_hr, self.b_r = triple() # Reset gate
 self.W_xh, self.W_hh, self.b_h = triple() # Candidate hidden state
```

### 10.2.4.2. Defining the Model

Now we are ready to define the GRU forward computation. Its structure is
the same as that of the basic RNN cell, except that the update equations
are more complex.

```python
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
 if H is None:
 # Initial state with shape: (batch_size, num_hiddens)
 H = torch.zeros((inputs.shape[1], self.num_hiddens),
 device=inputs.device)
 outputs = []
 for X in inputs:
 Z = torch.sigmoid(torch.matmul(X, self.W_xz) +
 torch.matmul(H, self.W_hz) + self.b_z)
 R = torch.sigmoid(torch.matmul(X, self.W_xr) +
 torch.matmul(H, self.W_hr) + self.b_r)
 H_tilde = torch.tanh(torch.matmul(X, self.W_xh) +
 torch.matmul(R * H, self.W_hh) + self.b_h)
 H = Z * H + (1 - Z) * H_tilde
 outputs.append(H)
 return outputs, H
```

### 10.2.4.3. Training

Training a language model on *The Time Machine* dataset works in exactly
the same manner as in Section 9.5.

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

## 10.2.5. Concise Implementation

In high-level APIs, we can directly instantiate a GRU model. This
encapsulates all the configuration detail that we made explicit above.

```python
class GRU(d2l.RNN):
 def __init__(self, num_inputs, num_hiddens):
 d2l.Module.__init__(self)
 self.save_hyperparameters()
 self.rnn = nn.GRU(num_inputs, num_hiddens)
```

The code is significantly faster in training as it uses compiled
operators rather than Python.

```python
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

After training, we print out the perplexity on the training set and the
predicted sequence following the provided prefix.

```python
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

## 10.2.6. Summary

Compared with LSTMs, GRUs achieve similar performance but tend to be
lighter computationally. Generally, compared with simple RNNs, gated
RNNS, just like LSTMs and GRUs, can better capture dependencies for
sequences with large time step distances. GRUs contain basic RNNs as
their extreme case whenever the reset gate is switched on. They can also
skip subsequences by turning on the update gate.
