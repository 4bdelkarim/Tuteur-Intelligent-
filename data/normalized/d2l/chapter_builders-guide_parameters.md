---
source_url: https://d2l.ai/chapter_builders-guide/parameters.html
title: 6.2. Parameter Management
chapter: '6'
section_number: '6.2'
date: null
extractor: d2l
source_type: web
---

# 6.2. Parameter Management

Once we have chosen an architecture and set our hyperparameters, we
proceed to the training loop, where our goal is to find parameter values
that minimize our loss function. After training, we will need these
parameters in order to make future predictions. Additionally, we will
sometimes wish to extract the parameters perhaps to reuse them in some
other context, to save our model to disk so that it may be executed in
other software, or for examination in the hope of gaining scientific
understanding.

Most of the time, we will be able to ignore the nitty-gritty details of
how parameters are declared and manipulated, relying on deep learning
frameworks to do the heavy lifting. However, when we move away from
stacked architectures with standard layers, we will sometimes need to
get into the weeds of declaring and manipulating parameters. In this
section, we cover the following:

- Accessing parameters for debugging, diagnostics, and visualizations.
- Sharing parameters across different model components.

```python
import torch
from torch import nn
```

We start by focusing on an MLP with one hidden layer.

```python
net = nn.Sequential(nn.LazyLinear(8),
 nn.ReLU(),
 nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

## 6.2.1. Parameter Access

Let’s start with how to access parameters from the models that you
already know.

When a model is defined via the `Sequential` class, we can first
access any layer by indexing into the model as though it were a list.
Each layer’s parameters are conveniently located in its attribute.

We can inspect the parameters of the second fully connected layer as
follows.

```python
net[2].state_dict()
```

We can see that this fully connected layer contains two parameters,
corresponding to that layer’s weights and biases, respectively.

### 6.2.1.1. Targeted Parameters

Note that each parameter is represented as an instance of the parameter
class. To do anything useful with the parameters, we first need to
access the underlying numerical values. There are several ways to do
this. Some are simpler while others are more general. The following code
extracts the bias from the second neural network layer, which returns a
parameter class instance, and further accesses that parameter’s value.

```python
type(net[2].bias), net[2].bias.data
```

Parameters are complex objects, containing values, gradients, and
additional information. That is why we need to request the value
explicitly.

In addition to the value, each parameter also allows us to access the
gradient. Because we have not invoked backpropagation for this network
yet, it is in its initial state.

```python
net[2].weight.grad == None
```

### 6.2.1.2. All Parameters at Once

When we need to perform operations on all parameters, accessing them
one-by-one can grow tedious. The situation can grow especially unwieldy
when we work with more complex, e.g., nested, modules, since we would
need to recurse through the entire tree to extract each sub-module’s
parameters. Below we demonstrate accessing the parameters of all layers.

```python
[(name, param.shape) for name, param in net.named_parameters()]
```

## 6.2.2. Tied Parameters

Often, we want to share parameters across multiple layers. Let’s see how
to do this elegantly. In the following we allocate a fully connected
layer and then use its parameters specifically to set those of another
layer. Here we need to run the forward propagation `net(X)` before
accessing the parameters.

```python
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
 shared, nn.ReLU(),
 shared, nn.ReLU(),
 nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

This example shows that the parameters of the second and third layer are
tied. They are not just equal, they are represented by the same exact
tensor. Thus, if we change one of the parameters, the other one changes,
too.

You might wonder, when parameters are tied what happens to the
gradients? Since the model parameters contain gradients, the gradients
of the second hidden layer and the third hidden layer are added together
during backpropagation.

## 6.2.3. Summary

We have several ways of accessing and tying model parameters.
