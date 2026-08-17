---
source_url: "https://d2l.ai/chapter_builders-guide/custom-layer.html"
title: "6.5. Custom Layers"
chapter: "6"
section_number: "6.5"
date: null
extractor: "d2l"
---

# 6.5. Custom Layers

One factor behind deep learning’s success is the availability of a wide
range of layers that can be composed in creative ways to design
architectures suitable for a wide variety of tasks. For instance,
researchers have invented layers specifically for handling images, text,
looping over sequential data, and performing dynamic programming. Sooner
or later, you will need a layer that does not exist yet in the deep
learning framework. In these cases, you must build a custom layer. In
this section, we show you how.

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

## 6.5.1. Layers without Parameters

To start, we construct a custom layer that does not have any parameters
of its own. This should look familiar if you recall our introduction to
modules in Section 6.1. The following
`CenteredLayer` class simply subtracts the mean from its input. To
build it, we simply need to inherit from the base layer class and
implement the forward propagation function.

```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

Let’s verify that our layer works as intended by feeding some data
through it.

```python
layer = CenteredLayer()
layer(torch.tensor([1.0, 2, 3, 4, 5]))
```

We can now incorporate our layer as a component in constructing more
complex models.

```python
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

As an extra sanity check, we can send random data through the network
and check that the mean is in fact 0. Because we are dealing with
floating point numbers, we may still see a very small nonzero number due
to quantization.

```python
Y = net(torch.rand(4, 8))
Y.mean()
```

## 6.5.2. Layers with Parameters

Now that we know how to define simple layers, let’s move on to defining
layers with parameters that can be adjusted through training. We can use
built-in functions to create parameters, which provide some basic
housekeeping functionality. In particular, they govern access,
initialization, sharing, saving, and loading model parameters. This way,
among other benefits, we will not need to write custom serialization
routines for every custom layer.

Now let’s implement our own version of the fully connected layer. Recall
that this layer requires two parameters, one to represent the weight and
the other for the bias. In this implementation, we bake in the ReLU
activation as a default. This layer requires two input arguments:
`in_units` and `units`, which denote the number of inputs and
outputs, respectively.

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

Next, we instantiate the `MyLinear` class and access its model
parameters.

```python
linear = MyLinear(5, 3)
linear.weight
```

We can directly carry out forward propagation calculations using custom
layers.

```python
linear(torch.rand(2, 5))
```

We can also construct models using custom layers. Once we have that we
can use it just like the built-in fully connected layer.

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

## 6.5.3. Summary

We can design custom layers via the basic layer class. This allows us to
define flexible new layers that behave differently from any existing
layers in the library. Once defined, custom layers can be invoked in
arbitrary contexts and architectures. Layers can have local parameters,
which can be created through built-in functions.

## 6.5.4. Exercises

1. Design a layer that takes an input and computes a tensor reduction,
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
2. Design a layer that returns the leading half of the Fourier
   coefficients of the data.
