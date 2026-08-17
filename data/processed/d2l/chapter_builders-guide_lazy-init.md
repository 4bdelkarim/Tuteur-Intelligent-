---
source_url: https://d2l.ai/chapter_builders-guide/lazy-init.html
title: 6.4. Lazy Initialization
chapter: '6'
section_number: '6.4'
date: null
extractor: d2l
source_type: web
---

# 6.4. Lazy Initialization

So far, it might seem that we got away with being sloppy in setting up
our networks. Specifically, we did the following unintuitive things,
which might not seem like they should work:

- We defined the network architectures without specifying the input
 dimensionality.
- We added layers without specifying the output dimension of the
 previous layer.
- We even “initialized” these parameters before providing enough
 information to determine how many parameters our models should
 contain.

You might be surprised that our code runs at all. After all, there is no
way the deep learning framework could tell what the input dimensionality
of a network would be. The trick here is that the framework *defers
initialization*, waiting until the first time we pass data through the
model, to infer the sizes of each layer on the fly.

Later on, when working with convolutional neural networks, this
technique will become even more convenient since the input
dimensionality (e.g., the resolution of an image) will affect the
dimensionality of each subsequent layer. Hence the ability to set
parameters without the need to know, at the time of writing the code,
the value of the dimension can greatly simplify the task of specifying
and subsequently modifying our models. Next, we go deeper into the
mechanics of initialization.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

To begin, let’s instantiate an MLP.

```python
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

At this point, the network cannot possibly know the dimensions of the
input layer’s weights because the input dimension remains unknown.

Consequently the framework has not yet initialized any parameters. We
confirm by attempting to access the parameters below.

```python
net[0].weight
```

Next let’s pass data through the network to make the framework finally
initialize parameters.

```python
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

As soon as we know the input dimensionality, 20, the framework can
identify the shape of the first layer’s weight matrix by plugging in the
value of 20. Having recognized the first layer’s shape, the framework
proceeds to the second layer, and so on through the computational graph
until all shapes are known. Note that in this case, only the first layer
requires lazy initialization, but the framework initializes
sequentially. Once all parameter shapes are known, the framework can
finally initialize the parameters.

The following method passes in dummy inputs through the network for a
dry run to infer all parameter shapes and subsequently initializes the
parameters. It will be used later when default random initializations
are not desired.

```python
@d2l.add_to_class(d2l.Module) #@save
def apply_init(self, inputs, init=None):
 self.forward(*inputs)
 if init is not None:
 self.net.apply(init)
```

## 6.4.1. Summary

Lazy initialization can be convenient, allowing the framework to infer
parameter shapes automatically, making it easy to modify architectures
and eliminating one common source of errors. We can pass data through
the model to make the framework finally initialize parameters.
