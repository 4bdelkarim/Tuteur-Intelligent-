---
source_url: https://d2l.ai/chapter_builders-guide/init-param.html
title: 6.3. Parameter Initialization
chapter: '6'
section_number: '6.3'
date: null
extractor: d2l
source_type: web
source: chapter_builders-guide_init-param
---

# 6.3. Parameter Initialization

Now that we know how to access the parameters, let’s look at how to
initialize them properly. We discussed the need for proper
initialization in Section 5.4. The deep learning
framework provides default random initializations to its layers.
However, we often want to initialize our weights according to various
other protocols. The framework provides most commonly used protocols,
and also allows to create a custom initializer.

```python
import torch
from torch import nn
```

By default, PyTorch initializes weight and bias matrices uniformly by
drawing from a range that is computed according to the input and output
dimension. PyTorch’s `nn.init` module provides a variety of preset
initialization methods.

```python
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

## 6.3.1. Built-in Initialization

Let’s begin by calling on built-in initializers. The code below
initializes all weight parameters as Gaussian random variables with
standard deviation 0.01, while bias parameters are cleared to zero.

```python
def init_normal(module):
 if type(module) == nn.Linear:
 nn.init.normal_(module.weight, mean=0, std=0.01)
 nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

We can also initialize all the parameters to a given constant value
(say, 1).

```python
def init_constant(module):
 if type(module) == nn.Linear:
 nn.init.constant_(module.weight, 1)
 nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

We can also apply different initializers for certain blocks. For
example, below we initialize the first layer with the Xavier initializer
and initialize the second layer to a constant value of 42.

```python
def init_xavier(module):
 if type(module) == nn.Linear:
 nn.init.xavier_uniform_(module.weight)

def init_42(module):
 if type(module) == nn.Linear:
 nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

### 6.3.1.1. Custom Initialization

Sometimes, the initialization methods we need are not provided by the
deep learning framework. In the example below, we define an initializer
for any weight parameter $w$ using the following strange
distribution:

$$
(6.3.1)\[\begin{split}\begin{aligned}
 w \sim \begin{cases}
 U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
 0 & \textrm{ with probability } \frac{1}{2} \\
 U(-10, -5) & \textrm{ with probability } \frac{1}{4}
 \end{cases}
\end{aligned}\end{split}
$$

Again, we implement a `my_init` function to apply to `net`.

```python
def my_init(module):
 if type(module) == nn.Linear:
 print("Init", *[(name, param.shape)
 for name, param in module.named_parameters()][0])
 nn.init.uniform_(module.weight, -10, 10)
 module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

Note that we always have the option of setting parameters directly.

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

## 6.3.2. Summary

We can initialize parameters using built-in and custom initializers.
