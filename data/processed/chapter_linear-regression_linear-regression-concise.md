---
source_url: https://d2l.ai/chapter_linear-regression/linear-regression-concise.html
title: 3.5. Concise Implementation of Linear Regression
chapter: '3'
section_number: '3.5'
date: null
extractor: d2l
source_type: web
source: chapter_linear-regression_linear-regression-concise
---

# 3.5. Concise Implementation of Linear Regression

Deep learning has witnessed a sort of Cambrian explosion over the past
decade. The sheer number of techniques, applications and algorithms by
far surpasses the progress of previous decades. This is due to a
fortuitous combination of multiple factors, one of which is the powerful
free tools offered by a number of open-source deep learning frameworks.
Theano (Bergstra et al., 2010), DistBelief
(Dean et al., 2012), and Caffe
(Jia et al., 2014) arguably represent the first
generation of such models that found widespread adoption. In contrast to
earlier (seminal) works like SN2 (Simulateur Neuristique)
(Bottou and Le Cun, 1988), which provided a Lisp-like programming
experience, modern frameworks offer automatic differentiation and the
convenience of Python. These frameworks allow us to automate and
modularize the repetitive work of implementing gradient-based learning
algorithms.

In Section 3.4, we relied only on (i) tensors for
data storage and linear algebra; and (ii) automatic differentiation for
calculating gradients. In practice, because data iterators, loss
functions, optimizers, and neural network layers are so common, modern
libraries implement these components for us as well. In this section, we
will show you how to implement the linear regression model from
Section 3.4 concisely by using high-level APIs of
deep learning frameworks.

```python
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
```

## 3.5.1. Defining the Model

When we implemented linear regression from scratch in
Section 3.4, we defined our model parameters
explicitly and coded up the calculations to produce output using basic
linear algebra operations. You *should* know how to do this. But once
your models get more complex, and once you have to do this nearly every
day, you will be glad of the assistance. The situation is similar to
coding up your own blog from scratch. Doing it once or twice is
rewarding and instructive, but you would be a lousy web developer if you
spent a month reinventing the wheel.

For standard operations, we can use a framework’s predefined layers,
which allow us to focus on the layers used to construct the model rather
than worrying about their implementation. Recall the architecture of a
single-layer network as described in Fig. 3.1.2. The
layer is called *fully connected*, since each of its inputs is connected
to each of its outputs by means of a matrix–vector multiplication.

In PyTorch, the fully connected layer is defined in `Linear` and
`LazyLinear` classes (available since version 1.8.0). The latter
allows users to specify *merely* the output dimension, while the former
additionally asks for how many inputs go into this layer. Specifying
input shapes is inconvenient and may require nontrivial calculations
(such as in convolutional layers). Thus, for simplicity, we will use
such “lazy” layers whenever we can.

```python
class LinearRegression(d2l.Module): #@save
 """The linear regression model implemented with high-level APIs."""
 def __init__(self, lr):
 super().__init__()
 self.save_hyperparameters()
 self.net = nn.LazyLinear(1)
 self.net.weight.data.normal_(0, 0.01)
 self.net.bias.data.fill_(0)
```

In the `forward` method we just invoke the built-in `__call__`
method of the predefined layers to compute the outputs.

```python
@d2l.add_to_class(LinearRegression) #@save
def forward(self, X):
 return self.net(X)
```

## 3.5.2. Defining the Loss Function

The `MSELoss` class computes the mean squared error (without the
$1/2$ factor in (3.1.5)). By default, `MSELoss`
returns the average loss over examples. It is faster (and easier to use)
than implementing our own.

```python
@d2l.add_to_class(LinearRegression) #@save
def loss(self, y_hat, y):
 fn = nn.MSELoss()
 return fn(y_hat, y)
```

## 3.5.3. Defining the Optimization Algorithm

Minibatch SGD is a standard tool for optimizing neural networks and thus
PyTorch supports it alongside a number of variations on this algorithm
in the `optim` module. When we instantiate an `SGD` instance, we
specify the parameters to optimize over, obtainable from our model via
`self.parameters()`, and the learning rate (`self.lr`) required by
our optimization algorithm.

```python
@d2l.add_to_class(LinearRegression) #@save
def configure_optimizers(self):
 return torch.optim.SGD(self.parameters(), self.lr)
```

## 3.5.4. Training

You might have noticed that expressing our model through high-level APIs
of a deep learning framework requires fewer lines of code. We did not
have to allocate parameters individually, define our loss function, or
implement minibatch SGD. Once we start working with much more complex
models, the advantages of the high-level API will grow considerably.

Now that we have all the basic pieces in place, the training loop itself
is the same as the one we implemented from scratch. So we just call the
`fit` method (introduced in Section 3.2.4), which
relies on the implementation of the `fit_epoch` method in
Section 3.4, to train our model.

```python
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Below, we compare the model parameters learned by training on finite
data and the actual parameters that generated our dataset. To access
parameters, we access the weights and bias of the layer that we need. As
in our implementation from scratch, note that our estimated parameters
are close to their true counterparts.

```python
@d2l.add_to_class(LinearRegression) #@save
def get_w_b(self):
 return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## 3.5.5. Summary

This section contains the first implementation of a deep network (in
this book) to tap into the conveniences afforded by modern deep learning
frameworks, such as MXNet (Chen et al., 2015), JAX
(Frostig et al., 2018), PyTorch
(Paszke et al., 2019), and Tensorflow
(Abadi et al., 2016). We used framework defaults for
loading data, defining a layer, a loss function, an optimizer and a
training loop. Whenever the framework provides all necessary features,
it is generally a good idea to use them, since the library
implementations of these components tend to be heavily optimized for
performance and properly tested for reliability. At the same time, try
not to forget that these modules *can* be implemented directly. This is
especially important for aspiring researchers who wish to live on the
leading edge of model development, where you will be inventing new
components that cannot possibly exist in any current library.

In PyTorch, the `data` module provides tools for data processing, the
`nn` module defines a large number of neural network layers and common
loss functions. We can initialize the parameters by replacing their
values with methods ending with `_`. Note that we need to specify the
input dimensions of the network. While this is trivial for now, it can
have significant knock-on effects when we want to design complex
networks with many layers. Careful considerations of how to parametrize
these networks is needed to allow portability.
