---
source_url: https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html
title: 3.3. Synthetic Regression Data
chapter: '3'
section_number: '3.3'
date: null
extractor: d2l
source_type: web
---

# 3.3. Synthetic Regression Data

Machine learning is all about extracting information from data. So you
might wonder, what could we possibly learn from synthetic data? While we
might not care intrinsically about the patterns that we ourselves baked
into an artificial data generating model, such datasets are nevertheless
useful for didactic purposes, helping us to evaluate the properties of
our learning algorithms and to confirm that our implementations work as
expected. For example, if we create data for which the correct
parameters are known *a priori*, then we can check that our model can in
fact recover them.

```python
%matplotlib inline
import random
import torch
from d2l import torch as d2l
```

## 3.3.1. Generating the Dataset

For this example, we will work in low dimension for succinctness. The
following code snippet generates 1000 examples with 2-dimensional
features drawn from a standard normal distribution. The resulting design
matrix $\mathbf{X}$ belongs to $\mathbb{R}^{1000 \times 2}$.
We generate each label by applying a *ground truth* linear function,
corrupting them via additive noise $\boldsymbol{\epsilon}$, drawn
independently and identically for each example:

$$
(3.3.1)\[\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.
$$

For convenience we assume that $\boldsymbol{\epsilon}$ is drawn
from a normal distribution with mean $\mu= 0$ and standard
deviation $\sigma = 0.01$. Note that for object-oriented design we
add the code to the `__init__` method of a subclass of
`d2l.DataModule` (introduced in Section 3.2.3). It is
good practice to allow the setting of any additional hyperparameters. We
accomplish this with `save_hyperparameters()`. The `batch_size` will
be determined later.

```python
class SyntheticRegressionData(d2l.DataModule): #@save
 """Synthetic data for linear regression."""
 def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
 batch_size=32):
 super().__init__()
 self.save_hyperparameters()
 n = num_train + num_val
 self.X = torch.randn(n, len(w))
 noise = torch.randn(n, 1) * noise
 self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
```

Below, we set the true parameters to $\mathbf{w} = [2, -3.4]^\top$
and $b = 4.2$. Later, we can check our estimated parameters
against these *ground truth* values.

```python
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
```

Each row in `features` consists of a vector in $\mathbb{R}^2$
and each row in `labels` is a scalar. Let’s have a look at the first
entry.

```python
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## 3.3.2. Reading the Dataset

Training machine learning models often requires multiple passes over a
dataset, grabbing one minibatch of examples at a time. This data is then
used to update the model. To illustrate how this works, we implement the
`get_dataloader` method, registering it in the
`SyntheticRegressionData` class via `add_to_class` (introduced in
Section 3.2.1). It takes a batch size, a matrix of
features, and a vector of labels, and generates minibatches of size
`batch_size`. As such, each minibatch consists of a tuple of features
and labels. Note that we need to be mindful of whether we’re in training
or validation mode: in the former, we will want to read the data in
random order, whereas for the latter, being able to read data in a
pre-defined order may be important for debugging purposes.

```python
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
 if train:
 indices = list(range(0, self.num_train))
 # The examples are read in random order
 random.shuffle(indices)
 else:
 indices = list(range(self.num_train, self.num_train+self.num_val))
 for i in range(0, len(indices), self.batch_size):
 batch_indices = torch.tensor(indices[i: i+self.batch_size])
 yield self.X[batch_indices], self.y[batch_indices]
```

To build some intuition, let’s inspect the first minibatch of data. Each
minibatch of features provides us with both its size and the
dimensionality of input features. Likewise, our minibatch of labels will
have a matching shape given by `batch_size`.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

While seemingly innocuous, the invocation of
`iter(data.train_dataloader())` illustrates the power of Python’s
object-oriented design. Note that we added a method to the
`SyntheticRegressionData` class *after* creating the `data` object.
Nonetheless, the object benefits from the *ex post facto* addition of
functionality to the class.

Throughout the iteration we obtain distinct minibatches until the entire
dataset has been exhausted (try this). While the iteration implemented
above is good for didactic purposes, it is inefficient in ways that
might get us into trouble with real problems. For example, it requires
that we load all the data in memory and that we perform lots of random
memory access. The built-in iterators implemented in a deep learning
framework are considerably more efficient and they can deal with sources
such as data stored in files, data received via a stream, and data
generated or processed on the fly. Next let’s try to implement the same
method using built-in iterators.

## 3.3.3. Concise Implementation of the Data Loader

Rather than writing our own iterator, we can call the existing API in a
framework to load data. As before, we need a dataset with features `X`
and labels `y`. Beyond that, we set `batch_size` in the built-in
data loader and let it take care of shuffling examples efficiently.

```python
@d2l.add_to_class(d2l.DataModule) #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
 tensors = tuple(a[indices] for a in tensors)
 dataset = torch.utils.data.TensorDataset(*tensors)
 return torch.utils.data.DataLoader(dataset, self.batch_size,
 shuffle=train)

@d2l.add_to_class(SyntheticRegressionData) #@save
def get_dataloader(self, train):
 i = slice(0, self.num_train) if train else slice(self.num_train, None)
 return self.get_tensorloader((self.X, self.y), train, i)
```

The new data loader behaves just like the previous one, except that it
is more efficient and has some added functionality.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

For instance, the data loader provided by the framework API supports the
built-in `__len__` method, so we can query its length, i.e., the
number of batches.

```python
len(data.train_dataloader())
```

## 3.3.4. Summary

Data loaders are a convenient way of abstracting out the process of
loading and manipulating data. This way the same machine learning
*algorithm* is capable of processing many different types and sources of
data without the need for modification. One of the nice things about
data loaders is that they can be composed. For instance, we might be
loading images and then have a postprocessing filter that crops them or
modifies them in other ways. As such, data loaders can be used to
describe an entire data processing pipeline.

As for the model itself, the two-dimensional linear model is about the
simplest we might encounter. It lets us test out the accuracy of
regression models without worrying about having insufficient amounts of
data or an underdetermined system of equations. We will put this to good
use in the next section.
