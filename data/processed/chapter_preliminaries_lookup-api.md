---
source_url: https://d2l.ai/chapter_preliminaries/lookup-api.html
title: 2.7. Documentation
chapter: '2'
section_number: '2.7'
date: null
extractor: d2l
source_type: web
source: chapter_preliminaries_lookup-api
---

# 2.7. Documentation

While we cannot possibly introduce every single PyTorch function and
class (and the information might become outdated quickly), the API
documentation and
additional
tutorials
and examples provide such documentation. This section provides some
guidance for how to explore the PyTorch API.

```python
import torch
```

## 2.7.1. Functions and Classes in a Module

To know which functions and classes can be called in a module, we invoke
the `dir` function. For instance, we can query all properties in the
module for generating random numbers:

```python
print(dir(torch.distributions))
```

Generally, we can ignore functions that start and end with `__`
(special objects in Python) or functions that start with a single
`_`(usually internal functions). Based on the remaining function or
attribute names, we might hazard a guess that this module offers various
methods for generating random numbers, including sampling from the
uniform distribution (`uniform`), normal distribution (`normal`),
and multinomial distribution (`multinomial`).

## 2.7.2. Specific Functions and Classes

For specific instructions on how to use a given function or class, we
can invoke the `help` function. As an example, let’s explore the usage
instructions for tensors’ `ones` function.

```python
help(torch.ones)
```

```
Help on built-in function ones in module torch:

ones(...)
 ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

 Returns a tensor filled with the scalar value 1, with the shape defined
 by the variable argument size.

 Args:
 size (int...): a sequence of integers defining the shape of the output tensor.
 Can be a variable number of arguments or a collection like a list or tuple.

 Keyword arguments:
 out (Tensor, optional): the output tensor.
 dtype (torch.dtype, optional): the desired data type of returned tensor.
 Default: if None, uses a global default (see torch.set_default_tensor_type()).
 layout (torch.layout, optional): the desired layout of returned Tensor.
 Default: torch.strided.
 device (torch.device, optional): the desired device of returned tensor.
 Default: if None, uses the current device for the default tensor type
 (see torch.set_default_tensor_type()). device will be the CPU
 for CPU tensor types and the current CUDA device for CUDA tensor types.
 requires_grad (bool, optional): If autograd should record operations on the
 returned tensor. Default: False.

 Example::

 >>> torch.ones(2, 3)
 tensor([[ 1., 1., 1.],
 [ 1., 1., 1.]])

 >>> torch.ones(5)
 tensor([ 1., 1., 1., 1., 1.])
```

From the documentation, we can see that the `ones` function creates a
new tensor with the specified shape and sets all the elements to the
value of 1. Whenever possible, you should run a quick test to confirm
your interpretation:

```python
torch.ones(4)
```

In the Jupyter notebook, we can use `?` to display the document in
another window. For example, `list?` will create content that is
almost identical to `help(list)`, displaying it in a new browser
window. In addition, if we use two question marks, such as `list??`,
the Python code implementing the function will also be displayed.

The official documentation provides plenty of descriptions and examples
that are beyond this book. We emphasize important use cases that will
get you started quickly with practical problems, rather than
completeness of coverage. We also encourage you to study the source code
of the libraries to see examples of high-quality implementations of
production code. By doing this you will become a better engineer in
addition to becoming a better scientist.
