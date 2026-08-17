---
source_url: https://d2l.ai/chapter_computational-performance/async-computation.html
title: 13.2. Asynchronous Computation
chapter: '13'
section_number: '13.2'
date: null
extractor: d2l
source_type: web
---

# 13.2. Asynchronous Computation

Today’s computers are highly parallel systems, consisting of multiple
CPU cores (often multiple threads per core), multiple processing
elements per GPU, and often multiple GPUs per device. In short, we can
process many different things at the same time, often on different
devices. Unfortunately Python is not a great way of writing parallel and
asynchronous code, at least not without some extra help. After all,
Python is single-threaded and this is unlikely to change in the future.
Deep learning frameworks such as MXNet and TensorFlow adopt an
*asynchronous programming* model to improve performance, while PyTorch
uses Python’s own scheduler leading to a different performance
trade-off. For PyTorch, by default, GPU operations are asynchronous.
When you call a function that uses the GPU, the operations are enqueued
to the particular device, but not necessarily executed until later. This
allows us to execute more computations in parallel, including operations
on the CPU or other GPUs.

Hence, understanding how asynchronous programming works helps us to
develop more efficient programs, by proactively reducing computational
requirements and mutual dependencies. This allows us to reduce memory
overhead and increase processor utilization.

```python
import os
import subprocess
import numpy
import torch
from torch import nn
from d2l import torch as d2l
```

## 13.2.1. Asynchrony via Backend

For a warmup consider the following toy problem: we want to generate a
random matrix and multiply it. Let’s do that both in NumPy and in
PyTorch tensor to see the difference. Note that PyTorch `tensor` is
defined on a GPU.

```python
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
 for _ in range(10):
 a = numpy.random.normal(size=(1000, 1000))
 b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
 for _ in range(10):
 a = torch.randn(size=(1000, 1000), device=device)
 b = torch.mm(a, a)
```

The benchmark output via PyTorch is orders of magnitude faster. NumPy
dot product is executed on the CPU processor while PyTorch matrix
multiplication is executed on GPU and hence the latter is expected to be
much faster. But the huge time difference suggests something else must
be going on. By default, GPU operations are asynchronous in PyTorch.
Forcing PyTorch to finish all computation prior to returning shows what
happened previously: computation is being executed by the backend while
the frontend returns control to Python.

```python
with d2l.Benchmark():
 for _ in range(10):
 a = torch.randn(size=(1000, 1000), device=device)
 b = torch.mm(a, a)
 torch.cuda.synchronize(device)
```

Broadly speaking, PyTorch has a frontend for direct interaction with the
users, e.g., via Python, as well as a backend used by the system to
perform the computation. As shown in Fig. 13.2.1, users
can write PyTorch programs in various frontend languages, such as Python
and C++. Regardless of the frontend programming language used, the
execution of PyTorch programs occurs primarily in the backend of C++
implementations. Operations issued by the frontend language are passed
on to the backend for execution. The backend manages its own threads
that continuously collect and execute queued tasks. Note that for this
to work the backend must be able to keep track of the dependencies
between various steps in the computational graph. Hence, it is not
possible to parallelize operations that depend on each other.

Fig. 13.2.1 Programming language frontends and deep learning framework backends.

Let’s look at another toy example to understand the dependency graph a
bit better.

```python
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

Fig. 13.2.2 The backend tracks dependencies between various steps in the
computational graph.

The code snippet above is also illustrated in
Fig. 13.2.2. Whenever the Python frontend thread executes
one of the first three statements, it simply returns the task to the
backend queue. When the last statement’s results need to be *printed*,
the Python frontend thread will wait for the C++ backend thread to
finish computing the result of the variable `z`. One benefit of this
design is that the Python frontend thread does not need to perform
actual computations. Thus, there is little impact on the program’s
overall performance, regardless of Python’s performance.
Fig. 13.2.3 illustrates how frontend and backend interact.

Fig. 13.2.3 Interactions of the frontend and backend.

## 13.2.2. Barriers and Blockers

## 13.2.3. Improving Computation

## 13.2.4. Summary

- Deep learning frameworks may decouple the Python frontend from an
 execution backend. This allows for fast asynchronous insertion of
 commands into the backend and associated parallelism.
- Asynchrony leads to a rather responsive frontend. However, use
 caution not to overfill the task queue since it may lead to excessive
 memory consumption. It is recommended to synchronize for each
 minibatch to keep frontend and backend approximately synchronized.
- Chip vendors offer sophisticated performance analysis tools to obtain
 a much more fine-grained insight into the efficiency of deep
 learning.
