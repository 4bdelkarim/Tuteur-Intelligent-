---
source_url: https://d2l.ai/chapter_builders-guide/use-gpu.html
title: 6.7. GPUs
chapter: '6'
section_number: '6.7'
date: null
extractor: d2l
source_type: web
source: chapter_builders-guide_use-gpu
---

# 6.7. GPUs

In `tab_intro_decade`, we illustrated the rapid growth of
computation over the past two decades. In a nutshell, GPU performance
has increased by a factor of 1000 every decade since 2000. This offers
great opportunities but it also suggests that there was significant
demand for such performance.

In this section, we begin to discuss how to harness this computational
performance for your research. First by using a single GPU and at a
later point, how to use multiple GPUs and multiple servers (with
multiple GPUs).

Specifically, we will discuss how to use a single NVIDIA GPU for
calculations. First, make sure you have at least one NVIDIA GPU
installed. Then, download the NVIDIA driver and
CUDA and follow the
prompts to set the appropriate path. Once these preparations are
complete, the `nvidia-smi` command can be used to view the graphics
card information.

In PyTorch, every array has a device; we often refer it as a *context*.
So far, by default, all variables and associated computation have been
assigned to the CPU. Typically, other contexts might be various GPUs.
Things can get even hairier when we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently, we can minimize the time
spent transferring data between devices. For example, when training
neural networks on a server with a GPU, we typically prefer for the
model’s parameters to live on the GPU.

To run the programs in this section, you need at least two GPUs. Note
that this might be extravagant for most desktop computers but it is
easily available in the cloud, e.g., by using the AWS EC2 multi-GPU
instances. Almost all other sections do *not* require multiple GPUs, but
here we simply wish to illustrate data flow between different devices.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 6.7.1. Computing Devices

We can specify devices, such as CPUs and GPUs, for storage and
calculation. By default, tensors are created in the main memory and then
the CPU is used for calculations.

In PyTorch, the CPU and GPU can be indicated by `torch.device('cpu')`
and `torch.device('cuda')`. It should be noted that the `cpu` device
means all physical CPUs and memory. This means that PyTorch’s
calculations will try to use all CPU cores. However, a `gpu` device
only represents one card and the corresponding memory. If there are
multiple GPUs, we use `torch.device(f'cuda:{i}')` to represent the
$i^\textrm{th}$ GPU ($i$ starts at 0). Also, `gpu:0` and
`gpu` are equivalent.

```python
def cpu(): #@save
 """Get the CPU device."""
 return torch.device('cpu')

def gpu(i=0): #@save
 """Get a GPU device."""
 return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

We can query the number of available GPUs.

```python
def num_gpus(): #@save
 """Get the number of available GPUs."""
 return torch.cuda.device_count()

num_gpus()
```

Now we define two convenient functions that allow us to run code even if
the requested GPUs do not exist.

```python
def try_gpu(i=0): #@save
 """Return gpu(i) if exists, otherwise return cpu()."""
 if num_gpus() >= i + 1:
 return gpu(i)
 return cpu()

def try_all_gpus(): #@save
 """Return all available GPUs, or [cpu(),] if no GPU exists."""
 return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## 6.7.2. Tensors and GPUs

By default, tensors are created on the CPU. We can query the device
where the tensor is located.

```python
x = torch.tensor([1, 2, 3])
x.device
```

It is important to note that whenever we want to operate on multiple
terms, they need to be on the same device. For instance, if we sum two
tensors, we need to make sure that both arguments live on the same
device—otherwise the framework would not know where to store the result
or even how to decide where to perform the computation.

### 6.7.2.1. Storage on the GPU

There are several ways to store a tensor on the GPU. For example, we can
specify a storage device when creating a tensor. Next, we create the
tensor variable `X` on the first `gpu`. The tensor created on a GPU
only consumes the memory of this GPU. We can use the `nvidia-smi`
command to view GPU memory usage. In general, we need to make sure that
we do not create data that exceeds the GPU memory limit.

```python
X = torch.ones(2, 3, device=try_gpu())
X
```

Assuming that you have at least two GPUs, the following code will create
a random tensor, `Y`, on the second GPU.

```python
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

### 6.7.2.2. Copying

If we want to compute `X + Y`, we need to decide where to perform this
operation. For instance, as shown in Fig. 6.7.1, we can
transfer `X` to the second GPU and perform the operation there. *Do
not* simply add `X` and `Y`, since this will result in an exception.
The runtime engine would not know what to do: it cannot find data on the
same device and it fails. Since `Y` lives on the second GPU, we need
to move `X` there before we can add the two.

Fig. 6.7.1 Copy data to perform an operation on the same device.

```python
Z = X.cuda(1)
print(X)
print(Z)
```

Now that the data (both `Z` and `Y`) are on the same GPU), we can
add them up.

```python
Y + Z
```

But what if your variable `Z` already lived on your second GPU? What
happens if we still call `Z.cuda(1)`? It will return `Z` instead of
making a copy and allocating new memory.

```python
Z.cuda(1) is Z
```

### 6.7.2.3. Side Notes

People use GPUs to do machine learning because they expect them to be
fast. But transferring variables between devices is slow: much slower
than computation. So we want you to be 100% certain that you want to do
something slow before we let you do it. If the deep learning framework
just did the copy automatically without crashing then you might not
realize that you had written some slow code.

Transferring data is not only slow, it also makes parallelization a lot
more difficult, since we have to wait for data to be sent (or rather to
be received) before we can proceed with more operations. This is why
copy operations should be taken with great care. As a rule of thumb,
many small operations are much worse than one big operation. Moreover,
several operations at a time are much better than many single operations
interspersed in the code unless you know what you are doing. This is the
case since such operations can block if one device has to wait for the
other before it can do something else. It is a bit like ordering your
coffee in a queue rather than pre-ordering it by phone and finding out
that it is ready when you are.

Last, when we print tensors or convert tensors to the NumPy format, if
the data is not in the main memory, the framework will copy it to the
main memory first, resulting in additional transmission overhead. Even
worse, it is now subject to the dreaded global interpreter lock that
makes everything wait for Python to complete.

## 6.7.3. Neural Networks and GPUs

Similarly, a neural network model can specify devices. The following
code puts the model parameters on the GPU.

```python
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

We will see many more examples of how to run models on GPUs in the
following chapters, simply because the models will become somewhat more
computationally intensive.

For example, when the input is a tensor on the GPU, the model will
calculate the result on the same GPU.

```python
net(X)
```

Let’s confirm that the model parameters are stored on the same GPU.

```python
net[0].weight.data.device
```

Let the trainer support GPU.

```python
@d2l.add_to_class(d2l.Trainer) #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
 self.save_hyperparameters()
 self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer) #@save
def prepare_batch(self, batch):
 if self.gpus:
 batch = [a.to(self.gpus[0]) for a in batch]
 return batch

@d2l.add_to_class(d2l.Trainer) #@save
def prepare_model(self, model):
 model.trainer = self
 model.board.xlim = [0, self.max_epochs]
 if self.gpus:
 model.to(self.gpus[0])
 self.model = model
```

In short, as long as all data and parameters are on the same device, we
can learn models efficiently. In the following chapters we will see
several such examples.

## 6.7.4. Summary

We can specify devices for storage and calculation, such as the CPU or
GPU. By default, data is created in the main memory and then uses the
CPU for calculations. The deep learning framework requires all input
data for calculation to be on the same device, be it CPU or the same
GPU. You can lose significant performance by moving data without care. A
typical mistake is as follows: computing the loss for every minibatch on
the GPU and reporting it back to the user on the command line (or
logging it in a NumPy `ndarray`) will trigger a global interpreter
lock which stalls all GPUs. It is much better to allocate memory for
logging inside the GPU and only move larger logs.
