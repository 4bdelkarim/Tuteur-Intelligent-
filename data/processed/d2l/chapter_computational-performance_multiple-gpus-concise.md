---
source_url: https://d2l.ai/chapter_computational-performance/multiple-gpus-concise.html
title: 13.6. Concise Implementation for Multiple GPUs
chapter: '13'
section_number: '13.6'
date: null
extractor: d2l
source_type: web
---

# 13.6. Concise Implementation for Multiple GPUs

Implementing parallelism from scratch for every new model is no fun.
Moreover, there is significant benefit in optimizing synchronization
tools for high performance. In the following we will show how to do this
using high-level APIs of deep learning frameworks. The mathematics and
the algorithms are the same as in Section 13.5. Quite
unsurprisingly you will need at least two GPUs to run code of this
section.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 13.6.1. A Toy Network

Let’s use a slightly more meaningful network than LeNet from
Section 13.5 that is still sufficiently easy and quick to
train. We pick a ResNet-18 variant (He et al., 2016). Since
the input images are tiny we modify it slightly. In particular, the
difference from Section 8.6 is that we use a smaller
convolution kernel, stride, and padding at the beginning. Moreover, we
remove the max-pooling layer.

```python
#@save
def resnet18(num_classes, in_channels=1):
 """A slightly modified ResNet-18 model."""
 def resnet_block(in_channels, out_channels, num_residuals,
 first_block=False):
 blk = []
 for i in range(num_residuals):
 if i == 0 and not first_block:
 blk.append(d2l.Residual(out_channels, use_1x1conv=True,
 strides=2))
 else:
 blk.append(d2l.Residual(out_channels))
 return nn.Sequential(*blk)

 # This model uses a smaller convolution kernel, stride, and padding and
 # removes the max-pooling layer
 net = nn.Sequential(
 nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
 nn.BatchNorm2d(64),
 nn.ReLU())
 net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
 net.add_module("resnet_block2", resnet_block(64, 128, 2))
 net.add_module("resnet_block3", resnet_block(128, 256, 2))
 net.add_module("resnet_block4", resnet_block(256, 512, 2))
 net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
 net.add_module("fc", nn.Sequential(nn.Flatten(),
 nn.Linear(512, num_classes)))
 return net
```

## 13.6.2. Network Initialization

We will initialize the network inside the training loop. For a refresher
on initialization methods see Section 5.4.

```python
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

## 13.6.3. Training

As before, the training code needs to perform several basic functions
for efficient parallelism:

- Network parameters need to be initialized across all devices.
- While iterating over the dataset minibatches are to be divided across
 all devices.
- We compute the loss and its gradient in parallel across devices.
- Gradients are aggregated and parameters are updated accordingly.

In the end we compute the accuracy (again in parallel) to report the
final performance of the network. The training routine is quite similar
to implementations in previous chapters, except that we need to split
and aggregate data.

```python
def train(net, num_gpus, batch_size, lr):
 train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
 devices = [d2l.try_gpu(i) for i in range(num_gpus)]
 def init_weights(module):
 if type(module) in [nn.Linear, nn.Conv2d]:
 nn.init.normal_(module.weight, std=0.01)
 net.apply(init_weights)
 # Set the model on multiple GPUs
 net = nn.DataParallel(net, device_ids=devices)
 trainer = torch.optim.SGD(net.parameters(), lr)
 loss = nn.CrossEntropyLoss()
 timer, num_epochs = d2l.Timer(), 10
 animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
 for epoch in range(num_epochs):
 net.train()
 timer.start()
 for X, y in train_iter:
 trainer.zero_grad()
 X, y = X.to(devices[0]), y.to(devices[0])
 l = loss(net(X), y)
 l.backward()
 trainer.step()
 timer.stop()
 animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
 print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
 f'on {str(devices)}')
```

Let’s see how this works in practice. As a warm-up we train the network
on a single GPU.

```python
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Next we use 2 GPUs for training. Compared with LeNet evaluated in
Section 13.5, the model for ResNet-18 is considerably more
complex. This is where parallelization shows its advantage. The time for
computation is meaningfully larger than the time for synchronizing
parameters. This improves scalability since the overhead for
parallelization is less relevant.

```python
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 13.6.4. Summary

- Data is automatically evaluated on the devices where the data can be
 found.
- Take care to initialize the networks on each device before trying to
 access the parameters on that device. Otherwise you will encounter an
 error.
- The optimization algorithms automatically aggregate over multiple
 GPUs.
