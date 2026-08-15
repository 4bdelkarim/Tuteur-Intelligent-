---
source_url: https://d2l.ai/chapter_convolutional-modern/googlenet.html
title: 8.4. Multi-Branch Networks (GoogLeNet)
chapter: '8'
section_number: '8.4'
date: null
extractor: d2l
source_type: web
source: chapter_convolutional-modern_googlenet
---

# 8.4. Multi-Branch Networks (GoogLeNet)

In 2014, *GoogLeNet* won the ImageNet Challenge
(Szegedy et al., 2015), using a structure that combined the
strengths of NiN (Lin et al., 2013), repeated blocks
(Simonyan and Zisserman, 2014), and a cocktail of convolution
kernels. It was arguably also the first network that exhibited a clear
distinction among the stem (data ingest), body (data processing), and
head (prediction) in a CNN. This design pattern has persisted ever since
in the design of deep networks: the *stem* is given by the first two or
three convolutions that operate on the image. They extract low-level
features from the underlying images. This is followed by a *body* of
convolutional blocks. Finally, the *head* maps the features obtained so
far to the required classification, segmentation, detection, or tracking
problem at hand.

The key contribution in GoogLeNet was the design of the network body. It
solved the problem of selecting convolution kernels in an ingenious way.
While other works tried to identify which convolution, ranging from
$1 \times 1$ to $11 \times 11$ would be best, it simply
*concatenated* multi-branch convolutions. In what follows we introduce a
slightly simplified version of GoogLeNet: the original design included a
number of tricks for stabilizing training through intermediate loss
functions, applied to multiple layers of the network. They are no longer
necessary due to the availability of improved training algorithms.

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

## 8.4.1. Inception Blocks

The basic convolutional block in GoogLeNet is called an *Inception
block*, stemming from the meme “we need to go deeper” from the movie
*Inception*.

Fig. 8.4.1 Structure of the Inception block.

As depicted in Fig. 8.4.1, the inception block consists
of four parallel branches. The first three branches use convolutional
layers with window sizes of $1\times 1$, $3\times 3$, and
$5\times 5$ to extract information from different spatial sizes.
The middle two branches also add a $1\times 1$ convolution of the
input to reduce the number of channels, reducing the model’s complexity.
The fourth branch uses a $3\times 3$ max-pooling layer, followed
by a $1\times 1$ convolutional layer to change the number of
channels. The four branches all use appropriate padding to give the
input and output the same height and width. Finally, the outputs along
each branch are concatenated along the channel dimension and comprise
the block’s output. The commonly-tuned hyperparameters of the Inception
block are the number of output channels per layer, i.e., how to allocate
capacity among convolutions of different size.

```python
class Inception(nn.Module):
 # c1--c4 are the number of output channels for each branch
 def __init__(self, c1, c2, c3, c4, **kwargs):
 super(Inception, self).__init__(**kwargs)
 # Branch 1
 self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
 # Branch 2
 self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
 self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
 # Branch 3
 self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
 self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
 # Branch 4
 self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
 self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

 def forward(self, x):
 b1 = F.relu(self.b1_1(x))
 b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
 b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
 b4 = F.relu(self.b4_2(self.b4_1(x)))
 return torch.cat((b1, b2, b3, b4), dim=1)
```

To gain some intuition for why this network works so well, consider the
combination of the filters. They explore the image in a variety of
filter sizes. This means that details at different extents can be
recognized efficiently by filters of different sizes. At the same time,
we can allocate different amounts of parameters for different filters.

## 8.4.2. GoogLeNet Model

As shown in Fig. 8.4.2, GoogLeNet uses a stack of a
total of 9 inception blocks, arranged into three groups with max-pooling
in between, and global average pooling in its head to generate its
estimates. Max-pooling between inception blocks reduces the
dimensionality. At its stem, the first module is similar to AlexNet and
LeNet.

Fig. 8.4.2 The GoogLeNet architecture.

We can now implement GoogLeNet piece by piece. Let’s begin with the
stem. The first module uses a 64-channel $7\times 7$ convolutional
layer.

```python
class GoogleNet(d2l.Classifier):
 def b1(self):
 return nn.Sequential(
 nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
 nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

The second module uses two convolutional layers: first, a 64-channel
$1\times 1$ convolutional layer, followed by a $3\times 3$
convolutional layer that triples the number of channels. This
corresponds to the second branch in the Inception block and concludes
the design of the body. At this point we have 192 channels.

```python
@d2l.add_to_class(GoogleNet)
def b2(self):
 return nn.Sequential(
 nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
 nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

The third module connects two complete Inception blocks in series. The
number of output channels of the first Inception block is
$64+128+32+32=256$. This amounts to a ratio of the number of
output channels among the four branches of $2:4:1:1$. To achieve
this, we first reduce the input dimensions by $\frac{1}{2}$ and by
$\frac{1}{12}$ in the second and third branch respectively to
arrive at $96 = 192/2$ and $16 = 192/12$ channels
respectively.

The number of output channels of the second Inception block is increased
to $128+192+96+64=480$, yielding a ratio of
$128:192:96:64 = 4:6:3:2$. As before, we need to reduce the number
of intermediate dimensions in the second and third channel. A scale of
$\frac{1}{2}$ and $\frac{1}{8}$ respectively suffices,
yielding $128$ and $32$ channels respectively. This is
captured by the arguments of the following `Inception` block
constructors.

```python
@d2l.add_to_class(GoogleNet)
def b3(self):
 return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
 Inception(128, (128, 192), (32, 96), 64),
 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

The fourth module is more complicated. It connects five Inception blocks
in series, and they have $192+208+48+64=512$,
$160+224+64+64=512$, $128+256+64+64=512$,
$112+288+64+64=528$, and $256+320+128+128=832$ output
channels, respectively. The number of channels assigned to these
branches is similar to that in the third module: the second branch with
the $3\times 3$ convolutional layer outputs the largest number of
channels, followed by the first branch with only the $1\times 1$
convolutional layer, the third branch with the $5\times 5$
convolutional layer, and the fourth branch with the $3\times 3$
max-pooling layer. The second and third branches will first reduce the
number of channels according to the ratio. These ratios are slightly
different in different Inception blocks.

```python
@d2l.add_to_class(GoogleNet)
def b4(self):
 return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
 Inception(160, (112, 224), (24, 64), 64),
 Inception(128, (128, 256), (24, 64), 64),
 Inception(112, (144, 288), (32, 64), 64),
 Inception(256, (160, 320), (32, 128), 128),
 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

The fifth module has two Inception blocks with
$256+320+128+128=832$ and $384+384+128+128=1024$ output
channels. The number of channels assigned to each branch is the same as
that in the third and fourth modules, but differs in specific values. It
should be noted that the fifth block is followed by the output layer.
This block uses the global average pooling layer to change the height
and width of each channel to 1, just as in NiN. Finally, we turn the
output into a two-dimensional array followed by a fully connected layer
whose number of outputs is the number of label classes.

```python
@d2l.add_to_class(GoogleNet)
def b5(self):
 return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
 Inception(384, (192, 384), (48, 128), 128),
 nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
```

Now that we defined all blocks `b1` through `b5`, it is just a
matter of assembling them all into a full network.

```python
@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
 super(GoogleNet, self).__init__()
 self.save_hyperparameters()
 self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
 self.b5(), nn.LazyLinear(num_classes))
 self.net.apply(d2l.init_cnn)
```

The GoogLeNet model is computationally complex. Note the large number of
relatively arbitrary hyperparameters in terms of the number of channels
chosen, the number of blocks prior to dimensionality reduction, the
relative partitioning of capacity across channels, etc. Much of it is
due to the fact that at the time when GoogLeNet was introduced,
automatic tools for network definition or design exploration were not
yet available. For instance, by now we take it for granted that a
competent deep learning framework is capable of inferring
dimensionalities of input tensors automatically. At the time, many such
configurations had to be specified explicitly by the experimenter, thus
often slowing down active experimentation. Moreover, the tools needed
for automatic exploration were still in flux and initial experiments
largely amounted to costly brute-force exploration, genetic algorithms,
and similar strategies.

For now the only modification we will carry out is to reduce the input
height and width from 224 to 96 to have a reasonable training time on
Fashion-MNIST. This simplifies the computation. Let’s have a look at the
changes in the shape of the output between the various modules.

```python
model = GoogleNet().layer_summary((1, 1, 96, 96))
```

## 8.4.3. Training

As before, we train our model using the Fashion-MNIST dataset. We
transform it to $96 \times 96$ pixel resolution before invoking
the training procedure.

```python
model = GoogleNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

## 8.4.4. Discussion

A key feature of GoogLeNet is that it is actually *cheaper* to compute
than its predecessors while simultaneously providing improved accuracy.
This marks the beginning of a much more deliberate network design that
trades off the cost of evaluating a network with a reduction in errors.
It also marks the beginning of experimentation at a block level with
network design hyperparameters, even though it was entirely manual at
the time. We will revisit this topic in Section 8.8 when
discussing strategies for network structure exploration.

Over the following sections we will encounter a number of design choices
(e.g., batch normalization, residual connections, and channel grouping)
that allow us to improve networks significantly. For now, you can be
proud to have implemented what is arguably the first truly modern CNN.
