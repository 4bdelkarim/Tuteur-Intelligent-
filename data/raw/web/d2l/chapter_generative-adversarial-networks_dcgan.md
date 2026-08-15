---
source_url: "https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html"
title: "20.2. Deep Convolutional Generative Adversarial Networks"
chapter: "20"
section_number: "20.2"
date: null
extractor: "d2l"
---

# 20.2. Deep Convolutional Generative Adversarial Networks

In Section 20.1, we introduced the basic ideas behind how
GANs work. We showed that they can draw samples from some simple,
easy-to-sample distribution, like a uniform or normal distribution, and
transform them into samples that appear to match the distribution of
some dataset. And while our example of matching a 2D Gaussian
distribution got the point across, it is not especially exciting.

In this section, we will demonstrate how you can use GANs to generate
photorealistic images. We will be basing our models on the deep
convolutional GANs (DCGAN) introduced in
Radford *et al.* (2015). We will borrow the convolutional
architecture that have proven so successful for discriminative computer
vision problems and show how via GANs, they can be leveraged to generate
photorealistic images.

```python
import warnings
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
```

## 20.2.1. The Pokemon Dataset

The dataset we will use is a collection of Pokemon sprites obtained from
[pokemondb](https://pokemondb.net/sprites). First download, extract
and load this dataset.

```python
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)
```

We resize each image into $64\times 64$. The `ToTensor`
transformation will project the pixel value into $[0, 1]$, while
our generator will use the tanh function to obtain outputs in
$[-1, 1]$. Therefore we normalize the data with $0.5$ mean
and $0.5$ standard deviation to match the value range.

```python
batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

Let’s visualize the first 20 images.

```python
warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[:20,:,:,:].permute(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

![../_images/output_dcgan_2541de_39_0.svg](../_images/output_dcgan_2541de_39_0.svg)

## 20.2.2. The Generator

The generator needs to map the noise variable
$\mathbf z\in\mathbb R^d$, a length-$d$ vector, to a RGB
image with width and height to be $64\times 64$ . In
Section 14.11 we introduced the fully convolutional network that
uses transposed convolution layer (refer to
Section 14.10) to enlarge input size. The basic block
of the generator contains a transposed convolution layer followed by the
batch normalization and ReLU activation.

```python
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

In default, the transposed convolution layer uses a
$k_h = k_w = 4$ kernel, a $s_h = s_w = 2$ strides, and a
$p_h = p_w = 1$ padding. With a input shape of
$n_h^{'} \times n_w^{'} = 16 \times 16$, the generator block will
double input’s width and height.

$$
(20.2.1)\[\begin{split}\begin{aligned}
n_h^{'} \times n_w^{'} &= [(n_h k_h - (n_h-1)(k_h-s_h)- 2p_h] \times [(n_w k_w - (n_w-1)(k_w-s_w)- 2p_w]\\
  &= [(k_h + s_h (n_h-1)- 2p_h] \times [(k_w + s_w (n_w-1)- 2p_w]\\
  &= [(4 + 2 \times (16-1)- 2 \times 1] \times [(4 + 2 \times (16-1)- 2 \times 1]\\
  &= 32 \times 32 .\\
\end{aligned}\end{split}
$$

```python
x = torch.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk(x).shape
```

If changing the transposed convolution layer to a $4\times 4$
kernel, $1\times 1$ strides and zero padding. With a input size of
$1 \times 1$, the output will have its width and height increased
by 3 respectively.

```python
x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk(x).shape
```

The generator consists of four basic blocks that increase input’s both
width and height from 1 to 32. At the same time, it first projects the
latent variable into $64\times 8$ channels, and then halve the
channels each time. At last, a transposed convolution layer is used to
generate the output. It further doubles the width and height to match
the desired $64\times 64$ shape, and reduces the channel size to
$3$. The tanh activation function is applied to project output
values into the $(-1, 1)$ range.

```python
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G*8,
            strides=1, padding=0),                  # Output: (64 * 8, 4, 4)
    G_block(in_channels=n_G*8, out_channels=n_G*4), # Output: (64 * 4, 8, 8)
    G_block(in_channels=n_G*4, out_channels=n_G*2), # Output: (64 * 2, 16, 16)
    G_block(in_channels=n_G*2, out_channels=n_G),   # Output: (64, 32, 32)
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                       kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh())  # Output: (3, 64, 64)
```

Generate a 100 dimensional latent variable to verify the generator’s
output shape.

```python
x = torch.zeros((1, 100, 1, 1))
net_G(x).shape
```

## 20.2.3. Discriminator

The discriminator is a normal convolutional network network except that
it uses a leaky ReLU as its activation function. Given
$\alpha \in[0, 1]$, its definition is

$$
(20.2.2)\[\begin{split}\textrm{leaky ReLU}(x) = \begin{cases}x & \textrm{if}\ x > 0\\ \alpha x &\textrm{otherwise}\end{cases}.\end{split}
$$

As it can be seen, it is normal ReLU if $\alpha=0$, and an
identity function if $\alpha=1$. For $\alpha \in (0, 1)$,
leaky ReLU is a nonlinear function that give a non-zero output for a
negative input. It aims to fix the “dying ReLU” problem that a neuron
might always output a negative value and therefore cannot make any
progress since the gradient of ReLU is 0.

```python
alphas = [0, .2, .4, .6, .8, 1]
x = torch.arange(-2, 1, 0.1)
Y = [nn.LeakyReLU(alpha)(x).detach().numpy() for alpha in alphas]
d2l.plot(x.detach().numpy(), Y, 'x', 'y', alphas)
```

![../_images/output_dcgan_2541de_111_0.svg](../_images/output_dcgan_2541de_111_0.svg)

The basic block of the discriminator is a convolution layer followed by
a batch normalization layer and a leaky ReLU activation. The
hyperparameters of the convolution layer are similar to the transpose
convolution layer in the generator block.

```python
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

A basic block with default settings will halve the width and height of
the inputs, as we demonstrated in Section 7.3. For example,
given a input shape $n_h = n_w = 16$, with a kernel shape
$k_h = k_w = 4$, a stride shape $s_h = s_w = 2$, and a
padding shape $p_h = p_w = 1$, the output shape will be:

$$
(20.2.3)\[\begin{split}\begin{aligned}
n_h^{'} \times n_w^{'} &= \lfloor(n_h-k_h+2p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+2p_w+s_w)/s_w\rfloor\\
  &= \lfloor(16-4+2\times 1+2)/2\rfloor \times \lfloor(16-4+2\times 1+2)/2\rfloor\\
  &= 8 \times 8 .\\
\end{aligned}\end{split}
$$

```python
x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk(x).shape
```

The discriminator is a mirror of the generator.

```python
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),  # Output: (64, 32, 32)
    D_block(in_channels=n_D, out_channels=n_D*2),  # Output: (64 * 2, 16, 16)
    D_block(in_channels=n_D*2, out_channels=n_D*4),  # Output: (64 * 4, 8, 8)
    D_block(in_channels=n_D*4, out_channels=n_D*8),  # Output: (64 * 8, 4, 4)
    nn.Conv2d(in_channels=n_D*8, out_channels=1,
              kernel_size=4, bias=False))  # Output: (1, 1, 1)
```

It uses a convolution layer with output channel $1$ as the last
layer to obtain a single prediction value.

```python
x = torch.zeros((1, 3, 64, 64))
net_D(x).shape
```

## 20.2.4. Training

Compared to the basic GAN in Section 20.1, we use the same
learning rate for both generator and discriminator since they are
similar to each other. In addition, we change $\beta_1$ in Adam
(Section 12.10) from $0.9$ to $0.5$. It decreases the
smoothness of the momentum, the exponentially weighted moving average of
past gradients, to take care of the rapid changing gradients because the
generator and the discriminator fight with each other. Besides, the
random generated noise `Z`, is a 4-D tensor and we are using GPU to
accelerate the computation.

```python
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

We train the model with a small number of epochs just for demonstration.
For better performance, the variable `num_epochs` can be set to a
larger number.

```python
latent_dim, lr, num_epochs = 100, 0.005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

![../_images/output_dcgan_2541de_183_1.svg](../_images/output_dcgan_2541de_183_1.svg)

## 20.2.5. Summary

- DCGAN architecture has four convolutional layers for the
  Discriminator and four “fractionally-strided” convolutional layers
  for the Generator.
- The Discriminator is a 4-layer strided convolutions with batch
  normalization (except its input layer) and leaky ReLU activations.
- Leaky ReLU is a nonlinear function that give a non-zero output for a
  negative input. It aims to fix the “dying ReLU” problem and helps the
  gradients flow easier through the architecture.

## 20.2.6. Exercises

1. What will happen if we use standard ReLU activation rather than leaky
   ReLU?
2. Apply DCGAN on Fashion-MNIST and see which category works well and
   which does not.
