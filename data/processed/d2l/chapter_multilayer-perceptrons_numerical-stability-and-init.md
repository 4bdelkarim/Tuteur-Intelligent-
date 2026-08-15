---
source_url: https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html
title: 5.4. Numerical Stability and Initialization
chapter: '5'
section_number: '5.4'
date: null
extractor: d2l
source_type: web
---

# 5.4. Numerical Stability and Initialization

Thus far, every model that we have implemented required that we
initialize its parameters according to some pre-specified distribution.
Until now, we took the initialization scheme for granted, glossing over
the details of how these choices are made. You might have even gotten
the impression that these choices are not especially important. On the
contrary, the choice of initialization scheme plays a significant role
in neural network learning, and it can be crucial for maintaining
numerical stability. Moreover, these choices can be tied up in
interesting ways with the choice of the nonlinear activation function.
Which function we choose and how we initialize parameters can determine
how quickly our optimization algorithm converges. Poor choices here can
cause us to encounter exploding or vanishing gradients while training.
In this section, we delve into these topics in greater detail and
discuss some useful heuristics that you will find useful throughout your
career in deep learning.

```python
%matplotlib inline
import torch
from d2l import torch as d2l
```

## 5.4.1. Vanishing and Exploding Gradients

Consider a deep network with $L$ layers, input $\mathbf{x}$
and output $\mathbf{o}$. With each layer $l$ defined by a
transformation $f_l$ parametrized by weights
$\mathbf{W}^{(l)}$, whose hidden layer output is
$\mathbf{h}^{(l)}$ (let $\mathbf{h}^{(0)} = \mathbf{x}$),
our network can be expressed as:

$$
(5.4.1)\[\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).
$$

If all the hidden layer output and the input are vectors, we can write
the gradient of $\mathbf{o}$ with respect to any set of parameters
$\mathbf{W}^{(l)}$ as follows:

$$
(5.4.2)\[\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.
$$

In other words, this gradient is the product of $L-l$ matrices
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$ and the gradient
vector $\mathbf{v}^{(l)}$. Thus we are susceptible to the same
problems of numerical underflow that often crop up when multiplying
together too many probabilities. When dealing with probabilities, a
common trick is to switch into log-space, i.e., shifting pressure from
the mantissa to the exponent of the numerical representation.
Unfortunately, our problem above is more serious: initially the matrices
$\mathbf{M}^{(l)}$ may have a wide variety of eigenvalues. They
might be small or large, and their product might be *very large* or
*very small*.

The risks posed by unstable gradients go beyond numerical
representation. Gradients of unpredictable magnitude also threaten the
stability of our optimization algorithms. We may be facing parameter
updates that are either (i) excessively large, destroying our model (the
*exploding gradient* problem); or (ii) excessively small (the *vanishing
gradient* problem), rendering learning impossible as parameters hardly
move on each update.

### 5.4.1.1. Vanishing Gradients

One frequent culprit causing the vanishing gradient problem is the
choice of the activation function $\sigma$ that is appended
following each layer’s linear operations. Historically, the sigmoid
function $1/(1 + \exp(-x))$ (introduced in Section 5.1)
was popular because it resembles a thresholding function. Since early
artificial neural networks were inspired by biological neural networks,
the idea of neurons that fire either *fully* or *not at all* (like
biological neurons) seemed appealing. Let’s take a closer look at the
sigmoid to see why it can cause vanishing gradients.

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
 legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

As you can see, the sigmoid’s gradient vanishes both when its inputs are
large and when they are small. Moreover, when backpropagating through
many layers, unless we are in the Goldilocks zone, where the inputs to
many of the sigmoids are close to zero, the gradients of the overall
product may vanish. When our network boasts many layers, unless we are
careful, the gradient will likely be cut off at some layer. Indeed, this
problem used to plague deep network training. Consequently, ReLUs, which
are more stable (but less neurally plausible), have emerged as the
default choice for practitioners.

### 5.4.1.2. Exploding Gradients

The opposite problem, when gradients explode, can be similarly vexing.
To illustrate this a bit better, we draw 100 Gaussian random matrices
and multiply them with some initial matrix. For the scale that we picked
(the choice of the variance $\sigma^2=1$), the matrix product
explodes. When this happens because of the initialization of a deep
network, we have no chance of getting a gradient descent optimizer to
converge.

```python
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
 M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

### 5.4.1.3. Breaking the Symmetry

Another problem in neural network design is the symmetry inherent in
their parametrization. Assume that we have a simple MLP with one hidden
layer and two units. In this case, we could permute the weights
$\mathbf{W}^{(1)}$ of the first layer and likewise permute the
weights of the output layer to obtain the same function. There is
nothing special differentiating the first and second hidden units. In
other words, we have permutation symmetry among the hidden units of each
layer.

This is more than just a theoretical nuisance. Consider the
aforementioned one-hidden-layer MLP with two hidden units. For
illustration, suppose that the output layer transforms the two hidden
units into only one output unit. Imagine what would happen if we
initialized all the parameters of the hidden layer as
$\mathbf{W}^{(1)} = c$ for some constant $c$. In this case,
during forward propagation either hidden unit takes the same inputs and
parameters producing the same activation which is fed to the output
unit. During backpropagation, differentiating the output unit with
respect to parameters $\mathbf{W}^{(1)}$ gives a gradient all of
whose elements take the same value. Thus, after gradient-based iteration
(e.g., minibatch stochastic gradient descent), all the elements of
$\mathbf{W}^{(1)}$ still take the same value. Such iterations
would never *break the symmetry* on their own and we might never be able
to realize the network’s expressive power. The hidden layer would behave
as if it had only a single unit. Note that while minibatch stochastic
gradient descent would not break this symmetry, dropout regularization
(to be introduced later) would!

## 5.4.2. Parameter Initialization

One way of addressing—or at least mitigating—the issues raised above is
through careful initialization. As we will see later, additional care
during optimization and suitable regularization can further enhance
stability.

### 5.4.2.1. Default Initialization

In the previous sections, e.g., in Section 3.5, we
used a normal distribution to initialize the values of our weights. If
we do not specify the initialization method, the framework will use a
default random initialization method, which often works well in practice
for moderate problem sizes.

### 5.4.2.2. Xavier Initialization

Let’s look at the scale distribution of an output $o_{i}$ for some
fully connected layer *without nonlinearities*. With
$n_\textrm{in}$ inputs $x_j$ and their associated weights
$w_{ij}$ for this layer, an output is given by

$$
(5.4.3)\[o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.
$$

The weights $w_{ij}$ are all drawn independently from the same
distribution. Furthermore, let’s assume that this distribution has zero
mean and variance $\sigma^2$. Note that this does not mean that
the distribution has to be Gaussian, just that the mean and variance
need to exist. For now, let’s assume that the inputs to the layer
$x_j$ also have zero mean and variance $\gamma^2$ and that
they are independent of $w_{ij}$ and independent of each other. In
this case, we can compute the mean of $o_i$:

$$
(5.4.4)\[\begin{split}\begin{aligned}
 E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}\end{split}
$$

and the variance:

$$
(5.4.5)\[\begin{split}\begin{aligned}
 \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
 & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
 & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
 & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}\end{split}
$$

One way to keep the variance fixed is to set
$n_\textrm{in} \sigma^2 = 1$. Now consider backpropagation. There
we face a similar problem, albeit with gradients being propagated from
the layers closer to the output. Using the same reasoning as for forward
propagation, we see that the gradients’ variance can blow up unless
$n_\textrm{out} \sigma^2 = 1$, where $n_\textrm{out}$ is the
number of outputs of this layer. This leaves us in a dilemma: we cannot
possibly satisfy both conditions simultaneously. Instead, we simply try
to satisfy:

$$
(5.4.6)\[\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

This is the reasoning underlying the now-standard and practically
beneficial *Xavier initialization*, named after the first author of its
creators (Glorot and Bengio, 2010). Typically, the Xavier
initialization samples weights from a Gaussian distribution with zero
mean and variance
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$. We can also
adapt this to choose the variance when sampling weights from a uniform
distribution. Note that the uniform distribution $U(-a, a)$ has
variance $\frac{a^2}{3}$. Plugging $\frac{a^2}{3}$ into our
condition on $\sigma^2$ prompts us to initialize according to

$$
(5.4.7)\[U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).
$$

Though the assumption for nonexistence of nonlinearities in the above
mathematical reasoning can be easily violated in neural networks, the
Xavier initialization method turns out to work well in practice.

### 5.4.2.3. Beyond

The reasoning above barely scratches the surface of modern approaches to
parameter initialization. A deep learning framework often implements
over a dozen different heuristics. Moreover, parameter initialization
continues to be a hot area of fundamental research in deep learning.
Among these are heuristics specialized for tied (shared) parameters,
super-resolution, sequence models, and other situations. For instance,
Xiao *et al.* (2018) demonstrated the
possibility of training 10,000-layer neural networks without
architectural tricks by using a carefully-designed initialization
method.

If the topic interests you we suggest a deep dive into this module’s
offerings, reading the papers that proposed and analyzed each heuristic,
and then exploring the latest publications on the topic. Perhaps you
will stumble across or even invent a clever idea and contribute an
implementation to deep learning frameworks.

## 5.4.3. Summary

Vanishing and exploding gradients are common issues in deep networks.
Great care in parameter initialization is required to ensure that
gradients and parameters remain well controlled. Initialization
heuristics are needed to ensure that the initial gradients are neither
too large nor too small. Random initialization is key to ensuring that
symmetry is broken before optimization. Xavier initialization suggests
that, for each layer, variance of any output is not affected by the
number of inputs, and variance of any gradient is not affected by the
number of outputs. ReLU activation functions mitigate the vanishing
gradient problem. This can accelerate convergence.
