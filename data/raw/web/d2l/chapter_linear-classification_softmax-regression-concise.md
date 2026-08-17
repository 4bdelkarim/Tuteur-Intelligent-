---
source_url: "https://d2l.ai/chapter_linear-classification/softmax-regression-concise.html"
title: "4.5. Concise Implementation of Softmax Regression"
chapter: "4"
section_number: "4.5"
date: null
extractor: "d2l"
---

# 4.5. Concise Implementation of Softmax Regression

Just as high-level deep learning frameworks made it easier to implement
linear regression (see Section 3.5), they are
similarly convenient here.

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

## 4.5.1. Defining the Model

As in Section 3.5, we construct our fully connected
layer using the built-in layer. The built-in `__call__` method then
invokes `forward` whenever we need to apply the network to some input.

We use a `Flatten` layer to convert the fourth-order tensor `X` to
second order by keeping the dimensionality along the first axis
unchanged.

```python
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

## 4.5.2. Softmax Revisited

In Section 4.4 we calculated our model’s output and
applied the cross-entropy loss. While this is perfectly reasonable
mathematically, it is risky computationally, because of numerical
underflow and overflow in the exponentiation.

Recall that the softmax function computes probabilities via
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$. If some of the
$o_k$ are very large, i.e., very positive, then $\exp(o_k)$
might be larger than the largest number we can have for certain data
types. This is called *overflow*. Likewise, if every argument is a very
large negative number, we will get *underflow*. For instance, single
precision floating point numbers approximately cover the range of
$10^{-38}$ to $10^{38}$. As such, if the largest term in
$\mathbf{o}$ lies outside the interval $[-90, 90]$, the
result will not be stable. A way round this problem is to subtract
$\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ from all entries:

$$
(4.5.1)\[\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

By construction we know that $o_j - \bar{o} \leq 0$ for all
$j$. As such, for a $q$-class classification problem, the
denominator is contained in the interval $[1, q]$. Moreover, the
numerator never exceeds $1$, thus preventing numerical overflow.
Numerical underflow only occurs when $\exp(o_j - \bar{o})$
numerically evaluates as $0$. Nonetheless, a few steps down the
road we might find ourselves in trouble when we want to compute
$\log \hat{y}_j$ as $\log 0$. In particular, in
backpropagation, we might find ourselves faced with a screenful of the
dreaded `NaN` (Not a Number) results.

Fortunately, we are saved by the fact that even though we are computing
exponential functions, we ultimately intend to take their log (when
calculating the cross-entropy loss). By combining softmax and
cross-entropy, we can escape the numerical stability issues altogether.
We have:

$$
(4.5.2)\[\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

This avoids both overflow and underflow. We will want to keep the
conventional softmax function handy in case we ever want to evaluate the
output probabilities by our model. But instead of passing softmax
probabilities into our new loss function, we just pass the logits and
compute the softmax and its log all at once inside the cross-entropy
loss function, which does smart things like the [“LogSumExp
trick”](https://en.wikipedia.org/wiki/LogSumExp).

```python
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')
```

## 4.5.3. Training

Next we train our model. We use Fashion-MNIST images, flattened to
784-dimensional feature vectors.

```python
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

![../_images/output_softmax-regression-concise_0b22ca_52_0.svg](../_images/output_softmax-regression-concise_0b22ca_52_0.svg)

As before, this algorithm converges to a solution that is reasonably
accurate, albeit this time with fewer lines of code than before.

## 4.5.4. Summary

High-level APIs are very convenient at hiding from their user
potentially dangerous aspects, such as numerical stability. Moreover,
they allow users to design models concisely with very few lines of code.
This is both a blessing and a curse. The obvious benefit is that it
makes things highly accessible, even to engineers who never took a
single class of statistics in their life (in fact, they are part of the
target audience of the book). But hiding the sharp edges also comes with
a price: a disincentive to add new and different components on your own,
since there is little muscle memory for doing it. Moreover, it makes it
more difficult to *fix* things whenever the protective padding of a
framework fails to cover all the corner cases entirely. Again, this is
due to lack of familiarity.

As such, we strongly urge you to review *both* the bare bones and the
elegant versions of many of the implementations that follow. While we
emphasize ease of understanding, the implementations are nonetheless
usually quite performant (convolutions are the big exception here). It
is our intention to allow you to build on these when you invent
something new that no framework can give you.

## 4.5.5. Exercises

1. Deep learning uses many different number formats, including FP64
   double precision (used extremely rarely), FP32 single precision,
   BFLOAT16 (good for compressed representations), FP16 (very unstable),
   TF32 (a new format from NVIDIA), and INT8. Compute the smallest and
   largest argument of the exponential function for which the result
   does not lead to numerical underflow or overflow.
2. INT8 is a very limited format consisting of nonzero numbers from
   $1$ to $255$. How could you extend its dynamic range
   without using more bits? Do standard multiplication and addition
   still work?
3. Increase the number of epochs for training. Why might the validation
   accuracy decrease after a while? How could we fix this?
4. What happens as you increase the learning rate? Compare the loss
   curves for several learning rates. Which one works better? When?
