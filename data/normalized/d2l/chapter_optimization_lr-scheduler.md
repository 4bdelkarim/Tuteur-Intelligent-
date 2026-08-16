---
source_url: https://d2l.ai/chapter_optimization/lr-scheduler.html
title: 12.11. Learning Rate Scheduling
chapter: '12'
section_number: '12.11'
date: null
extractor: d2l
source_type: web
---

# 12.11. Learning Rate Scheduling

So far we primarily focused on optimization *algorithms* for how to
update the weight vectors rather than on the *rate* at which they are
being updated. Nonetheless, adjusting the learning rate is often just as
important as the actual algorithm. There are a number of aspects to
consider:

- Most obviously the *magnitude* of the learning rate matters. If it is
 too large, optimization diverges, if it is too small, it takes too
 long to train or we end up with a suboptimal result. We saw
 previously that the condition number of the problem matters (see
 e.g., Section 12.6 for details). Intuitively it is the
 ratio of the amount of change in the least sensitive direction
 vs. the most sensitive one.
- Secondly, the rate of decay is just as important. If the learning
 rate remains large we may simply end up bouncing around the minimum
 and thus not reach optimality. Section 12.5
 discussed this in some detail and we analyzed performance guarantees
 in Section 12.4. In short, we want the rate to decay, but
 probably more slowly than $\mathcal{O}(t^{-\frac{1}{2}})$ which
 would be a good choice for convex problems.
- Another aspect that is equally important is *initialization*. This
 pertains both to how the parameters are set initially (review
 Section 5.4 for details) and also how they
 evolve initially. This goes under the moniker of *warmup*, i.e., how
 rapidly we start moving towards the solution initially. Large steps
 in the beginning might not be beneficial, in particular since the
 initial set of parameters is random. The initial update directions
 might be quite meaningless, too.
- Lastly, there are a number of optimization variants that perform
 cyclical learning rate adjustment. This is beyond the scope of the
 current chapter. We recommend the reader to review details in
 Izmailov *et al.* (2018), e.g., how to obtain
 better solutions by averaging over an entire *path* of parameters.

Given the fact that there is a lot of detail needed to manage learning
rates, most deep learning frameworks have tools to deal with this
automatically. In the current chapter we will review the effects that
different schedules have on accuracy and also show how this can be
managed efficiently via a *learning rate scheduler*.

## 12.11.1. Toy Problem

We begin with a toy problem that is cheap enough to compute easily, yet
sufficiently nontrivial to illustrate some of the key aspects. For that
we pick a slightly modernized version of LeNet (`relu` instead of
`sigmoid` activation, MaxPooling rather than AveragePooling), as
applied to Fashion-MNIST. Moreover, we hybridize the network for
performance. Since most of the code is standard we just introduce the
basics without further detailed discussion. See Section 7 for
a refresher as needed.

```python
%matplotlib inline
import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from d2l import torch as d2l

def net_fn():
 model = nn.Sequential(
 nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
 nn.MaxPool2d(kernel_size=2, stride=2),
 nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
 nn.MaxPool2d(kernel_size=2, stride=2),
 nn.Flatten(),
 nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
 nn.Linear(120, 84), nn.ReLU(),
 nn.Linear(84, 10))

 return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
 scheduler=None):
 net.to(device)
 animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
 legend=['train loss', 'train acc', 'test acc'])

 for epoch in range(num_epochs):
 metric = d2l.Accumulator(3) # train_loss, train_acc, num_examples
 for i, (X, y) in enumerate(train_iter):
 net.train()
 trainer.zero_grad()
 X, y = X.to(device), y.to(device)
 y_hat = net(X)
 l = loss(y_hat, y)
 l.backward()
 trainer.step()
 with torch.no_grad():
 metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
 train_loss = metric[0] / metric[2]
 train_acc = metric[1] / metric[2]
 if (i + 1) % 50 == 0:
 animator.add(epoch + i / len(train_iter),
 (train_loss, train_acc, None))

 test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
 animator.add(epoch+1, (None, None, test_acc))

 if scheduler:
 if scheduler.__module__ == lr_scheduler.__name__:
 # Using PyTorch In-Built scheduler
 scheduler.step()
 else:
 # Using custom defined scheduler
 for param_group in trainer.param_groups:
 param_group['lr'] = scheduler(epoch)

 print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
 f'test acc {test_acc:.3f}')
```

Let’s have a look at what happens if we invoke this algorithm with
default settings, such as a learning rate of $0.3$ and train for
$30$ iterations. Note how the training accuracy keeps on
increasing while progress in terms of test accuracy stalls beyond a
point. The gap between both curves indicates overfitting.

```python
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

## 12.11.2. Schedulers

One way of adjusting the learning rate is to set it explicitly at each
step. This is conveniently achieved by the `set_learning_rate` method.
We could adjust it downward after every epoch (or even after every
minibatch), e.g., in a dynamic manner in response to how optimization is
progressing.

```python
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

More generally we want to define a scheduler. When invoked with the
number of updates it returns the appropriate value of the learning rate.
Let’s define a simple one that sets the learning rate to
$\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```python
class SquareRootScheduler:
 def __init__(self, lr=0.1):
 self.lr = lr

 def __call__(self, num_update):
 return self.lr * pow(num_update + 1.0, -0.5)
```

Let’s plot its behavior over a range of values.

```python
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Now let’s see how this plays out for training on Fashion-MNIST. We
simply provide the scheduler as an additional argument to the training
algorithm.

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
 scheduler)
```

This worked quite a bit better than previously. Two things stand out:
the curve was rather more smooth than previously. Secondly, there was
less overfitting. Unfortunately it is not a well-resolved question as to
why certain strategies lead to less overfitting in *theory*. There is
some argument that a smaller stepsize will lead to parameters that are
closer to zero and thus simpler. However, this does not explain the
phenomenon entirely since we do not really stop early but simply reduce
the learning rate gently.

## 12.11.3. Policies

While we cannot possibly cover the entire variety of learning rate
schedulers, we attempt to give a brief overview of popular policies
below. Common choices are polynomial decay and piecewise constant
schedules. Beyond that, cosine learning rate schedules have been found
to work well empirically on some problems. Lastly, on some problems it
is beneficial to warm up the optimizer prior to using large learning
rates.

### 12.11.3.1. Factor Scheduler

One alternative to a polynomial decay would be a multiplicative one,
that is $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ for
$\alpha \in (0, 1)$. To prevent the learning rate from decaying
beyond a reasonable lower bound the update equation is often modified to
$\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```python
class FactorScheduler:
 def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
 self.factor = factor
 self.stop_factor_lr = stop_factor_lr
 self.base_lr = base_lr

 def __call__(self, num_update):
 self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
 return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(torch.arange(50), [scheduler(t) for t in range(50)])
```

This can also be accomplished by a built-in scheduler in MXNet via the
`lr_scheduler.FactorScheduler` object. It takes a few more parameters,
such as warmup period, warmup mode (linear or constant), the maximum
number of desired updates, etc.; Going forward we will use the built-in
schedulers as appropriate and only explain their functionality here. As
illustrated, it is fairly straightforward to build your own scheduler if
needed.

### 12.11.3.2. Multi Factor Scheduler

A common strategy for training deep networks is to keep the learning
rate piecewise constant and to decrease it by a given amount every so
often. That is, given a set of times when to decrease the rate, such as
$s = \{5, 10, 20\}$ decrease
$\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ whenever
$t \in s$. Assuming that the values are halved at each step we can
implement this as follows.

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
 lr = scheduler.get_last_lr()[0]
 trainer.step()
 scheduler.step()
 return lr

d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler)
 for t in range(num_epochs)])
```

The intuition behind this piecewise constant learning rate schedule is
that one lets optimization proceed until a stationary point has been
reached in terms of the distribution of weight vectors. Then (and only
then) do we decrease the rate such as to obtain a higher quality proxy
to a good local minimum. The example below shows how this can produce
ever slightly better solutions.

```python
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
 scheduler)
```

### 12.11.3.3. Cosine Scheduler

A rather perplexing heuristic was proposed by
Loshchilov and Hutter (2016). It relies on the observation that we
might not want to decrease the learning rate too drastically in the
beginning and moreover, that we might want to “refine” the solution in
the end using a very small learning rate. This results in a cosine-like
schedule with the following functional form for learning rates in the
range $t \in [0, T]$.

$$
(12.11.1)\[\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)
$$

Here $\eta_0$ is the initial learning rate, $\eta_T$ is the
target rate at time $T$. Furthermore, for $t > T$ we simply
pin the value to $\eta_T$ without increasing it again. In the
following example, we set the max update step $T = 20$.

```python
class CosineScheduler:
 def __init__(self, max_update, base_lr=0.01, final_lr=0,
 warmup_steps=0, warmup_begin_lr=0):
 self.base_lr_orig = base_lr
 self.max_update = max_update
 self.final_lr = final_lr
 self.warmup_steps = warmup_steps
 self.warmup_begin_lr = warmup_begin_lr
 self.max_steps = self.max_update - self.warmup_steps

 def get_warmup_lr(self, epoch):
 increase = (self.base_lr_orig - self.warmup_begin_lr) \
 * float(epoch) / float(self.warmup_steps)
 return self.warmup_begin_lr + increase

 def __call__(self, epoch):
 if epoch < self.warmup_steps:
 return self.get_warmup_lr(epoch)
 if epoch <= self.max_update:
 self.base_lr = self.final_lr + (
 self.base_lr_orig - self.final_lr) * (1 + math.cos(
 math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
 return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

In the context of computer vision this schedule *can* lead to improved
results. Note, though, that such improvements are not guaranteed (as can
be seen below).

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
 scheduler)
```

### 12.11.3.4. Warmup

In some cases initializing the parameters is not sufficient to guarantee
a good solution. This is particularly a problem for some advanced
network designs that may lead to unstable optimization problems. We
could address this by choosing a sufficiently small learning rate to
prevent divergence in the beginning. Unfortunately this means that
progress is slow. Conversely, a large learning rate initially leads to
divergence.

A rather simple fix for this dilemma is to use a warmup period during
which the learning rate *increases* to its initial maximum and to cool
down the rate until the end of the optimization process. For simplicity
one typically uses a linear increase for this purpose. This leads to a
schedule of the form indicated below.

```python
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Note that the network converges better initially (in particular observe
the performance during the first 5 epochs).

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
 scheduler)
```

Warmup can be applied to any scheduler (not just cosine). For a more
detailed discussion of learning rate schedules and many more experiments
see also (Gotmare et al., 2018). In particular they find
that a warmup phase limits the amount of divergence of parameters in
very deep networks. This makes intuitively sense since we would expect
significant divergence due to random initialization in those parts of
the network that take the most time to make progress in the beginning.

## 12.11.4. Summary

- Decreasing the learning rate during training can lead to improved
 accuracy and (most perplexingly) reduced overfitting of the model.
- A piecewise decrease of the learning rate whenever progress has
 plateaued is effective in practice. Essentially this ensures that we
 converge efficiently to a suitable solution and only then reduce the
 inherent variance of the parameters by reducing the learning rate.
- Cosine schedulers are popular for some computer vision problems. See
 e.g., GluonCV for details of such a
 scheduler.
- A warmup period before optimization can prevent divergence.
- Optimization serves multiple purposes in deep learning. Besides
 minimizing the training objective, different choices of optimization
 algorithms and learning rate scheduling can lead to rather different
 amounts of generalization and overfitting on the test set (for the
 same amount of training error).
