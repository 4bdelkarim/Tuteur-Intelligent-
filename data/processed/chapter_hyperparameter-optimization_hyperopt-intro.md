---
source_url: https://d2l.ai/chapter_hyperparameter-optimization/hyperopt-intro.html
title: 19.1. What Is Hyperparameter Optimization?
chapter: '19'
section_number: '19.1'
date: null
extractor: d2l
source_type: web
source: chapter_hyperparameter-optimization_hyperopt-intro
---

# 19.1. What Is Hyperparameter Optimization?

As we have seen in the previous chapters, deep neural networks come with
a large number of parameters or weights that are learned during
training. On top of these, every neural network has additional
*hyperparameters* that need to be configured by the user. For example,
to ensure that stochastic gradient descent converges to a local optimum
of the training loss (see Section 12), we have to
adjust the learning rate and batch size. To avoid overfitting on
training datasets, we might have to set regularization parameters, such
as weight decay (see Section 3.7) or dropout (see
Section 5.6). We can define the capacity and inductive bias
of the model by setting the number of layers and number of units or
filters per layer (i.e., the effective number of weights).

Unfortunately, we cannot simply adjust these hyperparameters by
minimizing the training loss, because this would lead to overfitting on
the training data. For example, setting regularization parameters, such
as dropout or weight decay to zero leads to a small training loss, but
might hurt the generalization performance.

Fig. 19.1.1 Typical workflow in machine learning that consists of training the
model multiple times with different hyperparameters.

Without a different form of automation, hyperparameters have to be set
manually in a trial-and-error fashion, in what amounts to a
time-consuming and difficult part of machine learning workflows. For
example, consider training a ResNet (see Section 8.6) on
CIFAR-10, which requires more than 2 hours on an Amazon Elastic Cloud
Compute (EC2) `g4dn.xlarge` instance. Even just trying ten
hyperparameter configurations in sequence, this would already take us
roughly one day. To make matters worse, hyperparameters are usually not
directly transferable across architectures and datasets
(Bardenet et al., 2013, Feurer et al., 2022, Wistuba et al., 2018), and need to be
re-optimized for every new task. Also, for most hyperparameters, there
are no rule-of-thumbs, and expert knowledge is required to find sensible
values.

*Hyperparameter optimization (HPO)* algorithms are designed to tackle
this problem in a principled and automated fashion
(Feurer and Hutter, 2018), by framing it as a global optimization
problem. The default objective is the error on a hold-out validation
dataset, but could in principle be any other business metric. It can be
combined with or constrained by secondary objectives, such as training
time, inference time, or model complexity.

Recently, hyperparameter optimization has been extended to *neural
architecture search (NAS)* (Elsken et al., 2018, Wistuba et al., 2019),
where the goal is to find entirely new neural network architectures.
Compared to classical HPO, NAS is even more expensive in terms of
computation and requires additional efforts to remain feasible in
practice. Both, HPO and NAS can be considered as sub-fields of AutoML
(Hutter et al., 2019), which aims to automate the entire ML pipeline.

In this section we will introduce HPO and show how we can automatically
find the best hyperparameters of the logistic regression example
introduced in Section 4.5.

## 19.1.1. The Optimization Problem

We will start with a simple toy problem: searching for the learning rate
of the multi-class logistic regression model `SoftmaxRegression` from
Section 4.5 to minimize the validation error on the
Fashion MNIST dataset. While other hyperparameters like batch size or
number of epochs are also worth tuning, we focus on learning rate alone
for simplicity.

```python
import numpy as np
import torch
from scipy import stats
from torch import nn
from d2l import torch as d2l
```

Before we can run HPO, we first need to define two ingredients: the
objective function and the configuration space.

### 19.1.1.1. The Objective Function

The performance of a learning algorithm can be seen as a function
$f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from the
hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the
validation loss. For every evaluation of $f(\mathbf{x})$, we have
to train and validate our machine learning model, which can be time and
compute intensive in the case of deep neural networks trained on large
datasets. Given our criterion $f(\mathbf{x})$ our goal is to find
$\mathbf{x}_{\star} \in \mathrm{argmin}_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$.

There is no simple way to compute gradients of $f$ with respect to
$\mathbf{x}$, because it would require to propagate the gradient
through the entire training process. While there is recent work
(Franceschi et al., 2017, Maclaurin et al., 2015) to drive HPO by
approximate “hypergradients”, none of the existing approaches are
competitive with the state-of-the-art yet, and we will not discuss them
here. Furthermore, the computational burden of evaluating $f$
requires HPO algorithms to approach the global optimum with as few
samples as possible.

The training of neural networks is stochastic (e.g., weights are
randomly initialized, mini-batches are randomly sampled), so that our
observations will be noisy: $y \sim f(\mathbf{x}) + \epsilon$,
where we usually assume that the $\epsilon \sim N(0, \sigma)$
observation noise is Gaussian distributed.

Faced with all these challenges, we usually try to identify a small set
of well performing hyperparameter configurations quickly, instead of
hitting the global optima exactly. However, due to large computational
demands of most neural networks models, even this can take days or weeks
of compute. We will explore in Section 19.4 how we can
speed-up the optimization process by either distributing the search or
using cheaper-to-evaluate approximations of the objective function.

We begin with a method for computing the validation error of a model.

```python
class HPOTrainer(d2l.Trainer): #@save
 def validation_error(self):
 self.model.eval()
 accuracy = 0
 val_batch_idx = 0
 for batch in self.val_dataloader:
 with torch.no_grad():
 x, y = self.prepare_batch(batch)
 y_hat = self.model(x)
 accuracy += self.model.accuracy(y_hat, y)
 val_batch_idx += 1
 return 1 - accuracy / val_batch_idx
```

We optimize validation error with respect to the hyperparameter
configuration `config`, consisting of the `learning_rate`. For each
evaluation, we train our model for `max_epochs` epochs, then compute
and return its validation error:

```python
def hpo_objective_softmax_classification(config, max_epochs=8):
 learning_rate = config["learning_rate"]
 trainer = d2l.HPOTrainer(max_epochs=max_epochs)
 data = d2l.FashionMNIST(batch_size=16)
 model = d2l.SoftmaxRegression(num_outputs=10, lr=learning_rate)
 trainer.fit(model=model, data=data)
 return trainer.validation_error().detach().numpy()
```

### 19.1.1.2. The Configuration Space

Along with the objective function $f(\mathbf{x})$, we also need to
define the feasible set $\mathbf{x} \in \mathcal{X}$ to optimize
over, known as *configuration space* or *search space*. For our logistic
regression example, we will use:

```python
config_space = {"learning_rate": stats.loguniform(1e-4, 1)}
```

Here we use the use the `loguniform` object from SciPy, which
represents a uniform distribution between -4 and -1 in the logarithmic
space. This object allows us to sample random variables from this
distribution.

Each hyperparameter has a data type, such as `float` for
`learning_rate`, as well as a closed bounded range (i.e., lower and
upper bounds). We usually assign a prior distribution (e.g, uniform or
log-uniform) to each hyperparameter to sample from. Some positive
parameters, such as `learning_rate`, are best represented on a
logarithmic scale as optimal values can differ by several orders of
magnitude, while others, such as momentum, come with linear scale.

Below we show a simple example of a configuration space consisting of
typical hyperparameters of a multi-layer perceptron including their type
and standard ranges.

: Example configuration space of multi-layer perceptron

Table 19.1.1 label:`tab_example_configspace`

| Name | Type | Hyperparameter Ranges | log-scale |
| --- | --- | --- | --- |
| learning rate | float | :math:` [10^{-6},10^{-1}]` | yes |
| batch size | integer | $[8,256]$ | yes |
| momentum | float | $[0,0.99]$ | no |
| activation function | categorical | :mat h:{textrm{tanh} , textrm{relu}} | |
| number of units | integer | $[32, 1024]$ | yes |
| number of layers | integer | $[1, 6]$ | no |

In general, the structure of the configuration space $\mathcal{X}$
can be complex and it can be quite different from $\mathbb{R}^d$.
In practice, some hyperparameters may depend on the value of others. For
example, assume we try to tune the number of layers for a multi-layer
perceptron, and for each layer the number of units. The number of units
of the $l\textrm{-th}$ layer is relevant only if the network has
at least $l+1$ layers. These advanced HPO problems are beyond the
scope of this chapter. We refer the interested reader to
(Baptista and Poloczek, 2018, Hutter et al., 2011, Jenatton et al., 2017).

The configuration space plays an important role for hyperparameter
optimization, since no algorithms can find something that is not
included in the configuration space. On the other hand, if the ranges
are too large, the computation budget to find well performing
configurations might become infeasible.

## 19.1.2. Random Search

*Random search* is the first hyperparameter optimization algorithm we
will consider. The main idea of random search is to independently sample
from the configuration space until a predefined budget (e.g maximum
number of iterations) is exhausted, and to return the best observed
configuration. All evaluations can be executed independently in parallel
(see Section 19.3), but here we use a sequential loop for
simplicity.

```python
errors, values = [], []
num_iterations = 5

for i in range(num_iterations):
 learning_rate = config_space["learning_rate"].rvs()
 print(f"Trial {i}: learning_rate = {learning_rate}")
 y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
 print(f" validation_error = {y}")
 values.append(learning_rate)
 errors.append(y)
```

The best learning rate is then simply the one with the lowest validation
error.

```python
best_idx = np.argmin(errors)
print(f"optimal learning rate = {values[best_idx]}")
```

Due to its simplicity and generality, random search is one of the most
frequently used HPO algorithms. It does not require any sophisticated
implementation and can be applied to any configuration space as long as
we can define some probability distribution for each hyperparameter.

Unfortunately random search also comes with a few shortcomings. First,
it does not adapt the sampling distribution based on the previous
observations it collected so far. Hence, it is equally likely to sample
a poorly performing configuration than a better performing
configuration. Second, the same amount of resources are spent for all
configurations, even though some may show poor initial performance and
are less likely to outperform previously seen configurations.

In the next sections we will look at more sample efficient
hyperparameter optimization algorithms that overcome the shortcomings of
random search by using a model to guide the search. We will also look at
algorithms that automatically stop the evaluation process of poorly
performing configurations to speed up the optimization process.

## 19.1.3. Summary

In this section we introduced hyperparameter optimization (HPO) and how
we can phrase it as a global optimization by defining a configuration
space and an objective function. We also implemented our first HPO
algorithm, random search, and applied it on a simple softmax
classification problem.

While random search is very simple, it is the better alternative to grid
search, which simply evaluates a fixed set of hyperparameters. Random
search somewhat mitigates the curse of dimensionality
(Bellman, 1966), and can be far more efficient than grid
search if the criterion most strongly depends on a small subset of the
hyperparameters.
