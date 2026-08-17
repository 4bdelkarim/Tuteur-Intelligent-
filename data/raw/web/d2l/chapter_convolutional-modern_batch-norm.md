---
source_url: "https://d2l.ai/chapter_convolutional-modern/batch-norm.html"
title: "8.5. Batch Normalization"
chapter: "8"
section_number: "8.5"
date: null
extractor: "d2l"
---

# 8.5. Batch Normalization

Training deep neural networks is difficult. Getting them to converge in
a reasonable amount of time can be tricky. In this section, we describe
*batch normalization*, a popular and effective technique that
consistently accelerates the convergence of deep networks
(Ioffe and Szegedy, 2015). Together with residual blocks—covered
later in Section 8.6—batch normalization has made it possible
for practitioners to routinely train networks with over 100 layers. A
secondary (serendipitous) benefit of batch normalization lies in its
inherent regularization.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 8.5.1. Training Deep Networks

When working with data, we often preprocess before training. Choices
regarding data preprocessing often make an enormous difference in the
final results. Recall our application of MLPs to predicting house prices
(Section 5.7). Our first step when working with real
data was to standardize our input features to have zero mean
$\boldsymbol{\mu} = 0$ and unit variance
$\boldsymbol{\Sigma} = \boldsymbol{1}$ across multiple
observations (Friedman, 1987), frequently rescaling the
latter so that the diagonal is unity, i.e., $\Sigma_{ii} = 1$. Yet
another strategy is to rescale vectors to unit length, possibly zero
mean *per observation*. This can work well, e.g., for spatial sensor
data. These preprocessing techniques and many others, are beneficial for
keeping the estimation problem well controlled. For a review of feature
selection and extraction see the article of Guyon *et al.* (2008),
for example. Standardizing vectors also has the nice side-effect of
constraining the function complexity of functions that act upon it. For
instance, the celebrated radius-margin bound (Vapnik, 1995) in
support vector machines and the Perceptron Convergence Theorem
(Novikoff, 1962) rely on inputs of bounded norm.

Intuitively, this standardization plays nicely with our optimizers since
it puts the parameters *a priori* on a similar scale. As such, it is
only natural to ask whether a corresponding normalization step *inside*
a deep network might not be beneficial. While this is not quite the
reasoning that led to the invention of batch normalization
(Ioffe and Szegedy, 2015), it is a useful way of understanding it and
its cousin, layer normalization (Ba et al., 2016), within a
unified framework.

Second, for a typical MLP or CNN, as we train, the variables in
intermediate layers (e.g., affine transformation outputs in MLP) may
take values with widely varying magnitudes: whether along the layers
from input to output, across units in the same layer, and over time due
to our updates to the model parameters. The inventors of batch
normalization postulated informally that this drift in the distribution
of such variables could hamper the convergence of the network.
Intuitively, we might conjecture that if one layer has variable
activations that are 100 times that of another layer, this might
necessitate compensatory adjustments in the learning rates. Adaptive
solvers such as AdaGrad (Duchi et al., 2011), Adam
(Kingma and Ba, 2014), Yogi (Zaheer et al., 2018), or
Distributed Shampoo (Anil et al., 2020) aim to address this from
the viewpoint of optimization, e.g., by adding aspects of second-order
methods. The alternative is to prevent the problem from occurring,
simply by adaptive normalization.

Third, deeper networks are complex and tend to be more liable to
overfitting. This means that regularization becomes more critical. A
common technique for regularization is noise injection. This has been
known for a long time, e.g., with regard to noise injection for the
inputs (Bishop, 1995). It also forms the basis of dropout in
Section 5.6. As it turns out, quite serendipitously, batch
normalization conveys all three benefits: preprocessing, numerical
stability, and regularization.

Batch normalization is applied to individual layers, or optionally, to
all of them: In each training iteration, we first normalize the inputs
(of batch normalization) by subtracting their mean and dividing by their
standard deviation, where both are estimated based on the statistics of
the current minibatch. Next, we apply a scale coefficient and an offset
to recover the lost degrees of freedom. It is precisely due to this
*normalization* based on *batch* statistics that *batch normalization*
derives its name.

Note that if we tried to apply batch normalization with minibatches of
size 1, we would not be able to learn anything. That is because after
subtracting the means, each hidden unit would take value 0. As you might
guess, since we are devoting a whole section to batch normalization,
with large enough minibatches the approach proves effective and stable.
One takeaway here is that when applying batch normalization, the choice
of batch size is even more significant than without batch normalization,
or at least, suitable calibration is needed as we might adjust batch
size.

Denote by $\mathcal{B}$ a minibatch and let
$\mathbf{x} \in \mathcal{B}$ be an input to batch normalization
($\textrm{BN}$). In this case the batch normalization is defined
as follows:

$$
(8.5.1)\[\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.
$$

In (8.5.1), $\hat{\boldsymbol{\mu}}_\mathcal{B}$
is the sample mean and $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ is
the sample standard deviation of the minibatch $\mathcal{B}$.
After applying standardization, the resulting minibatch has zero mean
and unit variance. The choice of unit variance (rather than some other
magic number) is arbitrary. We recover this degree of freedom by
including an elementwise *scale parameter* $\boldsymbol{\gamma}$
and *shift parameter* $\boldsymbol{\beta}$ that have the same
shape as $\mathbf{x}$. Both are parameters that need to be learned
as part of model training.

The variable magnitudes for intermediate layers cannot diverge during
training since batch normalization actively centers and rescales them
back to a given mean and size (via
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ and
${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$). Practical experience
confirms that, as alluded to when discussing feature rescaling, batch
normalization seems to allow for more aggressive learning rates. We
calculate $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and
${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ in
(8.5.1) as follows:

$$
(8.5.2)\[\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.
$$

Note that we add a small constant $\epsilon > 0$ to the variance
estimate to ensure that we never attempt division by zero, even in cases
where the empirical variance estimate might be very small or vanish. The
estimates $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and
${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ counteract the scaling
issue by using noisy estimates of mean and variance. You might think
that this noisiness should be a problem. On the contrary, it is actually
beneficial.

This turns out to be a recurring theme in deep learning. For reasons
that are not yet well-characterized theoretically, various sources of
noise in optimization often lead to faster training and less
overfitting: this variation appears to act as a form of regularization.
Teye *et al.* (2018) and Luo *et al.* (2018)
related the properties of batch normalization to Bayesian priors and
penalties, respectively. In particular, this sheds some light on the
puzzle of why batch normalization works best for moderate minibatch
sizes in the 50–100 range. This particular size of minibatch seems to
inject just the “right amount” of noise per layer, both in terms of
scale via $\hat{\boldsymbol{\sigma}}$, and in terms of offset via
$\hat{\boldsymbol{\mu}}$: a larger minibatch regularizes less due
to the more stable estimates, whereas tiny minibatches destroy useful
signal due to high variance. Exploring this direction further,
considering alternative types of preprocessing and filtering may yet
lead to other effective types of regularization.

Fixing a trained model, you might think that we would prefer using the
entire dataset to estimate the mean and variance. Once training is
complete, why would we want the same image to be classified differently,
depending on the batch in which it happens to reside? During training,
such exact calculation is infeasible because the intermediate variables
for all data examples change every time we update our model. However,
once the model is trained, we can calculate the means and variances of
each layer’s variables based on the entire dataset. Indeed this is
standard practice for models employing batch normalization; thus batch
normalization layers function differently in *training mode*
(normalizing by minibatch statistics) than in *prediction mode*
(normalizing by dataset statistics). In this form they closely resemble
the behavior of dropout regularization of Section 5.6, where
noise is only injected during training.

## 8.5.2. Batch Normalization Layers

Batch normalization implementations for fully connected layers and
convolutional layers are slightly different. One key difference between
batch normalization and other layers is that because the former operates
on a full minibatch at a time, we cannot just ignore the batch dimension
as we did before when introducing other layers.

### 8.5.2.1. Fully Connected Layers

When applying batch normalization to fully connected layers,
Ioffe and Szegedy (2015), in their original paper inserted batch
normalization after the affine transformation and *before* the nonlinear
activation function. Later applications experimented with inserting
batch normalization right *after* activation functions. Denoting the
input to the fully connected layer by $\mathbf{x}$, the affine
transformation by $\mathbf{W}\mathbf{x} + \mathbf{b}$ (with the
weight parameter $\mathbf{W}$ and the bias parameter
$\mathbf{b}$), and the activation function by $\phi$, we can
express the computation of a batch-normalization-enabled, fully
connected layer output $\mathbf{h}$ as follows:

$$
(8.5.3)\[\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).
$$

Recall that mean and variance are computed on the *same* minibatch on
which the transformation is applied.

### 8.5.2.2. Convolutional Layers

Similarly, with convolutional layers, we can apply batch normalization
after the convolution but before the nonlinear activation function. The
key difference from batch normalization in fully connected layers is
that we apply the operation on a per-channel basis *across all
locations*. This is compatible with our assumption of translation
invariance that led to convolutions: we assumed that the specific
location of a pattern within an image was not critical for the purpose
of understanding.

Assume that our minibatches contain $m$ examples and that for each
channel, the output of the convolution has height $p$ and width
$q$. For convolutional layers, we carry out each batch
normalization over the $m \cdot p \cdot q$ elements per output
channel simultaneously. Thus, we collect the values over all spatial
locations when computing the mean and variance and consequently apply
the same mean and variance within a given channel to normalize the value
at each spatial location. Each channel has its own scale and shift
parameters, both of which are scalars.

### 8.5.2.3. Layer Normalization

Note that in the context of convolutions the batch normalization is well
defined even for minibatches of size 1: after all, we have all the
locations across an image to average. Consequently, mean and variance
are well defined, even if it is just within a single observation. This
consideration led Ba *et al.* (2016) to introduce the
notion of *layer normalization*. It works just like a batch norm, only
that it is applied to one observation at a time. Consequently both the
offset and the scaling factor are scalars. For an $n$-dimensional
vector $\mathbf{x}$, layer norms are given by

$$
(8.5.4)\[\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},
$$

where scaling and offset are applied coefficient-wise and given by

$$
(8.5.5)\[\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.
$$

As before we add a small offset $\epsilon > 0$ to prevent division
by zero. One of the major benefits of using layer normalization is that
it prevents divergence. After all, ignoring $\epsilon$, the output
of the layer normalization is scale independent. That is, we have
$\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$
for any choice of $\alpha \neq 0$. This becomes an equality for
$|\alpha| \to \infty$ (the approximate equality is due to the
offset $\epsilon$ for the variance).

Another advantage of the layer normalization is that it does not depend
on the minibatch size. It is also independent of whether we are in
training or test regime. In other words, it is simply a deterministic
transformation that standardizes the activations to a given scale. This
can be very beneficial in preventing divergence in optimization. We skip
further details and recommend that interested readers consult the
original paper.

### 8.5.2.4. Batch Normalization During Prediction

As we mentioned earlier, batch normalization typically behaves
differently in training mode than in prediction mode. First, the noise
in the sample mean and the sample variance arising from estimating each
on minibatches is no longer desirable once we have trained the model.
Second, we might not have the luxury of computing per-batch
normalization statistics. For example, we might need to apply our model
to make one prediction at a time.

Typically, after training, we use the entire dataset to compute stable
estimates of the variable statistics and then fix them at prediction
time. Hence, batch normalization behaves differently during training
than at test time. Recall that dropout also exhibits this
characteristic.

## 8.5.3. Implementation from Scratch

To see how batch normalization works in practice, we implement one from
scratch below.

```python
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

We can now create a proper `BatchNorm` layer. Our layer will maintain
proper parameters for scale `gamma` and shift `beta`, both of which
will be updated in the course of training. Additionally, our layer will
maintain moving averages of the means and variances for subsequent use
during model prediction.

Putting aside the algorithmic details, note the design pattern
underlying our implementation of the layer. Typically, we define the
mathematics in a separate function, say `batch_norm`. We then
integrate this functionality into a custom layer, whose code mostly
addresses bookkeeping matters, such as moving data to the right device
context, allocating and initializing any required variables, keeping
track of moving averages (here for mean and variance), and so on. This
pattern enables a clean separation of mathematics from boilerplate code.
Also note that for the sake of convenience we did not worry about
automatically inferring the input shape here; thus we need to specify
the number of features throughout. By now all modern deep learning
frameworks offer automatic detection of size and shape in the high-level
batch normalization APIs (in practice we will use this instead).

```python
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

We used `momentum` to govern the aggregation over past mean and
variance estimates. This is somewhat of a misnomer as it has nothing
whatsoever to do with the *momentum* term of optimization. Nonetheless,
it is the commonly adopted name for this term and in deference to API
naming convention we use the same variable name in our code.

## 8.5.4. LeNet with Batch Normalization

To see how to apply `BatchNorm` in context, below we apply it to a
traditional LeNet model (Section 7.6). Recall that batch
normalization is applied after the convolutional layers or fully
connected layers but before the corresponding activation functions.

```python
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
            BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
```

As before, we will train our network on the Fashion-MNIST dataset. This
code is virtually identical to that when we first trained LeNet.

```python
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

![../_images/output_batch-norm_cf033c_65_0.svg](../_images/output_batch-norm_cf033c_65_0.svg)

Let’s have a look at the scale parameter `gamma` and the shift
parameter `beta` learned from the first batch normalization layer.

```python
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

## 8.5.5. Concise Implementation

Compared with the `BatchNorm` class, which we just defined ourselves,
we can use the `BatchNorm` class defined in high-level APIs from the
deep learning framework directly. The code looks virtually identical to
our implementation above, except that we no longer need to provide
additional arguments for it to get the dimensions right.

```python
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(num_classes))
```

Below, we use the same hyperparameters to train our model. Note that as
usual, the high-level API variant runs much faster because its code has
been compiled to C++ or CUDA while our custom implementation must be
interpreted by Python.

```python
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

![../_images/output_batch-norm_cf033c_110_0.svg](../_images/output_batch-norm_cf033c_110_0.svg)

## 8.5.6. Discussion

Intuitively, batch normalization is thought to make the optimization
landscape smoother. However, we must be careful to distinguish between
speculative intuitions and true explanations for the phenomena that we
observe when training deep models. Recall that we do not even know why
simpler deep neural networks (MLPs and conventional CNNs) generalize
well in the first place. Even with dropout and weight decay, they remain
so flexible that their ability to generalize to unseen data likely needs
significantly more refined learning-theoretic generalization guarantees.

The original paper proposing batch normalization
(Ioffe and Szegedy, 2015), in addition to introducing a powerful and
useful tool, offered an explanation for why it works: by reducing
*internal covariate shift*. Presumably by *internal covariate shift*
they meant something like the intuition expressed above—the notion that
the distribution of variable values changes over the course of training.
However, there were two problems with this explanation: i) This drift is
very different from *covariate shift*, rendering the name a misnomer. If
anything, it is closer to concept drift. ii) The explanation offers an
under-specified intuition but leaves the question of *why precisely this
technique works* an open question wanting for a rigorous explanation.
Throughout this book, we aim to convey the intuitions that practitioners
use to guide their development of deep neural networks. However, we
believe that it is important to separate these guiding intuitions from
established scientific fact. Eventually, when you master this material
and start writing your own research papers you will want to be clear to
delineate between technical claims and hunches.

Following the success of batch normalization, its explanation in terms
of *internal covariate shift* has repeatedly surfaced in debates in the
technical literature and broader discourse about how to present machine
learning research. In a memorable speech given while accepting a Test of
Time Award at the 2017 NeurIPS conference, Ali Rahimi used *internal
covariate shift* as a focal point in an argument likening the modern
practice of deep learning to alchemy. Subsequently, the example was
revisited in detail in a position paper outlining troubling trends in
machine learning (Lipton and Steinhardt, 2018). Other authors have
proposed alternative explanations for the success of batch
normalization, some (Santurkar et al., 2018) claiming
that batch normalization’s success comes despite exhibiting behavior
that is in some ways opposite to those claimed in the original paper.

We note that the *internal covariate shift* is no more worthy of
criticism than any of thousands of similarly vague claims made every
year in the technical machine learning literature. Likely, its resonance
as a focal point of these debates owes to its broad recognizability for
the target audience. Batch normalization has proven an indispensable
method, applied in nearly all deployed image classifiers, earning the
paper that introduced the technique tens of thousands of citations. We
conjecture, though, that the guiding principles of regularization
through noise injection, acceleration through rescaling and lastly
preprocessing may well lead to further inventions of layers and
techniques in the future.

On a more practical note, there are a number of aspects worth
remembering about batch normalization:

- During model training, batch normalization continuously adjusts the
  intermediate output of the network by utilizing the mean and standard
  deviation of the minibatch, so that the values of the intermediate
  output in each layer throughout the neural network are more stable.
- Batch normalization is slightly different for fully connected layers
  than for convolutional layers. In fact, for convolutional layers,
  layer normalization can sometimes be used as an alternative.
- Like a dropout layer, batch normalization layers have different
  behaviors in training mode than in prediction mode.
- Batch normalization is useful for regularization and improving
  convergence in optimization. By contrast, the original motivation of
  reducing internal covariate shift seems not to be a valid
  explanation.
- For more robust models that are less sensitive to input
  perturbations, consider removing batch normalization
  (Wang et al., 2022).

## 8.5.7. Exercises

1. Should we remove the bias parameter from the fully connected layer or
   the convolutional layer before the batch normalization? Why?
2. Compare the learning rates for LeNet with and without batch
   normalization.

   1. Plot the increase in validation accuracy.
   2. How large can you make the learning rate before the optimization
      fails in both cases?
3. Do we need batch normalization in every layer? Experiment with it.
4. Implement a “lite” version of batch normalization that only removes
   the mean, or alternatively one that only removes the variance. How
   does it behave?
5. Fix the parameters `beta` and `gamma`. Observe and analyze the
   results.
6. Can you replace dropout by batch normalization? How does the behavior
   change?
7. Research ideas: think of other normalization transforms that you can
   apply:

   1. Can you apply the probability integral transform?
   2. Can you use a full-rank covariance estimate? Why should you
      probably not do that?
   3. Can you use other compact matrix variants (block-diagonal,
      low-displacement rank, Monarch, etc.)?
   4. Does a sparsification compression act as a regularizer?
   5. Are there other projections (e.g., convex cone, symmetry
      group-specific transforms) that you can use?
