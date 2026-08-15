---
source_url: "https://d2l.ai/chapter_optimization/adadelta.html"
title: "12.9. Adadelta"
chapter: "12"
section_number: "12.9"
date: null
extractor: "d2l"
---

# 12.9. Adadelta

Adadelta is yet another variant of AdaGrad (Section 12.7).
The main difference lies in the fact that it decreases the amount by
which the learning rate is adaptive to coordinates. Moreover,
traditionally it referred to as not having a learning rate since it uses
the amount of change itself as calibration for future change. The
algorithm was proposed in Zeiler (2012). It is fairly
straightforward, given the discussion of previous algorithms so far.

## 12.9.1. The Algorithm

In a nutshell, Adadelta uses two state variables, $\mathbf{s}_t$
to store a leaky average of the second moment of the gradient and
$\Delta\mathbf{x}_t$ to store a leaky average of the second moment
of the change of parameters in the model itself. Note that we use the
original notation and naming of the authors for compatibility with other
publications and implementations (there is no other real reason why one
should use different Greek variables to indicate a parameter serving the
same purpose in momentum, Adagrad, RMSProp, and Adadelta).

Here are the technical details of Adadelta. Given the parameter du jour
is $\rho$, we obtain the following leaky updates similarly to
Section 12.8:

$$
(12.9.1)\[\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}
$$

The difference to Section 12.8 is that we perform updates
with the rescaled gradient $\mathbf{g}_t'$, i.e.,

$$
(12.9.2)\[\begin{split}\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}\end{split}
$$

So what is the rescaled gradient $\mathbf{g}_t'$? We can calculate
it as follows:

$$
(12.9.3)\[\begin{split}\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}\end{split}
$$

where $\Delta \mathbf{x}_{t-1}$ is the leaky average of the
squared rescaled gradients $\mathbf{g}_t'$. We initialize
$\Delta \mathbf{x}_{0}$ to be $0$ and update it at each step
with $\mathbf{g}_t'$, i.e.,

$$
(12.9.4)\[\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}
$$

and $\epsilon$ (a small value such as $10^{-5}$) is added to
maintain numerical stability.

## 12.9.2. Implementation

Adadelta needs to maintain two state variables for each variable,
$\mathbf{s}_t$ and $\Delta\mathbf{x}_t$. This yields the
following implementation.

```python
%matplotlib inline
import torch
from d2l import torch as d2l

def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

Choosing $\rho = 0.9$ amounts to a half-life time of 10 for each
parameter update. This tends to work quite well. We get the following
behavior.

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

![../_images/output_adadelta_0b41cb_15_1.svg](../_images/output_adadelta_0b41cb_15_1.svg)

For a concise implementation we simply use the Adadelta algorithm from
high-level APIs. This yields the following one-liner for a much more
compact invocation.

```python
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

![../_images/output_adadelta_0b41cb_27_1.svg](../_images/output_adadelta_0b41cb_27_1.svg)

## 12.9.3. Summary

- Adadelta has no learning rate parameter. Instead, it uses the rate of
  change in the parameters itself to adapt the learning rate.
- Adadelta requires two state variables to store the second moments of
  gradient and the change in parameters.
- Adadelta uses leaky averages to keep a running estimate of the
  appropriate statistics.

## 12.9.4. Exercises

1. Adjust the value of $\rho$. What happens?
2. Show how to implement the algorithm without the use of
   $\mathbf{g}_t'$. Why might this be a good idea?
3. Is Adadelta really learning rate free? Could you find optimization
   problems that break Adadelta?
4. Compare Adadelta to Adagrad and RMS prop to discuss their convergence
   behavior.
