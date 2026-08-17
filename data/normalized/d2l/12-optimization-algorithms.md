---
source_url: https://d2l.ai/chapter_optimization/index.html
title: 12. Optimization Algorithms
chapter: '12'
section_number: '12'
date: null
extractor: d2l
source_type: web
---

# 12. Optimization Algorithms

If you read the book in sequence up to this point you already used a
number of optimization algorithms to train deep learning models. They
were the tools that allowed us to continue updating model parameters and
to minimize the value of the loss function, as evaluated on the training
set. Indeed, anyone content with treating optimization as a black box
device to minimize objective functions in a simple setting might well
content oneself with the knowledge that there exists an array of
incantations of such a procedure (with names such as “SGD” and “Adam”).

To do well, however, some deeper knowledge is required. Optimization
algorithms are important for deep learning. On the one hand, training a
complex deep learning model can take hours, days, or even weeks. The
performance of the optimization algorithm directly affects the model’s
training efficiency. On the other hand, understanding the principles of
different optimization algorithms and the role of their hyperparameters
will enable us to tune the hyperparameters in a targeted manner to
improve the performance of deep learning models.

In this chapter, we explore common deep learning optimization algorithms
in depth. Almost all optimization problems arising in deep learning are
*nonconvex*. Nonetheless, the design and analysis of algorithms in the
context of *convex* problems have proven to be very instructive. It is
for that reason that this chapter includes a primer on convex
optimization and the proof for a very simple stochastic gradient descent
algorithm on a convex objective function.

- 12.1. Optimization and Deep Learning
 - 12.1.1. Goal of Optimization
 - 12.1.2. Optimization Challenges in Deep Learning
 - 12.1.3. Summary
 - 12.1.4. Exercises
- 12.2. Convexity
 - 12.2.1. Definitions
 - 12.2.2. Properties
 - 12.2.3. Constraints
 - 12.2.4. Summary
 - 12.2.5. Exercises
- 12.3. Gradient Descent
 - 12.3.1. One-Dimensional Gradient Descent
 - 12.3.2. Multivariate Gradient Descent
 - 12.3.3. Adaptive Methods
 - 12.3.4. Summary
 - 12.3.5. Exercises
- 12.4. Stochastic Gradient Descent
 - 12.4.1. Stochastic Gradient Updates
 - 12.4.2. Dynamic Learning Rate
 - 12.4.3. Convergence Analysis for Convex Objectives
 - 12.4.4. Stochastic Gradients and Finite Samples
 - 12.4.5. Summary
 - 12.4.6. Exercises
- 12.5. Minibatch Stochastic Gradient Descent
 - 12.5.1. Vectorization and Caches
 - 12.5.2. Minibatches
 - 12.5.3. Reading the Dataset
 - 12.5.4. Implementation from Scratch
 - 12.5.5. Concise Implementation
 - 12.5.6. Summary
 - 12.5.7. Exercises
- 12.6. Momentum
 - 12.6.1. Basics
 - 12.6.2. Practical Experiments
 - 12.6.3. Theoretical Analysis
 - 12.6.4. Summary
 - 12.6.5. Exercises
- 12.7. Adagrad
 - 12.7.1. Sparse Features and Learning Rates
 - 12.7.2. Preconditioning
 - 12.7.3. The Algorithm
 - 12.7.4. Implementation from Scratch
 - 12.7.5. Concise Implementation
 - 12.7.6. Summary
 - 12.7.7. Exercises
- 12.8. RMSProp
 - 12.8.1. The Algorithm
 - 12.8.2. Implementation from Scratch
 - 12.8.3. Concise Implementation
 - 12.8.4. Summary
 - 12.8.5. Exercises
- 12.9. Adadelta
 - 12.9.1. The Algorithm
 - 12.9.2. Implementation
 - 12.9.3. Summary
 - 12.9.4. Exercises
- 12.10. Adam
 - 12.10.1. The Algorithm
 - 12.10.2. Implementation
 - 12.10.3. Yogi
 - 12.10.4. Summary
 - 12.10.5. Exercises
- 12.11. Learning Rate Scheduling
 - 12.11.1. Toy Problem
 - 12.11.2. Schedulers
 - 12.11.3. Policies
 - 12.11.4. Summary
 - 12.11.5. Exercises
