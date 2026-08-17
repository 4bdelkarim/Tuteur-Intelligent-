---
source_url: "https://d2l.ai/chapter_hyperparameter-optimization/index.html"
title: "19. Hyperparameter Optimization"
chapter: "19"
section_number: "19"
date: null
extractor: "d2l"
---

# 19. Hyperparameter Optimization

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), and **Cedric
Archambeau** (*Amazon*)

The performance of every machine learning model depends on its
hyperparameters. They control the learning algorithm or the structure of
the underlying statistical model. However, there is no general way to
choose hyperparameters in practice. Instead, hyperparameters are often
set in a trial-and-error manner or sometimes left to their default
values by practitioners, leading to suboptimal generalization.

Hyperparameter optimization provides a systematic approach to this
problem, by casting it as an optimization problem: a good set of
hyperparameters should (at least) minimize a validation error. Compared
to most other optimization problems arising in machine learning,
hyperparameter optimization is a nested one, where each iteration
requires training and validating a machine learning model.

In this chapter, we will first introduce the basics of hyperparameter
optimization. We will also present some recent advancements that improve
the overall efficiency of hyperparameter optimization by exploiting
cheap-to-evaluate proxies of the original objective function. At the end
of this chapter, you should be able to apply state-of-the-art
hyperparameter optimization techniques to optimize the hyperparameter of
your own machine learning algorithm.

- 19.1. What Is Hyperparameter Optimization?
  - 19.1.1. The Optimization Problem
  - 19.1.2. Random Search
  - 19.1.3. Summary
  - 19.1.4. Exercises
- 19.2. Hyperparameter Optimization API
  - 19.2.1. Searcher
  - 19.2.2. Scheduler
  - 19.2.3. Tuner
  - 19.2.4. Bookkeeping the Performance of HPO Algorithms
  - 19.2.5. Example: Optimizing the Hyperparameters of a Convolutional Neural Network
  - 19.2.6. Comparing HPO Algorithms
  - 19.2.7. Summary
  - 19.2.8. Exercises
- 19.3. Asynchronous Random Search
  - 19.3.1. Objective Function
  - 19.3.2. Asynchronous Scheduler
  - 19.3.3. Visualize the Asynchronous Optimization Process
  - 19.3.4. Summary
  - 19.3.5. Exercises
- 19.4. Multi-Fidelity Hyperparameter Optimization
  - 19.4.1. Successive Halving
  - 19.4.2. Summary
- 19.5. Asynchronous Successive Halving
  - 19.5.1. Objective Function
  - 19.5.2. Asynchronous Scheduler
  - 19.5.3. Visualize the Optimization Process
  - 19.5.4. Summary
