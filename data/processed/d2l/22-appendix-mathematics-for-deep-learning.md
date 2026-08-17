---
source_url: https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html
title: '22. Appendix: Mathematics for Deep Learning'
chapter: '22'
section_number: '22'
date: null
extractor: d2l
source_type: web
---

# 22. Appendix: Mathematics for Deep Learning

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), and authors of
this book

One of the wonderful parts of modern deep learning is the fact that much
of it can be understood and used without a full understanding of the
mathematics below it. This is a sign that the field is maturing. Just as
most software developers no longer need to worry about the theory of
computable functions, neither should deep learning practitioners need to
worry about the theoretical foundations of maximum likelihood learning.

But, we are not quite there yet.

In practice, you will sometimes need to understand how architectural
choices influence gradient flow, or the implicit assumptions you make by
training with a certain loss function. You might need to know what in
the world entropy measures, and how it can help you understand exactly
what bits-per-character means in your model. These all require deeper
mathematical understanding.

This appendix aims to provide you the mathematical background you need
to understand the core theory of modern deep learning, but it is not
exhaustive. We will begin with examining linear algebra in greater
depth. We develop a geometric understanding of all the common linear
algebraic objects and operations that will enable us to visualize the
effects of various transformations on our data. A key element is the
development of the basics of eigen-decompositions.

We next develop the theory of differential calculus to the point that we
can fully understand why the gradient is the direction of steepest
descent, and why back-propagation takes the form it does. Integral
calculus is then discussed to the degree needed to support our next
topic, probability theory.

Problems encountered in practice frequently are not certain, and thus we
need a language to speak about uncertain things. We review the theory of
random variables and the most commonly encountered distributions so we
may discuss models probabilistically. This provides the foundation for
the naive Bayes classifier, a probabilistic classification technique.

Closely related to probability theory is the study of statistics. While
statistics is far too large a field to do justice in a short section, we
will introduce fundamental concepts that all machine learning
practitioners should be aware of, in particular: evaluating and
comparing estimators, conducting hypothesis tests, and constructing
confidence intervals.

Last, we turn to the topic of information theory, which is the
mathematical study of information storage and transmission. This
provides the core language by which we may discuss quantitatively how
much information a model holds on a domain of discourse.

Taken together, these form the core of the mathematical concepts needed
to begin down the path towards a deep understanding of deep learning.

- 22.1. Geometry and Linear Algebraic Operations
 - 22.1.1. Geometry of Vectors
 - 22.1.2. Dot Products and Angles
 - 22.1.3. Hyperplanes
 - 22.1.4. Geometry of Linear Transformations
 - 22.1.5. Linear Dependence
 - 22.1.6. Rank
 - 22.1.7. Invertibility
 - 22.1.8. Determinant
 - 22.1.9. Tensors and Common Linear Algebra Operations
 - 22.1.10. Summary
 - 22.1.11. Exercises
- 22.2. Eigendecompositions
 - 22.2.1. Finding Eigenvalues
 - 22.2.2. Decomposing Matrices
 - 22.2.3. Operations on Eigendecompositions
 - 22.2.4. Eigendecompositions of Symmetric Matrices
 - 22.2.5. Gershgorin Circle Theorem
 - 22.2.6. A Useful Application: The Growth of Iterated Maps
 - 22.2.7. Discussion
 - 22.2.8. Summary
 - 22.2.9. Exercises
- 22.3. Single Variable Calculus
 - 22.3.1. Differential Calculus
 - 22.3.2. Rules of Calculus
 - 22.3.3. Summary
 - 22.3.4. Exercises
- 22.4. Multivariable Calculus
 - 22.4.1. Higher-Dimensional Differentiation
 - 22.4.2. Geometry of Gradients and Gradient Descent
 - 22.4.3. A Note on Mathematical Optimization
 - 22.4.4. Multivariate Chain Rule
 - 22.4.5. The Backpropagation Algorithm
 - 22.4.6. Hessians
 - 22.4.7. A Little Matrix Calculus
 - 22.4.8. Summary
 - 22.4.9. Exercises
- 22.5. Integral Calculus
 - 22.5.1. Geometric Interpretation
 - 22.5.2. The Fundamental Theorem of Calculus
 - 22.5.3. Change of Variables
 - 22.5.4. A Comment on Sign Conventions
 - 22.5.5. Multiple Integrals
 - 22.5.6. Change of Variables in Multiple Integrals
 - 22.5.7. Summary
 - 22.5.8. Exercises
- 22.6. Random Variables
 - 22.6.1. Continuous Random Variables
 - 22.6.2. Summary
 - 22.6.3. Exercises
- 22.7. Maximum Likelihood
 - 22.7.1. The Maximum Likelihood Principle
 - 22.7.2. Numerical Optimization and the Negative Log-Likelihood
 - 22.7.3. Maximum Likelihood for Continuous Variables
 - 22.7.4. Summary
 - 22.7.5. Exercises
- 22.8. Distributions
 - 22.8.1. Bernoulli
 - 22.8.2. Discrete Uniform
 - 22.8.3. Continuous Uniform
 - 22.8.4. Binomial
 - 22.8.5. Poisson
 - 22.8.6. Gaussian
 - 22.8.7. Exponential Family
 - 22.8.8. Summary
 - 22.8.9. Exercises
- 22.9. Naive Bayes
 - 22.9.1. Optical Character Recognition
 - 22.9.2. The Probabilistic Model for Classification
 - 22.9.3. The Naive Bayes Classifier
 - 22.9.4. Training
 - 22.9.5. Summary
 - 22.9.6. Exercises
- 22.10. Statistics
 - 22.10.1. Evaluating and Comparing Estimators
 - 22.10.2. Conducting Hypothesis Tests
 - 22.10.3. Constructing Confidence Intervals
 - 22.10.4. Summary
 - 22.10.5. Exercises
- 22.11. Information Theory
 - 22.11.1. Information
 - 22.11.2. Entropy
 - 22.11.3. Mutual Information
 - 22.11.4. Kullback–Leibler Divergence
 - 22.11.5. Cross-Entropy
 - 22.11.6. Summary
 - 22.11.7. Exercises
