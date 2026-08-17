---
source_url: "https://d2l.ai/chapter_linear-regression/index.html"
title: "3. Linear Neural Networks for Regression"
chapter: "3"
section_number: "3"
date: null
extractor: "d2l"
---

# 3. Linear Neural Networks for Regression

Before we worry about making our neural networks deep, it will be
helpful to implement some shallow ones, for which the inputs connect
directly to the outputs. This will prove important for a few reasons.
First, rather than getting distracted by complicated architectures, we
can focus on the basics of neural network training, including
parametrizing the output layer, handling data, specifying a loss
function, and training the model. Second, this class of shallow networks
happens to comprise the set of linear models, which subsumes many
classical methods of statistical prediction, including linear and
softmax regression. Understanding these classical tools is pivotal
because they are widely used in many contexts and we will often need to
use them as baselines when justifying the use of fancier architectures.
This chapter will focus narrowly on linear regression and the next one
will extend our modeling repertoire by developing linear neural networks
for classification.

- 3.1. Linear Regression
  - 3.1.1. Basics
  - 3.1.2. Vectorization for Speed
  - 3.1.3. The Normal Distribution and Squared Loss
  - 3.1.4. Linear Regression as a Neural Network
  - 3.1.5. Summary
  - 3.1.6. Exercises
- 3.2. Object-Oriented Design for Implementation
  - 3.2.1. Utilities
  - 3.2.2. Models
  - 3.2.3. Data
  - 3.2.4. Training
  - 3.2.5. Summary
  - 3.2.6. Exercises
- 3.3. Synthetic Regression Data
  - 3.3.1. Generating the Dataset
  - 3.3.2. Reading the Dataset
  - 3.3.3. Concise Implementation of the Data Loader
  - 3.3.4. Summary
  - 3.3.5. Exercises
- 3.4. Linear Regression Implementation from Scratch
  - 3.4.1. Defining the Model
  - 3.4.2. Defining the Loss Function
  - 3.4.3. Defining the Optimization Algorithm
  - 3.4.4. Training
  - 3.4.5. Summary
  - 3.4.6. Exercises
- 3.5. Concise Implementation of Linear Regression
  - 3.5.1. Defining the Model
  - 3.5.2. Defining the Loss Function
  - 3.5.3. Defining the Optimization Algorithm
  - 3.5.4. Training
  - 3.5.5. Summary
  - 3.5.6. Exercises
- 3.6. Generalization
  - 3.6.1. Training Error and Generalization Error
  - 3.6.2. Underfitting or Overfitting?
  - 3.6.3. Model Selection
  - 3.6.4. Summary
  - 3.6.5. Exercises
- 3.7. Weight Decay
  - 3.7.1. Norms and Weight Decay
  - 3.7.2. High-Dimensional Linear Regression
  - 3.7.3. Implementation from Scratch
  - 3.7.4. Concise Implementation
  - 3.7.5. Summary
  - 3.7.6. Exercises
