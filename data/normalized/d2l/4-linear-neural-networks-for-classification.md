---
source_url: https://d2l.ai/chapter_linear-classification/index.html
title: 4. Linear Neural Networks for Classification
chapter: '4'
section_number: '4'
date: null
extractor: d2l
source_type: web
---

# 4. Linear Neural Networks for Classification

Now that you have worked through all of the mechanics you are ready to
apply the skills you have learned to broader kinds of tasks. Even as we
pivot towards classification, most of the plumbing remains the same:
loading the data, passing it through the model, generating output,
calculating the loss, taking gradients with respect to weights, and
updating the model. However, the precise form of the targets, the
parametrization of the output layer, and the choice of loss function
will adapt to suit the *classification* setting.

- 4.1. Softmax Regression
 - 4.1.1. Classification
 - 4.1.2. Loss Function
 - 4.1.3. Information Theory Basics
 - 4.1.4. Summary and Discussion
 - 4.1.5. Exercises
- 4.2. The Image Classification Dataset
 - 4.2.1. Loading the Dataset
 - 4.2.2. Reading a Minibatch
 - 4.2.3. Visualization
 - 4.2.4. Summary
 - 4.2.5. Exercises
- 4.3. The Base Classification Model
 - 4.3.1. The Classifier Class
 - 4.3.2. Accuracy
 - 4.3.3. Summary
 - 4.3.4. Exercises
- 4.4. Softmax Regression Implementation from Scratch
 - 4.4.1. The Softmax
 - 4.4.2. The Model
 - 4.4.3. The Cross-Entropy Loss
 - 4.4.4. Training
 - 4.4.5. Prediction
 - 4.4.6. Summary
 - 4.4.7. Exercises
- 4.5. Concise Implementation of Softmax Regression
 - 4.5.1. Defining the Model
 - 4.5.2. Softmax Revisited
 - 4.5.3. Training
 - 4.5.4. Summary
 - 4.5.5. Exercises
- 4.6. Generalization in Classification
 - 4.6.1. The Test Set
 - 4.6.2. Test Set Reuse
 - 4.6.3. Statistical Learning Theory
 - 4.6.4. Summary
 - 4.6.5. Exercises
- 4.7. Environment and Distribution Shift
 - 4.7.1. Types of Distribution Shift
 - 4.7.2. Examples of Distribution Shift
 - 4.7.3. Correction of Distribution Shift
 - 4.7.4. A Taxonomy of Learning Problems
 - 4.7.5. Fairness, Accountability, and Transparency in Machine Learning
 - 4.7.6. Summary
 - 4.7.7. Exercises
