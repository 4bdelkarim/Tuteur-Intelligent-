---
source_url: "https://d2l.ai/chapter_recommender-systems/index.html"
title: "21. Recommender Systems"
chapter: "21"
section_number: "21"
date: null
extractor: "d2l"
---

# 21. Recommender Systems

**Shuai Zhang** (*Amazon*), **Aston Zhang** (*Amazon*), and **Yi Tay**
(*Google*)

Recommender systems are widely employed in industry and are ubiquitous
in our daily lives. These systems are utilized in a number of areas such
as online shopping sites (e.g., amazon.com), music/movie services site
(e.g., Netflix and Spotify), mobile application stores (e.g., IOS app
store and google play), online advertising, just to name a few.

The major goal of recommender systems is to help users discover relevant
items such as movies to watch, text to read or products to buy, so as to
create a delightful user experience. Moreover, recommender systems are
among the most powerful machine learning systems that online retailers
implement in order to drive incremental revenue. Recommender systems are
replacements of search engines by reducing the efforts in proactive
searches and surprising users with offers they never searched for. Many
companies managed to position themselves ahead of their competitors with
the help of more effective recommender systems. As such, recommender
systems are central to not only our everyday lives but also highly
indispensable in some industries.

In this chapter, we will cover the fundamentals and advancements of
recommender systems, along with exploring some common fundamental
techniques for building recommender systems with different data sources
available and their implementations. Specifically, you will learn how to
predict the rating a user might give to a prospective item, how to
generate a recommendation list of items and how to predict the
click-through rate from abundant features. These tasks are commonplace
in real-world applications. By studying this chapter, you will get
hands-on experience pertaining to solving real world recommendation
problems with not only classical methods but the more advanced deep
learning based models as well.

- 21.1. Overview of Recommender Systems
  - 21.1.1. Collaborative Filtering
  - 21.1.2. Explicit Feedback and Implicit Feedback
  - 21.1.3. Recommendation Tasks
  - 21.1.4. Summary
  - 21.1.5. Exercises
- 21.2. The MovieLens Dataset
  - 21.2.1. Getting the Data
  - 21.2.2. Statistics of the Dataset
  - 21.2.3. Splitting the dataset
  - 21.2.4. Loading the data
  - 21.2.5. Summary
  - 21.2.6. Exercises
- 21.3. Matrix Factorization
  - 21.3.1. The Matrix Factorization Model
  - 21.3.2. Model Implementation
  - 21.3.3. Evaluation Measures
  - 21.3.4. Training and Evaluating the Model
  - 21.3.5. Summary
  - 21.3.6. Exercises
- 21.4. AutoRec: Rating Prediction with Autoencoders
  - 21.4.1. Model
  - 21.4.2. Implementing the Model
  - 21.4.3. Reimplementing the Evaluator
  - 21.4.4. Training and Evaluating the Model
  - 21.4.5. Summary
  - 21.4.6. Exercises
- 21.5. Personalized Ranking for Recommender Systems
  - 21.5.1. Bayesian Personalized Ranking Loss and its Implementation
  - 21.5.2. Hinge Loss and its Implementation
  - 21.5.3. Summary
  - 21.5.4. Exercises
- 21.6. Neural Collaborative Filtering for Personalized Ranking
  - 21.6.1. The NeuMF model
  - 21.6.2. Model Implementation
  - 21.6.3. Customized Dataset with Negative Sampling
  - 21.6.4. Evaluator
  - 21.6.5. Training and Evaluating the Model
  - 21.6.6. Summary
  - 21.6.7. Exercises
- 21.7. Sequence-Aware Recommender Systems
  - 21.7.1. Model Architectures
  - 21.7.2. Model Implementation
  - 21.7.3. Sequential Dataset with Negative Sampling
  - 21.7.4. Load the MovieLens 100K dataset
  - 21.7.5. Train the Model
  - 21.7.6. Summary
  - 21.7.7. Exercises
- 21.8. Feature-Rich Recommender Systems
  - 21.8.1. An Online Advertising Dataset
  - 21.8.2. Dataset Wrapper
  - 21.8.3. Summary
  - 21.8.4. Exercises
- 21.9. Factorization Machines
  - 21.9.1. 2-Way Factorization Machines
  - 21.9.2. An Efficient Optimization Criterion
  - 21.9.3. Model Implementation
  - 21.9.4. Load the Advertising Dataset
  - 21.9.5. Train the Model
  - 21.9.6. Summary
  - 21.9.7. Exercises
- 21.10. Deep Factorization Machines
  - 21.10.1. Model Architectures
  - 21.10.2. Implementation of DeepFM
  - 21.10.3. Training and Evaluating the Model
  - 21.10.4. Summary
  - 21.10.5. Exercises
