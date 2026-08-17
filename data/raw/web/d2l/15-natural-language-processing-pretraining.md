---
source_url: "https://d2l.ai/chapter_natural-language-processing-pretraining/index.html"
title: "15. Natural Language Processing: Pretraining"
chapter: "15"
section_number: "15"
date: null
extractor: "d2l"
---

# 15. Natural Language Processing: Pretraining

Humans need to communicate. Out of this basic need of the human
condition, a vast amount of written text has been generated on an
everyday basis. Given rich text in social media, chat apps, emails,
product reviews, news articles, research papers, and books, it becomes
vital to enable computers to understand them to offer assistance or make
decisions based on human languages.

*Natural language processing* studies interactions between computers and
humans using natural languages. In practice, it is very common to use
natural language processing techniques to process and analyze text
(human natural language) data, such as language models in
Section 9.3 and machine translation models in
Section 10.5.

To understand text, we can begin by learning its representations.
Leveraging the existing text sequences from large corpora,
*self-supervised learning* has been extensively used to pretrain text
representations, such as by predicting some hidden part of the text
using some other part of their surrounding text. In this way, models
learn through supervision from *massive* text data without *expensive*
labeling efforts!

As we will see in this chapter, when treating each word or subword as an
individual token, the representation of each token can be pretrained
using word2vec, GloVe, or subword embedding models on large corpora.
After pretraining, representation of each token can be a vector,
however, it remains the same no matter what the context is. For
instance, the vector representation of “bank” is the same in both “go to
the bank to deposit some money” and “go to the bank to sit down”. Thus,
many more recent pretraining models adapt representation of the same
token to different contexts. Among them is BERT, a much deeper
self-supervised model based on the Transformer encoder. In this chapter,
we will focus on how to pretrain such representations for text, as
highlighted in Fig. 15.1.

![../_images/nlp-map-pretrain.svg](../_images/nlp-map-pretrain.svg)

Fig. 15.1 Pretrained text representations can be fed to various deep learning
architectures for different downstream natural language processing
applications. This chapter focuses on the upstream text
representation pretraining.

For sight of the big picture, Fig. 15.1 shows
that the pretrained text representations can be fed to a variety of deep
learning architectures for different downstream natural language
processing applications. We will cover them in Section 16.

- 15.1. Word Embedding (word2vec)
  - 15.1.1. One-Hot Vectors Are a Bad Choice
  - 15.1.2. Self-Supervised word2vec
  - 15.1.3. The Skip-Gram Model
  - 15.1.4. The Continuous Bag of Words (CBOW) Model
  - 15.1.5. Summary
  - 15.1.6. Exercises
- 15.2. Approximate Training
  - 15.2.1. Negative Sampling
  - 15.2.2. Hierarchical Softmax
  - 15.2.3. Summary
  - 15.2.4. Exercises
- 15.3. The Dataset for Pretraining Word Embeddings
  - 15.3.1. Reading the Dataset
  - 15.3.2. Subsampling
  - 15.3.3. Extracting Center Words and Context Words
  - 15.3.4. Negative Sampling
  - 15.3.5. Loading Training Examples in Minibatches
  - 15.3.6. Putting It All Together
  - 15.3.7. Summary
  - 15.3.8. Exercises
- 15.4. Pretraining word2vec
  - 15.4.1. The Skip-Gram Model
  - 15.4.2. Training
  - 15.4.3. Applying Word Embeddings
  - 15.4.4. Summary
  - 15.4.5. Exercises
- 15.5. Word Embedding with Global Vectors (GloVe)
  - 15.5.1. Skip-Gram with Global Corpus Statistics
  - 15.5.2. The GloVe Model
  - 15.5.3. Interpreting GloVe from the Ratio of Co-occurrence Probabilities
  - 15.5.4. Summary
  - 15.5.5. Exercises
- 15.6. Subword Embedding
  - 15.6.1. The fastText Model
  - 15.6.2. Byte Pair Encoding
  - 15.6.3. Summary
  - 15.6.4. Exercises
- 15.7. Word Similarity and Analogy
  - 15.7.1. Loading Pretrained Word Vectors
  - 15.7.2. Applying Pretrained Word Vectors
  - 15.7.3. Summary
  - 15.7.4. Exercises
- 15.8. Bidirectional Encoder Representations from Transformers (BERT)
  - 15.8.1. From Context-Independent to Context-Sensitive
  - 15.8.2. From Task-Specific to Task-Agnostic
  - 15.8.3. BERT: Combining the Best of Both Worlds
  - 15.8.4. Input Representation
  - 15.8.5. Pretraining Tasks
  - 15.8.6. Putting It All Together
  - 15.8.7. Summary
  - 15.8.8. Exercises
- 15.9. The Dataset for Pretraining BERT
  - 15.9.1. Defining Helper Functions for Pretraining Tasks
  - 15.9.2. Transforming Text into the Pretraining Dataset
  - 15.9.3. Summary
  - 15.9.4. Exercises
- 15.10. Pretraining BERT
  - 15.10.1. Pretraining BERT
  - 15.10.2. Representing Text with BERT
  - 15.10.3. Summary
  - 15.10.4. Exercises
