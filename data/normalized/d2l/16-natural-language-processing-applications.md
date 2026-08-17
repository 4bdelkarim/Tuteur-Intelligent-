---
source_url: https://d2l.ai/chapter_natural-language-processing-applications/index.html
title: '16. Natural Language Processing: Applications'
chapter: '16'
section_number: '16'
date: null
extractor: d2l
source_type: web
---

# 16. Natural Language Processing: Applications

We have seen how to represent tokens in text sequences and train their
representations in Section 15. Such pretrained text
representations can be fed to various models for different downstream
natural language processing tasks.

In fact, earlier chapters have already discussed some natural language
processing applications *without pretraining*, just for explaining deep
learning architectures. For instance, in Section 9, we have
relied on RNNs to design language models to generate novella-like text.
In Section 10 and
Section 11, we have also designed
models based on RNNs and attention mechanisms for machine translation.

However, this book does not intend to cover all such applications in a
comprehensive manner. Instead, our focus is on *how to apply (deep)
representation learning of languages to addressing natural language
processing problems*. Given pretrained text representations, this
chapter will explore two popular and representative downstream natural
language processing tasks: sentiment analysis and natural language
inference, which analyze single text and relationships of text pairs,
respectively.

Fig. 16.1 Pretrained text representations can be fed to various deep learning
architectures for different downstream natural language processing
applications. This chapter focuses on how to design models for
different downstream natural language processing applications.

As depicted in Fig. 16.1, this chapter focuses on
describing the basic ideas of designing natural language processing
models using different types of deep learning architectures, such as
MLPs, CNNs, RNNs, and attention. Though it is possible to combine any
pretrained text representations with any architecture for either
application in Fig. 16.1, we select a few
representative combinations. Specifically, we will explore popular
architectures based on RNNs and CNNs for sentiment analysis. For natural
language inference, we choose attention and MLPs to demonstrate how to
analyze text pairs. In the end, we introduce how to fine-tune a
pretrained BERT model for a wide range of natural language processing
applications, such as on a sequence level (single text classification
and text pair classification) and a token level (text tagging and
question answering). As a concrete empirical case, we will fine-tune
BERT for natural language inference.

As we have introduced in Section 15.8, BERT requires minimal
architecture changes for a wide range of natural language processing
applications. However, this benefit comes at the cost of fine-tuning a
huge number of BERT parameters for the downstream applications. When
space or time is limited, those crafted models based on MLPs, CNNs,
RNNs, and attention are more feasible. In the following, we start by the
sentiment analysis application and illustrate the model design based on
RNNs and CNNs, respectively.

- 16.1. Sentiment Analysis and the Dataset
 - 16.1.1. Reading the Dataset
 - 16.1.2. Preprocessing the Dataset
 - 16.1.3. Creating Data Iterators
 - 16.1.4. Putting It All Together
 - 16.1.5. Summary
 - 16.1.6. Exercises
- 16.2. Sentiment Analysis: Using Recurrent Neural Networks
 - 16.2.1. Representing Single Text with RNNs
 - 16.2.2. Loading Pretrained Word Vectors
 - 16.2.3. Training and Evaluating the Model
 - 16.2.4. Summary
 - 16.2.5. Exercises
- 16.3. Sentiment Analysis: Using Convolutional Neural Networks
 - 16.3.1. One-Dimensional Convolutions
 - 16.3.2. Max-Over-Time Pooling
 - 16.3.3. The textCNN Model
 - 16.3.4. Summary
 - 16.3.5. Exercises
- 16.4. Natural Language Inference and the Dataset
 - 16.4.1. Natural Language Inference
 - 16.4.2. The Stanford Natural Language Inference (SNLI) Dataset
 - 16.4.3. Summary
 - 16.4.4. Exercises
- 16.5. Natural Language Inference: Using Attention
 - 16.5.1. The Model
 - 16.5.2. Training and Evaluating the Model
 - 16.5.3. Summary
 - 16.5.4. Exercises
- 16.6. Fine-Tuning BERT for Sequence-Level and Token-Level Applications
 - 16.6.1. Single Text Classification
 - 16.6.2. Text Pair Classification or Regression
 - 16.6.3. Text Tagging
 - 16.6.4. Question Answering
 - 16.6.5. Summary
 - 16.6.6. Exercises
- 16.7. Natural Language Inference: Fine-Tuning BERT
 - 16.7.1. Loading Pretrained BERT
 - 16.7.2. The Dataset for Fine-Tuning BERT
 - 16.7.3. Fine-Tuning BERT
 - 16.7.4. Summary
 - 16.7.5. Exercises
