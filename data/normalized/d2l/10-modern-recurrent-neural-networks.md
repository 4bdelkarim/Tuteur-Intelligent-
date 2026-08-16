---
source_url: https://d2l.ai/chapter_recurrent-modern/index.html
title: 10. Modern Recurrent Neural Networks
chapter: '10'
section_number: '10'
date: null
extractor: d2l
source_type: web
---

# 10. Modern Recurrent Neural Networks

The previous chapter introduced the key ideas behind recurrent neural
networks (RNNs). However, just as with convolutional neural networks,
there has been a tremendous amount of innovation in RNN architectures,
culminating in several complex designs that have proven successful in
practice. In particular, the most popular designs feature mechanisms for
mitigating the notorious numerical instability faced by RNNs, as
typified by vanishing and exploding gradients. Recall that in
Section 9 we dealt with exploding gradients by applying a
blunt gradient clipping heuristic. Despite the efficacy of this hack, it
leaves open the problem of vanishing gradients.

In this chapter, we introduce the key ideas behind the most successful
RNN architectures for sequences, which stem from two papers. The first,
*Long Short-Term Memory* (Hochreiter and Schmidhuber, 1997),
introduces the *memory cell*, a unit of computation that replaces
traditional nodes in the hidden layer of a network. With these memory
cells, networks are able to overcome difficulties with training
encountered by earlier recurrent networks. Intuitively, the memory cell
avoids the vanishing gradient problem by keeping values in each memory
cell’s internal state cascading along a recurrent edge with weight 1
across many successive time steps. A set of multiplicative gates help
the network to determine not only the inputs to allow into the memory
state, but when the content of the memory state should influence the
model’s output.

The second paper, *Bidirectional Recurrent Neural Networks*
(Schuster and Paliwal, 1997), introduces an architecture in which
information from both the future (subsequent time steps) and the past
(preceding time steps) are used to determine the output at any point in
the sequence. This is in contrast to previous networks, in which only
past input can affect the output. Bidirectional RNNs have become a
mainstay for sequence labeling tasks in natural language processing,
among a myriad of other tasks. Fortunately, the two innovations are not
mutually exclusive, and have been successfully combined for phoneme
classification (Graves and Schmidhuber, 2005) and handwriting
recognition (Graves et al., 2008).

The first sections in this chapter will explain the LSTM architecture, a
lighter-weight version called the gated recurrent unit (GRU), the key
ideas behind bidirectional RNNs and a brief explanation of how RNN
layers are stacked together to form deep RNNs. Subsequently, we will
explore the application of RNNs in sequence-to-sequence tasks,
introducing machine translation along with key ideas such as
*encoder–decoder* architectures and *beam search*.

- 10.1. Long Short-Term Memory (LSTM)
 - 10.1.1. Gated Memory Cell
 - 10.1.2. Implementation from Scratch
 - 10.1.3. Concise Implementation
 - 10.1.4. Summary
 - 10.1.5. Exercises
- 10.2. Gated Recurrent Units (GRU)
 - 10.2.1. Reset Gate and Update Gate
 - 10.2.2. Candidate Hidden State
 - 10.2.3. Hidden State
 - 10.2.4. Implementation from Scratch
 - 10.2.5. Concise Implementation
 - 10.2.6. Summary
 - 10.2.7. Exercises
- 10.3. Deep Recurrent Neural Networks
 - 10.3.1. Implementation from Scratch
 - 10.3.2. Concise Implementation
 - 10.3.3. Summary
 - 10.3.4. Exercises
- 10.4. Bidirectional Recurrent Neural Networks
 - 10.4.1. Implementation from Scratch
 - 10.4.2. Concise Implementation
 - 10.4.3. Summary
 - 10.4.4. Exercises
- 10.5. Machine Translation and the Dataset
 - 10.5.1. Downloading and Preprocessing the Dataset
 - 10.5.2. Tokenization
 - 10.5.3. Loading Sequences of Fixed Length
 - 10.5.4. Reading the Dataset
 - 10.5.5. Summary
 - 10.5.6. Exercises
- 10.6. The Encoder–Decoder Architecture
 - 10.6.1. Encoder
 - 10.6.2. Decoder
 - 10.6.3. Putting the Encoder and Decoder Together
 - 10.6.4. Summary
 - 10.6.5. Exercises
- 10.7. Sequence-to-Sequence Learning for Machine Translation
 - 10.7.1. Teacher Forcing
 - 10.7.2. Encoder
 - 10.7.3. Decoder
 - 10.7.4. Encoder–Decoder for Sequence-to-Sequence Learning
 - 10.7.5. Loss Function with Masking
 - 10.7.6. Training
 - 10.7.7. Prediction
 - 10.7.8. Evaluation of Predicted Sequences
 - 10.7.9. Summary
 - 10.7.10. Exercises
- 10.8. Beam Search
 - 10.8.1. Greedy Search
 - 10.8.2. Exhaustive Search
 - 10.8.3. Beam Search
 - 10.8.4. Summary
 - 10.8.5. Exercises
