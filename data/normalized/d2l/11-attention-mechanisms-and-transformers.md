---
source_url: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html
title: 11. Attention Mechanisms and Transformers
chapter: '11'
section_number: '11'
date: null
extractor: d2l
source_type: web
---

# 11. Attention Mechanisms and Transformers

The earliest years of the deep learning boom were driven primarily by
results produced using the multilayer perceptron, convolutional network,
and recurrent network architectures. Remarkably, the model architectures
that underpinned many of deep learning’s breakthroughs in the 2010s had
changed remarkably little relative to their antecedents despite the
lapse of nearly 30 years. While plenty of new methodological innovations
made their way into most practitioner’s toolkits—ReLU activations,
residual layers, batch normalization, dropout, and adaptive learning
rate schedules come to mind—the core underlying architectures were
clearly recognizable as scaled-up implementations of classic ideas.
Despite thousands of papers proposing alternative ideas, models
resembling classical convolutional neural networks
(Section 7) retained *state-of-the-art* status in computer
vision and models resembling Sepp Hochreiter’s original design for the
LSTM recurrent neural network (Section 10.1), dominated most
applications in natural language processing. Arguably, to that point,
the rapid emergence of deep learning appeared to be primarily
attributable to shifts in the available computational resources (thanks
to innovations in parallel computing with GPUs) and the availability of
massive data resources (thanks to cheap storage and Internet services).
While these factors may indeed remain the primary drivers behind this
technology’s increasing power we are also witnessing, at long last, a
sea change in the landscape of dominant architectures.

At the present moment, the dominant models for nearly all natural
language processing tasks are based on the Transformer architecture.
Given any new task in natural language processing, the default
first-pass approach is to grab a large Transformer-based pretrained
model, (e.g., BERT (Devlin et al., 2018), ELECTRA
(Clark et al., 2020), RoBERTa (Liu et al., 2019), or
Longformer (Beltagy et al., 2020)) adapting the output layers
as necessary, and fine-tuning the model on the available data for the
downstream task. If you have been paying attention to the last few years
of breathless news coverage centered on OpenAI’s large language models,
then you have been tracking a conversation centered on the GPT-2 and
GPT-3 Transformer-based models
(Brown et al., 2020, Radford et al., 2019). Meanwhile, the
vision Transformer has emerged as a default model for diverse vision
tasks, including image recognition, object detection, semantic
segmentation, and superresolution
(Dosovitskiy et al., 2021, Liu et al., 2021). Transformers
also showed up as competitive methods for speech recognition
(Gulati et al., 2020), reinforcement learning
(Chen et al., 2021), and graph neural networks
(Dwivedi and Bresson, 2020).

The core idea behind the Transformer model is the *attention mechanism*,
an innovation that was originally envisioned as an enhancement for
encoder–decoder RNNs applied to sequence-to-sequence applications, such
as machine translations (Bahdanau et al., 2014). You might
recall that in the first sequence-to-sequence models for machine
translation (Sutskever et al., 2014), the entire input was
compressed by the encoder into a single fixed-length vector to be fed
into the decoder. The intuition behind attention is that rather than
compressing the input, it might be better for the decoder to revisit the
input sequence at every step. Moreover, rather than always seeing the
same representation of the input, one might imagine that the decoder
should selectively focus on particular parts of the input sequence at
particular decoding steps. Bahdanau’s attention mechanism provided a
simple means by which the decoder could dynamically *attend* to
different parts of the input at each decoding step. The high-level idea
is that the encoder could produce a representation of length equal to
the original input sequence. Then, at decoding time, the decoder can
(via some control mechanism) receive as input a context vector
consisting of a weighted sum of the representations on the input at each
time step. Intuitively, the weights determine the extent to which each
step’s context “focuses” on each input token, and the key is to make
this process for assigning the weights differentiable so that it can be
learned along with all of the other neural network parameters.

Initially, the idea was a remarkably successful enhancement to the
recurrent neural networks that already dominated machine translation
applications. The models performed better than the original
encoder–decoder sequence-to-sequence architectures. Furthermore,
researchers noted that some nice qualitative insights sometimes emerged
from inspecting the pattern of attention weights. In translation tasks,
attention models often assigned high attention weights to cross-lingual
synonyms when generating the corresponding words in the target language.
For example, when translating the sentence “my feet hurt” to “j’ai mal
au pieds”, the neural network might assign high attention weights to the
representation of “feet” when generating the corresponding French word
“pieds”. These insights spurred claims that attention models confer
“interpretability” although what precisely the attention weights
mean—i.e., how, if at all, they should be *interpreted* remains a hazy
research topic.

However, attention mechanisms soon emerged as more significant concerns,
beyond their usefulness as an enhancement for encoder–decoder recurrent
neural networks and their putative usefulness for picking out salient
inputs. Vaswani *et al.* (2017) proposed the
Transformer architecture for machine translation, dispensing with
recurrent connections altogether, and instead relying on cleverly
arranged attention mechanisms to capture all relationships among input
and output tokens. The architecture performed remarkably well, and by
2018 the Transformer began showing up in the majority of
state-of-the-art natural language processing systems. Moreover, at the
same time, the dominant practice in natural language processing became
to pretrain large-scale models on enormous generic background corpora to
optimize some self-supervised pretraining objective, and then to
fine-tune these models using the available downstream data. The gap
between Transformers and traditional architectures grew especially wide
when applied in this pretraining paradigm, and thus the ascendance of
Transformers coincided with the ascendence of such large-scale
pretrained models, now sometimes called *foundation models*
(Bommasani et al., 2021).

In this chapter, we introduce attention models, starting with the most
basic intuitions and the simplest instantiations of the idea. We then
work our way up to the Transformer architecture, the vision Transformer,
and the landscape of modern Transformer-based pretrained models.

- 11.1. Queries, Keys, and Values
 - 11.1.1. Visualization
 - 11.1.2. Summary
 - 11.1.3. Exercises
- 11.2. Attention Pooling by Similarity
 - 11.2.1. Kernels and Data
 - 11.2.2. Attention Pooling via Nadaraya–Watson Regression
 - 11.2.3. Adapting Attention Pooling
 - 11.2.4. Summary
 - 11.2.5. Exercises
- 11.3. Attention Scoring Functions
 - 11.3.1. Dot Product Attention
 - 11.3.2. Convenience Functions
 - 11.3.3. Scaled Dot Product Attention
 - 11.3.4. Additive Attention
 - 11.3.5. Summary
 - 11.3.6. Exercises
- 11.4. The Bahdanau Attention Mechanism
 - 11.4.1. Model
 - 11.4.2. Defining the Decoder with Attention
 - 11.4.3. Training
 - 11.4.4. Summary
 - 11.4.5. Exercises
- 11.5. Multi-Head Attention
 - 11.5.1. Model
 - 11.5.2. Implementation
 - 11.5.3. Summary
 - 11.5.4. Exercises
- 11.6. Self-Attention and Positional Encoding
 - 11.6.1. Self-Attention
 - 11.6.2. Comparing CNNs, RNNs, and Self-Attention
 - 11.6.3. Positional Encoding
 - 11.6.4. Summary
 - 11.6.5. Exercises
- 11.7. The Transformer Architecture
 - 11.7.1. Model
 - 11.7.2. Positionwise Feed-Forward Networks
 - 11.7.3. Residual Connection and Layer Normalization
 - 11.7.4. Encoder
 - 11.7.5. Decoder
 - 11.7.6. Training
 - 11.7.7. Summary
 - 11.7.8. Exercises
- 11.8. Transformers for Vision
 - 11.8.1. Model
 - 11.8.2. Patch Embedding
 - 11.8.3. Vision Transformer Encoder
 - 11.8.4. Putting It All Together
 - 11.8.5. Training
 - 11.8.6. Summary and Discussion
 - 11.8.7. Exercises
- 11.9. Large-Scale Pretraining with Transformers
 - 11.9.1. Encoder-Only
 - 11.9.2. Encoder–Decoder
 - 11.9.3. Decoder-Only
 - 11.9.4. Scalability
 - 11.9.5. Large Language Models
 - 11.9.6. Summary and Discussion
 - 11.9.7. Exercises
