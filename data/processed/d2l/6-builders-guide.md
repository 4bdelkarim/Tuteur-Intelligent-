---
source_url: https://d2l.ai/chapter_builders-guide/index.html
title: 6. Builders’ Guide
chapter: '6'
section_number: '6'
date: null
extractor: d2l
source_type: web
---

# 6. Builders’ Guide

Alongside giant datasets and powerful hardware, great software tools
have played an indispensable role in the rapid progress of deep
learning. Starting with the pathbreaking Theano library released in
2007, flexible open-source tools have enabled researchers to rapidly
prototype models, avoiding repetitive work when recycling standard
components while still maintaining the ability to make low-level
modifications. Over time, deep learning’s libraries have evolved to
offer increasingly coarse abstractions. Just as semiconductor designers
went from specifying transistors to logical circuits to writing code,
neural networks researchers have moved from thinking about the behavior
of individual artificial neurons to conceiving of networks in terms of
whole layers, and now often design architectures with far coarser
*blocks* in mind.

So far, we have introduced some basic machine learning concepts, ramping
up to fully-functional deep learning models. In the last chapter, we
implemented each component of an MLP from scratch and even showed how to
leverage high-level APIs to roll out the same models effortlessly. To
get you that far that fast, we *called upon* the libraries, but skipped
over more advanced details about *how they work*. In this chapter, we
will peel back the curtain, digging deeper into the key components of
deep learning computation, namely model construction, parameter access
and initialization, designing custom layers and blocks, reading and
writing models to disk, and leveraging GPUs to achieve dramatic
speedups. These insights will move you from *end user* to *power user*,
giving you the tools needed to reap the benefits of a mature deep
learning library while retaining the flexibility to implement more
complex models, including those you invent yourself! While this chapter
does not introduce any new models or datasets, the advanced modeling
chapters that follow rely heavily on these techniques.

- 6.1. Layers and Modules
 - 6.1.1. A Custom Module
 - 6.1.2. The Sequential Module
 - 6.1.3. Executing Code in the Forward Propagation Method
 - 6.1.4. Summary
 - 6.1.5. Exercises
- 6.2. Parameter Management
 - 6.2.1. Parameter Access
 - 6.2.2. Tied Parameters
 - 6.2.3. Summary
 - 6.2.4. Exercises
- 6.3. Parameter Initialization
 - 6.3.1. Built-in Initialization
 - 6.3.2. Summary
 - 6.3.3. Exercises
- 6.4. Lazy Initialization
 - 6.4.1. Summary
 - 6.4.2. Exercises
- 6.5. Custom Layers
 - 6.5.1. Layers without Parameters
 - 6.5.2. Layers with Parameters
 - 6.5.3. Summary
 - 6.5.4. Exercises
- 6.6. File I/O
 - 6.6.1. Loading and Saving Tensors
 - 6.6.2. Loading and Saving Model Parameters
 - 6.6.3. Summary
 - 6.6.4. Exercises
- 6.7. GPUs
 - 6.7.1. Computing Devices
 - 6.7.2. Tensors and GPUs
 - 6.7.3. Neural Networks and GPUs
 - 6.7.4. Summary
 - 6.7.5. Exercises
