---
source_url: https://d2l.ai/chapter_convolutional-modern/index.html
title: 8. Modern Convolutional Neural Networks
chapter: '8'
section_number: '8'
date: null
extractor: d2l
source_type: web
---

# 8. Modern Convolutional Neural Networks

Now that we understand the basics of wiring together CNNs, let’s take a
tour of modern CNN architectures. This tour is, by necessity,
incomplete, thanks to the plethora of exciting new designs being added.
Their importance derives from the fact that not only can they be used
directly for vision tasks, but they also serve as basic feature
generators for more advanced tasks such as tracking
(Zhang et al., 2021), segmentation
(Long et al., 2015), object detection
(Redmon and Farhadi, 2018), or style transformation
(Gatys et al., 2016). In this chapter, most sections
correspond to a significant CNN architecture that was at some point (or
currently) the base model upon which many research projects and deployed
systems were built. Each of these networks was briefly a dominant
architecture and many were winners or runners-up in the ImageNet
competition which has
served as a barometer of progress on supervised learning in computer
vision since 2010. It is only recently that Transformers have begun to
displace CNNs, starting with
Dosovitskiy *et al.* (2021) and followed by the Swin
Transformer (Liu et al., 2021). We will cover this development later
in Section 11.

While the idea of *deep* neural networks is quite simple (stack together
a bunch of layers), performance can vary wildly across architectures and
hyperparameter choices. The neural networks described in this chapter
are the product of intuition, a few mathematical insights, and a lot of
trial and error. We present these models in chronological order, partly
to convey a sense of the history so that you can form your own
intuitions about where the field is heading and perhaps develop your own
architectures. For instance, batch normalization and residual
connections described in this chapter have offered two popular ideas for
training and designing deep models, both of which have since also been
applied to architectures beyond computer vision.

We begin our tour of modern CNNs with AlexNet
(Krizhevsky et al., 2012), the first large-scale
network deployed to beat conventional computer vision methods on a
large-scale vision challenge; the VGG network
(Simonyan and Zisserman, 2014), which makes use of a number of
repeating blocks of elements; the network in network (NiN) that
convolves whole neural networks patch-wise over inputs
(Lin et al., 2013); GoogLeNet that uses networks with
multi-branch convolutions (Szegedy et al., 2015); the
residual network (ResNet) (He et al., 2016), which remains
one of the most popular off-the-shelf architectures in computer vision;
ResNeXt blocks (Xie et al., 2017) for sparser
connections; and DenseNet (Huang et al., 2017) for
a generalization of the residual architecture. Over time many special
optimizations for efficient networks have been developed, such as
coordinate shifts (ShiftNet) (Wu et al., 2018). This culminated in
the automatic search for efficient architectures such as MobileNet v3
(Howard et al., 2019). It also includes the
semi-automatic design exploration of
Radosavovic *et al.* (2020) that led to the
RegNetX/Y which we will discuss later in this chapter. The work is
instructive insofar as it offers a path for marrying brute force
computation with the ingenuity of an experimenter in the search for
efficient design spaces. Of note is also the work of
Liu *et al.* (2022) as it shows that training techniques (e.g.,
optimizers, data augmentation, and regularization) play a pivotal role
in improving accuracy. It also shows that long-held assumptions, such as
the size of a convolution window, may need to be revisited, given the
increase in computation and data. We will cover this and many more
questions in due course throughout this chapter.

- 8.1. Deep Convolutional Neural Networks (AlexNet)
 - 8.1.1. Representation Learning
 - 8.1.2. AlexNet
 - 8.1.3. Training
 - 8.1.4. Discussion
 - 8.1.5. Exercises
- 8.2. Networks Using Blocks (VGG)
 - 8.2.1. VGG Blocks
 - 8.2.2. VGG Network
 - 8.2.3. Training
 - 8.2.4. Summary
 - 8.2.5. Exercises
- 8.3. Network in Network (NiN)
 - 8.3.1. NiN Blocks
 - 8.3.2. NiN Model
 - 8.3.3. Training
 - 8.3.4. Summary
 - 8.3.5. Exercises
- 8.4. Multi-Branch Networks (GoogLeNet)
 - 8.4.1. Inception Blocks
 - 8.4.2. GoogLeNet Model
 - 8.4.3. Training
 - 8.4.4. Discussion
 - 8.4.5. Exercises
- 8.5. Batch Normalization
 - 8.5.1. Training Deep Networks
 - 8.5.2. Batch Normalization Layers
 - 8.5.3. Implementation from Scratch
 - 8.5.4. LeNet with Batch Normalization
 - 8.5.5. Concise Implementation
 - 8.5.6. Discussion
 - 8.5.7. Exercises
- 8.6. Residual Networks (ResNet) and ResNeXt
 - 8.6.1. Function Classes
 - 8.6.2. Residual Blocks
 - 8.6.3. ResNet Model
 - 8.6.4. Training
 - 8.6.5. ResNeXt
 - 8.6.6. Summary and Discussion
 - 8.6.7. Exercises
- 8.7. Densely Connected Networks (DenseNet)
 - 8.7.1. From ResNet to DenseNet
 - 8.7.2. Dense Blocks
 - 8.7.3. Transition Layers
 - 8.7.4. DenseNet Model
 - 8.7.5. Training
 - 8.7.6. Summary and Discussion
 - 8.7.7. Exercises
- 8.8. Designing Convolution Network Architectures
 - 8.8.1. The AnyNet Design Space
 - 8.8.2. Distributions and Parameters of Design Spaces
 - 8.8.3. RegNet
 - 8.8.4. Training
 - 8.8.5. Discussion
 - 8.8.6. Exercises
