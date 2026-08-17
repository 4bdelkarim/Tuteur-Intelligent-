---
source_url: https://d2l.ai/chapter_computer-vision/index.html
title: 14. Computer Vision
chapter: '14'
section_number: '14'
date: null
extractor: d2l
source_type: web
---

# 14. Computer Vision

Whether it is medical diagnosis, self-driving vehicles, camera
monitoring, or smart filters, many applications in the field of computer
vision are closely related to our current and future lives. In recent
years, deep learning has been the transformative power for advancing the
performance of computer vision systems. It can be said that the most
advanced computer vision applications are almost inseparable from deep
learning. In view of this, this chapter will focus on the field of
computer vision, and investigate methods and applications that have
recently been influential in academia and industry.

In Section 7 and Section 8, we studied
various convolutional neural networks that are commonly used in computer
vision, and applied them to simple image classification tasks. At the
beginning of this chapter, we will describe two methods that may improve
model generalization, namely *image augmentation* and *fine-tuning*, and
apply them to image classification. Since deep neural networks can
effectively represent images in multiple levels, such layerwise
representations have been successfully used in various computer vision
tasks such as *object detection*, *semantic segmentation*, and *style
transfer*. Following the key idea of leveraging layerwise
representations in computer vision, we will begin with major components
and techniques for object detection. Next, we will show how to use
*fully convolutional networks* for semantic segmentation of images. Then
we will explain how to use style transfer techniques to generate images
like the cover of this book. In the end, we conclude this chapter by
applying the materials of this chapter and several previous chapters on
two popular computer vision benchmark datasets.

- 14.1. Image Augmentation
 - 14.1.1. Common Image Augmentation Methods
 - 14.1.2. Training with Image Augmentation
 - 14.1.3. Summary
 - 14.1.4. Exercises
- 14.2. Fine-Tuning
 - 14.2.1. Steps
 - 14.2.2. Hot Dog Recognition
 - 14.2.3. Summary
 - 14.2.4. Exercises
- 14.3. Object Detection and Bounding Boxes
 - 14.3.1. Bounding Boxes
 - 14.3.2. Summary
 - 14.3.3. Exercises
- 14.4. Anchor Boxes
 - 14.4.1. Generating Multiple Anchor Boxes
 - 14.4.2. Intersection over Union (IoU)
 - 14.4.3. Labeling Anchor Boxes in Training Data
 - 14.4.4. Predicting Bounding Boxes with Non-Maximum Suppression
 - 14.4.5. Summary
 - 14.4.6. Exercises
- 14.5. Multiscale Object Detection
 - 14.5.1. Multiscale Anchor Boxes
 - 14.5.2. Multiscale Detection
 - 14.5.3. Summary
 - 14.5.4. Exercises
- 14.6. The Object Detection Dataset
 - 14.6.1. Downloading the Dataset
 - 14.6.2. Reading the Dataset
 - 14.6.3. Demonstration
 - 14.6.4. Summary
 - 14.6.5. Exercises
- 14.7. Single Shot Multibox Detection
 - 14.7.1. Model
 - 14.7.2. Training
 - 14.7.3. Prediction
 - 14.7.4. Summary
 - 14.7.5. Exercises
- 14.8. Region-based CNNs (R-CNNs)
 - 14.8.1. R-CNNs
 - 14.8.2. Fast R-CNN
 - 14.8.3. Faster R-CNN
 - 14.8.4. Mask R-CNN
 - 14.8.5. Summary
 - 14.8.6. Exercises
- 14.9. Semantic Segmentation and the Dataset
 - 14.9.1. Image Segmentation and Instance Segmentation
 - 14.9.2. The Pascal VOC2012 Semantic Segmentation Dataset
 - 14.9.3. Summary
 - 14.9.4. Exercises
- 14.10. Transposed Convolution
 - 14.10.1. Basic Operation
 - 14.10.2. Padding, Strides, and Multiple Channels
 - 14.10.3. Connection to Matrix Transposition
 - 14.10.4. Summary
 - 14.10.5. Exercises
- 14.11. Fully Convolutional Networks
 - 14.11.1. The Model
 - 14.11.2. Initializing Transposed Convolutional Layers
 - 14.11.3. Reading the Dataset
 - 14.11.4. Training
 - 14.11.5. Prediction
 - 14.11.6. Summary
 - 14.11.7. Exercises
- 14.12. Neural Style Transfer
 - 14.12.1. Method
 - 14.12.2. Reading the Content and Style Images
 - 14.12.3. Preprocessing and Postprocessing
 - 14.12.4. Extracting Features
 - 14.12.5. Defining the Loss Function
 - 14.12.6. Initializing the Synthesized Image
 - 14.12.7. Training
 - 14.12.8. Summary
 - 14.12.9. Exercises
- 14.13. Image Classification (CIFAR-10) on Kaggle
 - 14.13.1. Obtaining and Organizing the Dataset
 - 14.13.2. Image Augmentation
 - 14.13.3. Reading the Dataset
 - 14.13.4. Defining the Model
 - 14.13.5. Defining the Training Function
 - 14.13.6. Training and Validating the Model
 - 14.13.7. Classifying the Testing Set and Submitting Results on Kaggle
 - 14.13.8. Summary
 - 14.13.9. Exercises
- 14.14. Dog Breed Identification (ImageNet Dogs) on Kaggle
 - 14.14.1. Obtaining and Organizing the Dataset
 - 14.14.2. Image Augmentation
 - 14.14.3. Reading the Dataset
 - 14.14.4. Fine-Tuning a Pretrained Model
 - 14.14.5. Defining the Training Function
 - 14.14.6. Training and Validating the Model
 - 14.14.7. Classifying the Testing Set and Submitting Results on Kaggle
 - 14.14.8. Summary
 - 14.14.9. Exercises
