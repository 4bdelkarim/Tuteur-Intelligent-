---
source_url: https://d2l.ai/chapter_appendix-tools-for-deep-learning/sagemaker.html
title: 23.2. Using Amazon SageMaker
chapter: '23'
section_number: '23.2'
date: null
extractor: d2l
source_type: web
source: chapter_appendix-tools-for-deep-learning_sagemaker
---

# 23.2. Using Amazon SageMaker

Deep learning applications may demand so much computational resource
that easily goes beyond what your local machine can offer. Cloud
computing services allow you to run GPU-intensive code of this book more
easily using more powerful computers. This section will introduce how to
use Amazon SageMaker to run the code of this book.

## 23.2.1. Signing Up

First, we need to sign up an account at <https://aws.amazon.com/>. For
additional security, using two-factor authentication is encouraged. It
is also a good idea to set up detailed billing and spending alerts to
avoid any surprise, e.g., when forgetting to stop running instances.
After logging into your AWS account, go to your
console and search for “Amazon
SageMaker” (see Fig. 23.2.1), then click it to open the
SageMaker panel.

Fig. 23.2.1 Search for and open the SageMaker panel.

## 23.2.2. Creating a SageMaker Instance

Next, let’s create a notebook instance as described in
Fig. 23.2.2.

Fig. 23.2.2 Create a SageMaker instance.

SageMaker provides multiple instance
types with
varying computational power and prices. When creating a notebook
instance, we can specify its name and type. In
Fig. 23.2.3, we choose `ml.p3.2xlarge`: with
one Tesla V100 GPU and an 8-core CPU, this instance is powerful enough
for most of the book.

Fig. 23.2.3 Choose the instance type.

The entire book in the ipynb format for running with SageMaker is
available at <https://github.com/d2l-ai/d2l-pytorch-sagemaker>. We can
specify this GitHub repository URL (Fig. 23.2.4)
to allow SageMaker to clone it when creating the instance.

Fig. 23.2.4 Specify the GitHub repository.

## 23.2.3. Running and Stopping an Instance

Creating an instance may take a few minutes. When it is ready, click on
the “Open Jupyter” link next to it (Fig. 23.2.5) so
you can edit and run all the Jupyter notebooks of this book on this
instance (similar to steps in Section 23.1).

Fig. 23.2.5 Open Jupyter on the created SageMaker instance.

After finishing your work, do not forget to stop the instance to avoid
being charged further (Fig. 23.2.6).

Fig. 23.2.6 Stop a SageMaker instance.

## 23.2.4. Updating Notebooks

Notebooks of this open-source book will be regularly updated in the
d2l-ai/d2l-pytorch-sagemaker
repository on GitHub. To update to the latest version, you may open a
terminal on the SageMaker instance (Fig. 23.2.7).

Fig. 23.2.7 Open a terminal on the SageMaker instance.

You may wish to commit your local changes before pulling updates from
the remote repository. Otherwise, simply discard all your local changes
with the following commands in the terminal:

```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```

## 23.2.5. Summary

- We can create a notebook instance using Amazon SageMaker to run
 GPU-intensive code of this book.
- We can update notebooks via the terminal on the Amazon SageMaker
 instance.
