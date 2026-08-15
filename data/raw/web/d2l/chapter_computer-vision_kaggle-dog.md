---
source_url: "https://d2l.ai/chapter_computer-vision/kaggle-dog.html"
title: "14.14. Dog Breed Identification (ImageNet Dogs) on Kaggle"
chapter: "14"
section_number: "14.14"
date: null
extractor: "d2l"
---

# 14.14. Dog Breed Identification (ImageNet Dogs) on Kaggle

In this section, we will practice the dog breed identification problem
on Kaggle. The web address of this competition is
<https://www.kaggle.com/c/dog-breed-identification>

In this competition, 120 different breeds of dogs will be recognized. In
fact, the dataset for this competition is a subset of the ImageNet
dataset. Unlike the images in the CIFAR-10 dataset in
Section 14.13, the images in the ImageNet dataset are
both higher and wider in varying dimensions. Fig. 14.14.1
shows the information on the competition’s webpage. You need a Kaggle
account to submit your results.

Fig. 14.14.1 The dog breed identification competition website. The competition
dataset can be obtained by clicking the “Data” tab.

```python
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
```

## 14.14.1. Obtaining and Organizing the Dataset

The competition dataset is divided into a training set and a test set,
which contain 10222 and 10357 JPEG images of three RGB (color) channels,
respectively. Among the training dataset, there are 120 breeds of dogs
such as Labradors, Poodles, Dachshunds, Samoyeds, Huskies, Chihuahuas,
and Yorkshire Terriers.

### 14.14.1.1. Downloading the Dataset

After logging into Kaggle, you can click on the “Data” tab on the
competition webpage shown in Fig. 14.14.1 and download the
dataset by clicking the “Download All” button. After unzipping the
downloaded file in `../data`, you will find the entire dataset in the
following paths:

- ../data/dog-breed-identification/labels.csv
- ../data/dog-breed-identification/sample\_submission.csv
- ../data/dog-breed-identification/train
- ../data/dog-breed-identification/test

You may have noticed that the above structure is similar to that of the
CIFAR-10 competition in Section 14.13, where folders
`train/` and `test/` contain training and testing dog images,
respectively, and `labels.csv` contains the labels for the training
images. Similarly, to make it easier to get started, we provide a small
sample of the dataset mentioned above: `train_valid_test_tiny.zip`. If
you are going to use the full dataset for the Kaggle competition, you
need to change the `demo` variable below to `False`.

```python
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### 14.14.1.2. Organizing the Dataset

We can organize the dataset similarly to what we did in
Section 14.13, namely splitting out a validation set
from the original training set, and moving images into subfolders
grouped by labels.

The `reorg_dog_data` function below reads the training data labels,
splits out the validation set, and organizes the training set.

```python
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## 14.14.2. Image Augmentation

Recall that this dog breed dataset is a subset of the ImageNet dataset,
whose images are larger than those of the CIFAR-10 dataset in
Section 14.13. The following lists a few image
augmentation operations that might be useful for relatively larger
images.

```python
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

During prediction, we only use image preprocessing operations without
randomness.

```python
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## 14.14.3. Reading the Dataset

As in Section 14.13, we can read the organized dataset
consisting of raw image files.

```python
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

Below we create data iterator instances the same way as in
Section 14.13.

```python
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## 14.14.4. Fine-Tuning a Pretrained Model

Again, the dataset for this competition is a subset of the ImageNet
dataset. Therefore, we can use the approach discussed in
Section 14.2 to select a model pretrained on the full
ImageNet dataset and use it to extract image features to be fed into a
custom small-scale output network. High-level APIs of deep learning
frameworks provide a wide range of models pretrained on the ImageNet
dataset. Here, we choose a pretrained ResNet-34 model, where we simply
reuse the input of this model’s output layer (i.e., the extracted
features). Then we can replace the original output layer with a small
custom output network that can be trained, such as stacking two fully
connected layers. Different from the experiment in
Section 14.2, the following does not retrain the
pretrained model used for feature extraction. This reduces training time
and memory for storing gradients.

Recall that we standardized images using the means and standard
deviations of the three RGB channels for the full ImageNet dataset. In
fact, this is also consistent with the standardization operation by the
pretrained model on ImageNet.

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

Before calculating the loss, we first obtain the input of the pretrained
model’s output layer, i.e., the extracted feature. Then we use this
feature as input for our small custom output network to calculate the
loss.

```python
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## 14.14.5. Defining the Training Function

We will select the model and tune hyperparameters according to the
model’s performance on the validation set. The model training function
`train` only iterates parameters of the small custom output network.

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## 14.14.6. Training and Validating the Model

Now we can train and validate the model. The following hyperparameters
are all tunable. For example, the number of epochs can be increased.
Because `lr_period` and `lr_decay` are set to 2 and 0.9,
respectively, the learning rate of the optimization algorithm will be
multiplied by 0.9 after every 2 epochs.

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

![../_images/output_kaggle-dog_571091_93_1.svg](../_images/output_kaggle-dog_571091_93_1.svg)

## 14.14.7. Classifying the Testing Set and Submitting Results on Kaggle

Similar to the final step in Section 14.13, in the end
all the labeled data (including the validation set) are used for
training the model and classifying the testing set. We will use the
trained custom output network for classification.

```python
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

![../_images/output_kaggle-dog_571091_102_1.svg](../_images/output_kaggle-dog_571091_102_1.svg)

The above code will generate a `submission.csv` file to be submitted
to Kaggle in the same way described in Section 5.7.

## 14.14.8. Summary

- Images in the ImageNet dataset are larger (with varying dimensions)
  than CIFAR-10 images. We may modify image augmentation operations for
  tasks on a different dataset.
- To classify a subset of the ImageNet dataset, we can leverage
  pre-trained models on the full ImageNet dataset to extract features
  and only train a custom small-scale output network. This will lead to
  less computational time and memory cost.

## 14.14.9. Exercises

1. When using the full Kaggle competition dataset, what results can you
   achieve when you increase `batch_size` (batch size) and
   `num_epochs` (number of epochs) while setting some other
   hyperparameters as `lr = 0.01`, `lr_period = 10`, and
   `lr_decay = 0.1`?
2. Do you get better results if you use a deeper pretrained model? How
   do you tune hyperparameters? Can you further improve the results?
