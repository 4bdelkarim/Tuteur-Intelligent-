---
source_url: https://d2l.ai/chapter_preliminaries/pandas.html
title: 2.2. Data Preprocessing
chapter: '2'
section_number: '2.2'
date: null
extractor: d2l
source_type: web
source: chapter_preliminaries_pandas
---

# 2.2. Data Preprocessing

So far, we have been working with synthetic data that arrived in
ready-made tensors. However, to apply deep learning in the wild we must
extract messy data stored in arbitrary formats, and preprocess it to
suit our needs. Fortunately, the *pandas*
library can do much of the heavy
lifting. This section, while no substitute for a proper *pandas*
tutorial,
will give you a crash course on some of the most common routines.

## 2.2.1. Reading the Dataset

Comma-separated values (CSV) files are ubiquitous for the storing of
tabular (spreadsheet-like) data. In them, each line corresponds to one
record and consists of several (comma-separated) fields, e.g., “Albert
Einstein,March 14 1879,Ulm,Federal polytechnic school,field of
gravitational physics”. To demonstrate how to load CSV files with
`pandas`, we create a CSV file below `../data/house_tiny.csv`. This
file represents a dataset of homes, where each row corresponds to a
distinct home and the columns correspond to the number of rooms
(`NumRooms`), the roof type (`RoofType`), and the price (`Price`).

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
 f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Now let’s import `pandas` and load the dataset with `read_csv`.

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 2.2.2. Data Preparation

In supervised learning, we train models to predict a designated *target*
value, given some set of *input* values. Our first step in processing
the dataset is to separate out columns corresponding to input versus
target values. We can select columns either by name or via
integer-location based indexing (`iloc`).

You might have noticed that `pandas` replaced all CSV entries with
value `NA` with a special `NaN` (*not a number*) value. This can
also happen whenever an entry is empty, e.g., “3,,,270000”. These are
called *missing values* and they are the “bed bugs” of data science, a
persistent menace that you will confront throughout your career.
Depending upon the context, missing values might be handled either via
*imputation* or *deletion*. Imputation replaces missing values with
estimates of their values while deletion simply discards either those
rows or those columns that contain missing values.

Here are some common imputation heuristics. For categorical input
fields, we can treat `NaN` as a category. Since the `RoofType`
column takes values `Slate` and `NaN`, `pandas` can convert this
column into two columns `RoofType_Slate` and `RoofType_nan`. A row
whose roof type is `Slate` will set values of `RoofType_Slate` and
`RoofType_nan` to 1 and 0, respectively. The converse holds for a row
with a missing `RoofType` value.

```python
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

For missing numerical values, one common heuristic is to replace the
`NaN` entries with the mean value of the corresponding column.

```python
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## 2.2.3. Conversion to the Tensor Format

Now that all the entries in `inputs` and `targets` are numerical, we
can load them into a tensor (recall Section 2.1).

```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

## 2.2.4. Discussion

You now know how to partition data columns, impute missing variables,
and load `pandas` data into tensors. In Section 5.7,
you will pick up some more data processing skills. While this crash
course kept things simple, data processing can get hairy. For example,
rather than arriving in a single CSV file, our dataset might be spread
across multiple files extracted from a relational database. For
instance, in an e-commerce application, customer addresses might live in
one table and purchase data in another. Moreover, practitioners face
myriad data types beyond categorical and numeric, for example, text
strings, images, audio data, and point clouds. Oftentimes, advanced
tools and efficient algorithms are required in order to prevent data
processing from becoming the biggest bottleneck in the machine learning
pipeline. These problems will arise when we get to computer vision and
natural language processing. Finally, we must pay attention to data
quality. Real-world datasets are often plagued by outliers, faulty
measurements from sensors, and recording errors, which must be addressed
before feeding the data into any model. Data visualization tools such as
seaborn,
Bokeh, or
matplotlib can help you to manually
inspect the data and develop intuitions about the type of problems you
may need to address.
