---
source_url: "https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html"
title: "16.3. Sentiment Analysis: Using Convolutional Neural Networks"
chapter: "16"
section_number: "16.3"
date: null
extractor: "d2l"
---

# 16.3. Sentiment Analysis: Using Convolutional Neural Networks

In Section 7, we investigated mechanisms for processing
two-dimensional image data with two-dimensional CNNs, which were applied
to local features such as adjacent pixels. Though originally designed
for computer vision, CNNs are also widely used for natural language
processing. Simply put, just think of any text sequence as a
one-dimensional image. In this way, one-dimensional CNNs can process
local features such as $n$-grams in text.

In this section, we will use the *textCNN* model to demonstrate how to
design a CNN architecture for representing single text
(Kim, 2014). Compared with Fig. 16.2.1 that
uses an RNN architecture with GloVe pretraining for sentiment analysis,
the only difference in Fig. 16.3.1 lies in the choice
of the architecture.

![../_images/nlp-map-sa-cnn.svg](../_images/nlp-map-sa-cnn.svg)

Fig. 16.3.1 This section feeds pretrained GloVe to a CNN-based architecture for
sentiment analysis.

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## 16.3.1. One-Dimensional Convolutions

Before introducing the model, let’s see how a one-dimensional
convolution works. Bear in mind that it is just a special case of a
two-dimensional convolution based on the cross-correlation operation.

![../_images/conv1d.svg](../_images/conv1d.svg)

Fig. 16.3.2 One-dimensional cross-correlation operation. The shaded portions are
the first output element as well as the input and kernel tensor
elements used for the output computation:
$0\times1+1\times2=2$.

As shown in Fig. 16.3.2, in the one-dimensional case, the
convolution window slides from left to right across the input tensor.
During sliding, the input subtensor (e.g., $0$ and $1$ in
Fig. 16.3.2) contained in the convolution window at a certain
position and the kernel tensor (e.g., $1$ and $2$ in
Fig. 16.3.2) are multiplied elementwise. The sum of these
multiplications gives the single scalar value (e.g.,
$0\times1+1\times2=2$ in Fig. 16.3.2) at the
corresponding position of the output tensor.

We implement one-dimensional cross-correlation in the following
`corr1d` function. Given an input tensor `X` and a kernel tensor
`K`, it returns the output tensor `Y`.

```python
def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

We can construct the input tensor `X` and the kernel tensor `K` from
Fig. 16.3.2 to validate the output of the above
one-dimensional cross-correlation implementation.

```python
X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
corr1d(X, K)
```

For any one-dimensional input with multiple channels, the convolution
kernel needs to have the same number of input channels. Then for each
channel, perform a cross-correlation operation on the one-dimensional
tensor of the input and the one-dimensional tensor of the convolution
kernel, summing the results over all the channels to produce the
one-dimensional output tensor. Fig. 16.3.3 shows a
one-dimensional cross-correlation operation with 3 input channels.

![../_images/conv1d-channel.svg](../_images/conv1d-channel.svg)

Fig. 16.3.3 One-dimensional cross-correlation operation with 3 input channels.
The shaded portions are the first output element as well as the input
and kernel tensor elements used for the output computation:
$0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.

We can implement the one-dimensional cross-correlation operation for
multiple input channels and validate the results in
Fig. 16.3.3.

```python
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

Note that multi-input-channel one-dimensional cross-correlations are
equivalent to single-input-channel two-dimensional cross-correlations.
To illustrate, an equivalent form of the multi-input-channel
one-dimensional cross-correlation in Fig. 16.3.3 is
the single-input-channel two-dimensional cross-correlation in
Fig. 16.3.4, where the height of the convolution kernel
has to be the same as that of the input tensor.

![../_images/conv1d-2d.svg](../_images/conv1d-2d.svg)

Fig. 16.3.4 Two-dimensional cross-correlation operation with a single input
channel. The shaded portions are the first output element as well as
the input and kernel tensor elements used for the output computation:
$2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.

Both the outputs in Fig. 16.3.2 and
Fig. 16.3.3 have only one channel. Same as
two-dimensional convolutions with multiple output channels described in
Section 7.4.2, we can also specify multiple
output channels for one-dimensional convolutions.

## 16.3.2. Max-Over-Time Pooling

Similarly, we can use pooling to extract the highest value from sequence
representations as the most important feature across time steps. The
*max-over-time pooling* used in textCNN works like the one-dimensional
global max-pooling (Collobert et al., 2011). For a
multi-channel input where each channel stores values at different time
steps, the output at each channel is the maximum value for that channel.
Note that the max-over-time pooling allows different numbers of time
steps at different channels.

## 16.3.3. The textCNN Model

Using the one-dimensional convolution and max-over-time pooling, the
textCNN model takes individual pretrained token representations as
input, then obtains and transforms sequence representations for the
downstream application.

For a single text sequence with $n$ tokens represented by
$d$-dimensional vectors, the width, height, and number of channels
of the input tensor are $n$, $1$, and $d$,
respectively. The textCNN model transforms the input into the output as
follows:

1. Define multiple one-dimensional convolution kernels and perform
   convolution operations separately on the inputs. Convolution kernels
   with different widths may capture local features among different
   numbers of adjacent tokens.
2. Perform max-over-time pooling on all the output channels, and then
   concatenate all the scalar pooling outputs as a vector.
3. Transform the concatenated vector into the output categories using
   the fully connected layer. Dropout can be used for reducing
   overfitting.

![../_images/textcnn.svg](../_images/textcnn.svg)

Fig. 16.3.5 The model architecture of textCNN.

Fig. 16.3.5 illustrates the model architecture of
textCNN with a concrete example. The input is a sentence with 11 tokens,
where each token is represented by a 6-dimensional vectors. So we have a
6-channel input with width 11. Define two one-dimensional convolution
kernels of widths 2 and 4, with 4 and 5 output channels, respectively.
They produce 4 output channels with width $11-2+1=10$ and 5 output
channels with width $11-4+1=8$. Despite different widths of these
9 channels, the max-over-time pooling gives a concatenated 9-dimensional
vector, which is finally transformed into a 2-dimensional output vector
for binary sentiment predictions.

### 16.3.3.1. Defining the Model

We implement the textCNN model in the following class. Compared with the
bidirectional RNN model in Section 16.2, besides
replacing recurrent layers with convolutional layers, we also use two
embedding layers: one with trainable weights and the other with fixed
weights.

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

Let’s create a textCNN instance. It has 3 convolutional layers with
kernel widths of 3, 4, and 5, all with 100 output channels.

```python
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)

net.apply(init_weights);
```

### 16.3.3.2. Loading Pretrained Word Vectors

Same as Section 16.2, we load pretrained
100-dimensional GloVe embeddings as the initialized token
representations. These token representations (embedding weights) will be
trained in `embedding` and fixed in `constant_embedding`.

```python
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### 16.3.3.3. Training and Evaluating the Model

Now we can train the textCNN model for sentiment analysis.

```python
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

![../_images/output_sentiment-analysis-cnn_900d1d_66_1.svg](../_images/output_sentiment-analysis-cnn_900d1d_66_1.svg)

Below we use the trained model to predict the sentiment for two simple
sentences.

```python
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```python
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## 16.3.4. Summary

- One-dimensional CNNs can process local features such as
  $n$-grams in text.
- Multi-input-channel one-dimensional cross-correlations are equivalent
  to single-input-channel two-dimensional cross-correlations.
- The max-over-time pooling allows different numbers of time steps at
  different channels.
- The textCNN model transforms individual token representations into
  downstream application outputs using one-dimensional convolutional
  layers and max-over-time pooling layers.

## 16.3.5. Exercises

1. Tune hyperparameters and compare the two architectures for sentiment
   analysis in Section 16.2 and in this section, such
   as in classification accuracy and computational efficiency.
2. Can you further improve the classification accuracy of the model by
   using the methods introduced in the exercises of
   Section 16.2?
3. Add positional encoding in the input representations. Does it improve
   the classification accuracy?
