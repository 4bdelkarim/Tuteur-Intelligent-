---
source_url: https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html
title: '16.2. Sentiment Analysis: Using Recurrent Neural Networks'
chapter: '16'
section_number: '16.2'
date: null
extractor: d2l
source_type: web
---

# 16.2. Sentiment Analysis: Using Recurrent Neural Networks

Like word similarity and analogy tasks, we can also apply pretrained
word vectors to sentiment analysis. Since the IMDb review dataset in
Section 16.1 is not very big, using text representations
that were pretrained on large-scale corpora may reduce overfitting of
the model. As a specific example illustrated in
Fig. 16.2.1, we will represent each token using the
pretrained GloVe model, and feed these token representations into a
multilayer bidirectional RNN to obtain the text sequence representation,
which will be transformed into sentiment analysis outputs
(Maas et al., 2011). For the same downstream application,
we will consider a different architectural choice later.

Fig. 16.2.1 This section feeds pretrained GloVe to an RNN-based architecture for
sentiment analysis.

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## 16.2.1. Representing Single Text with RNNs

In text classifications tasks, such as sentiment analysis, a
varying-length text sequence will be transformed into fixed-length
categories. In the following `BiRNN` class, while each token of a text
sequence gets its individual pretrained GloVe representation via the
embedding layer (`self.embedding`), the entire sequence is encoded by
a bidirectional RNN (`self.encoder`). More concretely, the hidden
states (at the last layer) of the bidirectional LSTM at both the initial
and final time steps are concatenated as the representation of the text
sequence. This single text representation is then transformed into
output categories by a fully connected layer (`self.decoder`) with two
outputs (“positive” and “negative”).

```python
class BiRNN(nn.Module):
 def __init__(self, vocab_size, embed_size, num_hiddens,
 num_layers, **kwargs):
 super(BiRNN, self).__init__(**kwargs)
 self.embedding = nn.Embedding(vocab_size, embed_size)
 # Set `bidirectional` to True to get a bidirectional RNN
 self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
 bidirectional=True)
 self.decoder = nn.Linear(4 * num_hiddens, 2)

 def forward(self, inputs):
 # The shape of `inputs` is (batch size, no. of time steps). Because
 # LSTM requires its input's first dimension to be the temporal
 # dimension, the input is transposed before obtaining token
 # representations. The output shape is (no. of time steps, batch size,
 # word vector dimension)
 embeddings = self.embedding(inputs.T)
 self.encoder.flatten_parameters()
 # Returns hidden states of the last hidden layer at different time
 # steps. The shape of `outputs` is (no. of time steps, batch size,
 # 2 * no. of hidden units)
 outputs, _ = self.encoder(embeddings)
 # Concatenate the hidden states at the initial and final time steps as
 # the input of the fully connected layer. Its shape is (batch size,
 # 4 * no. of hidden units)
 encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
 outs = self.decoder(encoding)
 return outs
```

Let’s construct a bidirectional RNN with two hidden layers to represent
single text for sentiment analysis.

```python
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(module):
 if type(module) == nn.Linear:
 nn.init.xavier_uniform_(module.weight)
 if type(module) == nn.LSTM:
 for param in module._flat_weights_names:
 if "weight" in param:
 nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## 16.2.2. Loading Pretrained Word Vectors

Below we load the pretrained 100-dimensional (needs to be consistent
with `embed_size`) GloVe embeddings for tokens in the vocabulary.

```python
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

Print the shape of the vectors for all the tokens in the vocabulary.

```python
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

We use these pretrained word vectors to represent tokens in the reviews
and will not update these vectors during training.

```python
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## 16.2.3. Training and Evaluating the Model

Now we can train the bidirectional RNN for sentiment analysis.

```python
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

We define the following function to predict the sentiment of a text
sequence using the trained model `net`.

```python
#@save
def predict_sentiment(net, vocab, sequence):
 """Predict the sentiment of a text sequence."""
 sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
 label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
 return 'positive' if label == 1 else 'negative'
```

Finally, let’s use the trained model to predict the sentiment for two
simple sentences.

```python
predict_sentiment(net, vocab, 'this movie is so great')
```

```python
predict_sentiment(net, vocab, 'this movie is so bad')
```

## 16.2.4. Summary

- Pretrained word vectors can represent individual tokens in a text
 sequence.
- Bidirectional RNNs can represent a text sequence, such as via the
 concatenation of its hidden states at the initial and final time
 steps. This single text representation can be transformed into
 categories using a fully connected layer.
