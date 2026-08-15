---
source_url: https://d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-attention.html
title: '16.5. Natural Language Inference: Using Attention'
chapter: '16'
section_number: '16.5'
date: null
extractor: d2l
source_type: web
source: chapter_natural-language-processing-applications_natural-language-inference-attention
---

# 16.5. Natural Language Inference: Using Attention

We introduced the natural language inference task and the SNLI dataset
in Section 16.4. In view of
many models that are based on complex and deep architectures,
Parikh *et al.* (2016) proposed to address natural
language inference with attention mechanisms and called it a
“decomposable attention model”. This results in a model without
recurrent or convolutional layers, achieving the best result at the time
on the SNLI dataset with much fewer parameters. In this section, we will
describe and implement this attention-based method (with MLPs) for
natural language inference, as depicted in
Fig. 16.5.1.

Fig. 16.5.1 This section feeds pretrained GloVe to an architecture based on
attention and MLPs for natural language inference.

## 16.5.1. The Model

Simpler than preserving the order of tokens in premises and hypotheses,
we can just align tokens in one text sequence to every token in the
other, and vice versa, then compare and aggregate such information to
predict the logical relationships between premises and hypotheses.
Similar to alignment of tokens between source and target sentences in
machine translation, the alignment of tokens between premises and
hypotheses can be neatly accomplished by attention mechanisms.

Fig. 16.5.2 Natural language inference using attention mechanisms.

Fig. 16.5.2 depicts the natural language inference
method using attention mechanisms. At a high level, it consists of three
jointly trained steps: attending, comparing, and aggregating. We will
illustrate them step by step in the following.

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

### 16.5.1.1. Attending

The first step is to align tokens in one text sequence to each token in
the other sequence. Suppose that the premise is “i do need sleep” and
the hypothesis is “i am tired”. Due to semantical similarity, we may
wish to align “i” in the hypothesis with “i” in the premise, and align
“tired” in the hypothesis with “sleep” in the premise. Likewise, we may
wish to align “i” in the premise with “i” in the hypothesis, and align
“need” and “sleep” in the premise with “tired” in the hypothesis. Note
that such alignment is *soft* using weighted average, where ideally
large weights are associated with the tokens to be aligned. For ease of
demonstration, Fig. 16.5.2 shows such alignment in a
*hard* way.

Now we describe the soft alignment using attention mechanisms in more
detail. Denote by
$\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$ and
$\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ the premise
and hypothesis, whose number of tokens are $m$ and $n$,
respectively, where
$\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$
($i = 1, \ldots, m, j = 1, \ldots, n$) is a $d$-dimensional
word vector. For soft alignment, we compute the attention weights
$e_{ij} \in \mathbb{R}$ as

$$
(16.5.1)\[e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),
$$

where the function $f$ is an MLP defined in the following `mlp`
function. The output dimension of $f$ is specified by the
`num_hiddens` argument of `mlp`.

```python
def mlp(num_inputs, num_hiddens, flatten):
 net = []
 net.append(nn.Dropout(0.2))
 net.append(nn.Linear(num_inputs, num_hiddens))
 net.append(nn.ReLU())
 if flatten:
 net.append(nn.Flatten(start_dim=1))
 net.append(nn.Dropout(0.2))
 net.append(nn.Linear(num_hiddens, num_hiddens))
 net.append(nn.ReLU())
 if flatten:
 net.append(nn.Flatten(start_dim=1))
 return nn.Sequential(*net)
```

It should be highlighted that, in (16.5.1) $f$ takes
inputs $\mathbf{a}_i$ and $\mathbf{b}_j$ separately rather
than takes a pair of them together as input. This *decomposition* trick
leads to only $m + n$ applications (linear complexity) of
$f$ rather than $mn$ applications (quadratic complexity).

Normalizing the attention weights in (16.5.1), we compute the
weighted average of all the token vectors in the hypothesis to obtain
representation of the hypothesis that is softly aligned with the token
indexed by $i$ in the premise:

$$
(16.5.2)\[\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Likewise, we compute soft alignment of premise tokens for each token
indexed by $j$ in the hypothesis:

$$
(16.5.3)\[\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Below we define the `Attend` class to compute the soft alignment of
hypotheses (`beta`) with input premises `A` and soft alignment of
premises (`alpha`) with input hypotheses `B`.

```python
class Attend(nn.Module):
 def __init__(self, num_inputs, num_hiddens, **kwargs):
 super(Attend, self).__init__(**kwargs)
 self.f = mlp(num_inputs, num_hiddens, flatten=False)

 def forward(self, A, B):
 # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
 # `embed_size`)
 # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
 # `num_hiddens`)
 f_A = self.f(A)
 f_B = self.f(B)
 # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
 # no. of tokens in sequence B)
 e = torch.bmm(f_A, f_B.permute(0, 2, 1))
 # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
 # `embed_size`), where sequence B is softly aligned with each token
 # (axis 1 of `beta`) in sequence A
 beta = torch.bmm(F.softmax(e, dim=-1), B)
 # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
 # `embed_size`), where sequence A is softly aligned with each token
 # (axis 1 of `alpha`) in sequence B
 alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
 return beta, alpha
```

### 16.5.1.2. Comparing

In the next step, we compare a token in one sequence with the other
sequence that is softly aligned with that token. Note that in soft
alignment, all the tokens from one sequence, though with probably
different attention weights, will be compared with a token in the other
sequence. For easy of demonstration, Fig. 16.5.2 pairs
tokens with aligned tokens in a *hard* way. For example, suppose that
the attending step determines that “need” and “sleep” in the premise are
both aligned with “tired” in the hypothesis, the pair “tired–need sleep”
will be compared.

In the comparing step, we feed the concatenation (operator
$[\cdot, \cdot]$) of tokens from one sequence and aligned tokens
from the other sequence into a function $g$ (an MLP):

$$
(16.5.4)\[\begin{split}\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.\end{split}
$$

In (16.5.4), $\mathbf{v}_{A,i}$ is the comparison
between token $i$ in the premise and all the hypothesis tokens
that are softly aligned with token $i$; while
$\mathbf{v}_{B,j}$ is the comparison between token $j$ in
the hypothesis and all the premise tokens that are softly aligned with
token $j$. The following `Compare` class defines such as
comparing step.

```python
class Compare(nn.Module):
 def __init__(self, num_inputs, num_hiddens, **kwargs):
 super(Compare, self).__init__(**kwargs)
 self.g = mlp(num_inputs, num_hiddens, flatten=False)

 def forward(self, A, B, beta, alpha):
 V_A = self.g(torch.cat([A, beta], dim=2))
 V_B = self.g(torch.cat([B, alpha], dim=2))
 return V_A, V_B
```

### 16.5.1.3. Aggregating

With two sets of comparison vectors $\mathbf{v}_{A,i}$
($i = 1, \ldots, m$) and $\mathbf{v}_{B,j}$
($j = 1, \ldots, n$) on hand, in the last step we will aggregate
such information to infer the logical relationship. We begin by summing
up both sets:

$$
(16.5.5)\[\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Next we feed the concatenation of both summarization results into
function $h$ (an MLP) to obtain the classification result of the
logical relationship:

$$
(16.5.6)\[\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

The aggregation step is defined in the following `Aggregate` class.

```python
class Aggregate(nn.Module):
 def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
 super(Aggregate, self).__init__(**kwargs)
 self.h = mlp(num_inputs, num_hiddens, flatten=True)
 self.linear = nn.Linear(num_hiddens, num_outputs)

 def forward(self, V_A, V_B):
 # Sum up both sets of comparison vectors
 V_A = V_A.sum(dim=1)
 V_B = V_B.sum(dim=1)
 # Feed the concatenation of both summarization results into an MLP
 Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
 return Y_hat
```

### 16.5.1.4. Putting It All Together

By putting the attending, comparing, and aggregating steps together, we
define the decomposable attention model to jointly train these three
steps.

```python
class DecomposableAttention(nn.Module):
 def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
 super(DecomposableAttention, self).__init__(**kwargs)
 self.embedding = nn.Embedding(len(vocab), embed_size)
 self.attend = Attend(num_inputs_attend, num_hiddens)
 self.compare = Compare(num_inputs_compare, num_hiddens)
 # There are 3 possible outputs: entailment, contradiction, and neutral
 self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

 def forward(self, X):
 premises, hypotheses = X
 A = self.embedding(premises)
 B = self.embedding(hypotheses)
 beta, alpha = self.attend(A, B)
 V_A, V_B = self.compare(A, B, beta, alpha)
 Y_hat = self.aggregate(V_A, V_B)
 return Y_hat
```

## 16.5.2. Training and Evaluating the Model

Now we will train and evaluate the defined decomposable attention model
on the SNLI dataset. We begin by reading the dataset.

### 16.5.2.1. Reading the dataset

We download and read the SNLI dataset using the function defined in
Section 16.4. The batch size
and sequence length are set to $256$ and $50$, respectively.

```python
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### 16.5.2.2. Creating the Model

We use the pretrained 100-dimensional GloVe embedding to represent the
input tokens. Thus, we predefine the dimension of vectors
$\mathbf{a}_i$ and $\mathbf{b}_j$ in (16.5.1) as
100. The output dimension of functions $f$ in (16.5.1)
and $g$ in (16.5.4) is set to 200. Then we create a
model instance, initialize its parameters, and load the GloVe embedding
to initialize vectors of input tokens.

```python
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### 16.5.2.3. Training and Evaluating the Model

In contrast to the `split_batch` function in Section 13.5
that takes single inputs such as text sequences (or images), we define a
`split_batch_multi_inputs` function to take multiple inputs such as
premises and hypotheses in minibatches.

Now we can train and evaluate the model on the SNLI dataset.

```python
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### 16.5.2.4. Using the Model

Finally, define the prediction function to output the logical
relationship between a pair of premise and hypothesis.

```python
#@save
def predict_snli(net, vocab, premise, hypothesis):
 """Predict the logical relationship between the premise and hypothesis."""
 net.eval()
 premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
 hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
 label = torch.argmax(net([premise.reshape((1, -1)),
 hypothesis.reshape((1, -1))]), dim=1)
 return 'entailment' if label == 0 else 'contradiction' if label == 1 \
 else 'neutral'
```

We can use the trained model to obtain the natural language inference
result for a sample pair of sentences.

```python
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## 16.5.3. Summary

- The decomposable attention model consists of three steps for
 predicting the logical relationships between premises and hypotheses:
 attending, comparing, and aggregating.
- With attention mechanisms, we can align tokens in one text sequence
 to every token in the other, and vice versa. Such alignment is soft
 using weighted average, where ideally large weights are associated
 with the tokens to be aligned.
- The decomposition trick leads to a more desirable linear complexity
 than quadratic complexity when computing attention weights.
- We can use pretrained word vectors as the input representation for
 downstream natural language processing task such as natural language
 inference.
