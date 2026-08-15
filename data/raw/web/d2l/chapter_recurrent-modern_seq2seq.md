---
source_url: "https://d2l.ai/chapter_recurrent-modern/seq2seq.html"
title: "10.7. Sequence-to-Sequence Learning for Machine Translation"
chapter: "10"
section_number: "10.7"
date: null
extractor: "d2l"
---

# 10.7. Sequence-to-Sequence Learning for Machine Translation

In so-called sequence-to-sequence problems such as machine translation
(as discussed in Section 10.5), where inputs and
outputs each consist of variable-length unaligned sequences, we
generally rely on encoder–decoder architectures
(Section 10.6). In this section, we will demonstrate
the application of an encoder–decoder architecture, where both the
encoder and decoder are implemented as RNNs, to the task of machine
translation
(Cho et al., 2014, Sutskever et al., 2014).

Here, the encoder RNN will take a variable-length sequence as input and
transform it into a fixed-shape hidden state. Later, in
Section 11, we will introduce attention
mechanisms, which allow us to access encoded inputs without having to
compress the entire input into a single fixed-length representation.

Then to generate the output sequence, one token at a time, the decoder
model, consisting of a separate RNN, will predict each successive target
token given both the input sequence and the preceding tokens in the
output. During training, the decoder will typically be conditioned upon
the preceding tokens in the official “ground truth” label. However, at
test time, we will want to condition each output of the decoder on the
tokens already predicted. Note that if we ignore the encoder, the
decoder in a sequence-to-sequence architecture behaves just like a
normal language model. Fig. 10.7.1 illustrates how to use
two RNNs for sequence-to-sequence learning in machine translation.

![../_images/seq2seq.svg](../_images/seq2seq.svg)

Fig. 10.7.1 Sequence-to-sequence learning with an RNN encoder and an RNN decoder.

In Fig. 10.7.1, the special “<eos>” token marks the end of
the sequence. Our model can stop making predictions once this token is
generated. At the initial time step of the RNN decoder, there are two
special design decisions to be aware of: First, we begin every input
with a special beginning-of-sequence “<bos>” token. Second, we may feed
the final hidden state of the encoder into the decoder at every single
decoding time step (Cho et al., 2014). In
some other designs, such as that of
Sutskever *et al.* (2014), the final hidden state of the RNN
encoder is used to initiate the hidden state of the decoder only at the
first decoding step.

```python
import collections
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

## 10.7.1. Teacher Forcing

While running the encoder on the input sequence is relatively
straightforward, handling the input and output of the decoder requires
more care. The most common approach is sometimes called *teacher
forcing*. Here, the original target sequence (token labels) is fed into
the decoder as input. More concretely, the special beginning-of-sequence
token and the original target sequence, excluding the final token, are
concatenated as input to the decoder, while the decoder output (labels
for training) is the original target sequence, shifted by one token:
“<bos>”, “Ils”, “regardent”, “.” $\rightarrow$ “Ils”, “regardent”,
“.”, “<eos>” (Fig. 10.7.1).

Our implementation in Section 10.5.3 prepared
training data for teacher forcing, where shifting tokens for
self-supervised learning is similar to the training of language models
in Section 9.3. An alternative approach is to feed
the *predicted* token from the previous time step as the current input
to the decoder.

In the following, we explain the design depicted in
Fig. 10.7.1 in greater detail. We will train this model for
machine translation on the English–French dataset as introduced in
Section 10.5.

## 10.7.2. Encoder

Recall that the encoder transforms an input sequence of variable length
into a fixed-shape *context variable* $\mathbf{c}$ (see
Fig. 10.7.1).

Consider a single sequence example (batch size 1). Suppose the input
sequence is $x_1, \ldots, x_T$, such that $x_t$ is the
$t^{\textrm{th}}$ token. At time step $t$, the RNN
transforms the input feature vector $\mathbf{x}_t$ for $x_t$
and the hidden state $\mathbf{h} _{t-1}$ from the previous time
step into the current hidden state $\mathbf{h}_t$. We can use a
function $f$ to express the transformation of the RNN’s recurrent
layer:

$$
(10.7.1)\[\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}).
$$

In general, the encoder transforms the hidden states at all time steps
into a context variable through a customized function $q$:

$$
(10.7.2)\[\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).
$$

For example, in Fig. 10.7.1, the context variable is just
the hidden state $\mathbf{h}_T$ corresponding to the encoder RNN’s
representation after processing the final token of the input sequence.

In this example, we have used a unidirectional RNN to design the
encoder, where the hidden state only depends on the input subsequence at
and before the time step of the hidden state. We can also construct
encoders using bidirectional RNNs. In this case, a hidden state depends
on the subsequence before and after the time step (including the input
at the current time step), which encodes the information of the entire
sequence.

Now let’s implement the RNN encoder. Note that we use an *embedding
layer* to obtain the feature vector for each token in the input
sequence. The weight of an embedding layer is a matrix, where the number
of rows corresponds to the size of the input vocabulary (`vocab_size`)
and number of columns corresponds to the feature vector’s dimension
(`embed_size`). For any input token index $i$, the embedding
layer fetches the $i^{\textrm{th}}$ row (starting from 0) of the
weight matrix to return its feature vector. Here we implement the
encoder with a multilayer GRU.

```python
def init_seq2seq(module):  #@save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class Seq2SeqEncoder(d2l.Encoder):  #@save
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X.t().type(torch.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

Let’s use a concrete example to illustrate the above encoder
implementation. Below, we instantiate a two-layer GRU encoder whose
number of hidden units is 16. Given a minibatch of sequence inputs `X`
(batch size $=4$; number of time steps $=9$), the hidden
states of the final layer at all the time steps (`enc_outputs`
returned by the encoder’s recurrent layers) are a tensor of shape
(number of time steps, batch size, number of hidden units).

```python
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = torch.zeros((batch_size, num_steps))
enc_outputs, enc_state = encoder(X)
d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
```

Since we are using a GRU here, the shape of the multilayer hidden states
at the final time step is (number of hidden layers, batch size, number
of hidden units).

```python
d2l.check_shape(enc_state, (num_layers, batch_size, num_hiddens))
```

## 10.7.3. Decoder

Given a target output sequence $y_1, y_2, \ldots, y_{T'}$ for each
time step $t'$ (we use $t^\prime$ to differentiate from the
input sequence time steps), the decoder assigns a predicted probability
to each possible token occurring at step $y_{t'+1}$ conditioned
upon the previous tokens in the target $y_1, \ldots, y_{t'}$ and
the context variable $\mathbf{c}$, i.e.,
$P(y_{t'+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$.

To predict the subsequent token $t^\prime+1$ in the target
sequence, the RNN decoder takes the previous step’s target token
$y_{t^\prime}$, the hidden RNN state from the previous time step
$\mathbf{s}_{t^\prime-1}$, and the context variable
$\mathbf{c}$ as its input, and transforms them into the hidden
state $\mathbf{s}_{t^\prime}$ at the current time step. We can use
a function $g$ to express the transformation of the decoder’s
hidden layer:

$$
(10.7.3)\[\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).
$$

After obtaining the hidden state of the decoder, we can use an output
layer and the softmax operation to compute the predictive distribution
$p(y_{t^{\prime}+1} \mid y_1, \ldots, y_{t^\prime}, \mathbf{c})$
over the subsequent output token ${t^\prime+1}$.

Following Fig. 10.7.1, when implementing the decoder as
follows, we directly use the hidden state at the final time step of the
encoder to initialize the hidden state of the decoder. This requires
that the RNN encoder and the RNN decoder have the same number of layers
and hidden units. To further incorporate the encoded input sequence
information, the context variable is concatenated with the decoder input
at all the time steps. To predict the probability distribution of the
output token, we use a fully connected layer to transform the hidden
state at the final layer of the RNN decoder.

```python
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size+num_hiddens, num_hiddens,
                           num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(X.t().type(torch.int32))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

To illustrate the implemented decoder, below we instantiate it with the
same hyperparameters from the aforementioned encoder. As we can see, the
output shape of the decoder becomes (batch size, number of time steps,
vocabulary size), where the final dimension of the tensor stores the
predicted token distribution.

```python
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
state = decoder.init_state(encoder(X))
dec_outputs, state = decoder(X, state)
d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))
```

The layers in the above RNN encoder–decoder model are summarized in
Fig. 10.7.2.

![../_images/seq2seq-details.svg](../_images/seq2seq-details.svg)

Fig. 10.7.2 Layers in an RNN encoder–decoder model.

## 10.7.4. Encoder–Decoder for Sequence-to-Sequence Learning

Putting it all together in code yields the following:

```python
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """The RNN encoder--decoder for sequence to sequence learning."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

## 10.7.5. Loss Function with Masking

At each time step, the decoder predicts a probability distribution for
the output tokens. As with language modeling, we can apply softmax to
obtain the distribution and calculate the cross-entropy loss for
optimization. Recall from Section 10.5 that the
special padding tokens are appended to the end of sequences and so
sequences of varying lengths can be efficiently loaded in minibatches of
the same shape. However, prediction of padding tokens should be excluded
from loss calculations. To this end, we can mask irrelevant entries with
zero values so that multiplication of any irrelevant prediction with
zero equates to zero.

```python
@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32)
    return (l * mask).sum() / mask.sum()
```

## 10.7.6. Training

Now we can create and train an RNN encoder–decoder model for
sequence-to-sequence learning on the machine translation dataset.

```python
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
encoder = Seq2SeqEncoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                lr=0.005)
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

![../_images/output_seq2seq_13725e_123_0.svg](../_images/output_seq2seq_13725e_123_0.svg)

## 10.7.7. Prediction

To predict the output sequence at each step, the predicted token from
the previous time step is fed into the decoder as an input. One simple
strategy is to sample whichever token that has been assigned by the
decoder the highest probability when predicting at each step. As in
training, at the initial time step the beginning-of-sequence (“<bos>”)
token is fed into the decoder. This prediction process is illustrated in
Fig. 10.7.3. When the end-of-sequence (“<eos>”)
token is predicted, the prediction of the output sequence is complete.

![../_images/seq2seq-predict.svg](../_images/seq2seq-predict.svg)

Fig. 10.7.3 Predicting the output sequence token by token using an RNN
encoder–decoder.

In the next section, we will introduce more sophisticated strategies
based on beam search (Section 10.8).

```python
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
    batch = [a.to(device) for a in batch]
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs = self.encoder(src, src_valid_len)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
    for _ in range(num_steps):
        Y, dec_state = self.decoder(outputs[-1], dec_state)
        outputs.append(Y.argmax(2))
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    return torch.cat(outputs[1:], 1), attention_weights
```

## 10.7.8. Evaluation of Predicted Sequences

We can evaluate a predicted sequence by comparing it with the target
sequence (the ground truth). But what precisely is the appropriate
measure for comparing similarity between two sequences?

Bilingual Evaluation Understudy (BLEU), though originally proposed for
evaluating machine translation results
(Papineni et al., 2002), has been extensively used in
measuring the quality of output sequences for different applications. In
principle, for any $n$-gram
(Section 9.3.1.1) in the predicted
sequence, BLEU evaluates whether this $n$-gram appears in the
target sequence.

Denote by $p_n$ the precision of an $n$-gram, defined as the
ratio of the number of matched $n$-grams in the predicted and
target sequences to the number of $n$-grams in the predicted
sequence. To explain, given a target sequence $A$, $B$,
$C$, $D$, $E$, $F$, and a predicted sequence
$A$, $B$, $B$, $C$, $D$, we have
$p_1 = 4/5$, $p_2 = 3/4$, $p_3 = 1/3$, and
$p_4 = 0$. Now let $\textrm{len}_{\textrm{label}}$ and
$\textrm{len}_{\textrm{pred}}$ be the numbers of tokens in the
target sequence and the predicted sequence, respectively. Then, BLEU is
defined as

$$
(10.7.4)\[\exp\left(\min\left(0, 1 - \frac{\textrm{len}_{\textrm{label}}}{\textrm{len}_{\textrm{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},
$$

where $k$ is the longest $n$-gram for matching.

Based on the definition of BLEU in (10.7.4), whenever the
predicted sequence is the same as the target sequence, BLEU is 1.
Moreover, since matching longer $n$-grams is more difficult, BLEU
assigns a greater weight when a longer $n$-gram has high
precision. Specifically, when $p_n$ is fixed, $p_n^{1/2^n}$
increases as $n$ grows (the original paper uses
$p_n^{1/n}$). Furthermore, since predicting shorter sequences
tends to yield a higher $p_n$ value, the coefficient before the
multiplication term in (10.7.4) penalizes shorter predicted
sequences. For example, when $k=2$, given the target sequence
$A$, $B$, $C$, $D$, $E$, $F$ and the
predicted sequence $A$, $B$, although $p_1 = p_2 = 1$,
the penalty factor $\exp(1-6/2) \approx 0.14$ lowers the BLEU.

We implement the BLEU measure as follows.

```python
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

In the end, we use the trained RNN encoder–decoder to translate a few
English sentences into French and compute the BLEU of the results.

```python
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{bleu(" ".join(translation), fr, k=2):.3f}')
```

## 10.7.9. Summary

Following the design of the encoder–decoder architecture, we can use two
RNNs to design a model for sequence-to-sequence learning. In
encoder–decoder training, the teacher forcing approach feeds original
output sequences (in contrast to predictions) into the decoder. When
implementing the encoder and the decoder, we can use multilayer RNNs. We
can use masks to filter out irrelevant computations, such as when
calculating the loss. For evaluating output sequences, BLEU is a popular
measure that matches $n$-grams between the predicted sequence and
the target sequence.

## 10.7.10. Exercises

1. Can you adjust the hyperparameters to improve the translation
   results?
2. Rerun the experiment without using masks in the loss calculation.
   What results do you observe? Why?
3. If the encoder and the decoder differ in the number of layers or the
   number of hidden units, how can we initialize the hidden state of the
   decoder?
4. In training, replace teacher forcing with feeding the prediction at
   the previous time step into the decoder. How does this influence the
   performance?
5. Rerun the experiment by replacing GRU with LSTM.
6. Are there any other ways to design the output layer of the decoder?
