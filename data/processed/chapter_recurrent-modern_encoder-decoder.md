---
source_url: https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html
title: 10.6. The Encoder–Decoder Architecture
chapter: '10'
section_number: '10.6'
date: null
extractor: d2l
source_type: web
source: chapter_recurrent-modern_encoder-decoder
---

# 10.6. The Encoder–Decoder Architecture

In general sequence-to-sequence problems like machine translation
(Section 10.5), inputs and outputs are of varying
lengths that are unaligned. The standard approach to handling this sort
of data is to design an *encoder–decoder* architecture
(Fig. 10.6.1) consisting of two major components: an
*encoder* that takes a variable-length sequence as input, and a
*decoder* that acts as a conditional language model, taking in the
encoded input and the leftwards context of the target sequence and
predicting the subsequent token in the target sequence.

Fig. 10.6.1 The encoder–decoder architecture.

Let’s take machine translation from English to French as an example.
Given an input sequence in English: “They”, “are”, “watching”, “.”, this
encoder–decoder architecture first encodes the variable-length input
into a state, then decodes the state to generate the translated
sequence, token by token, as output: “Ils”, “regardent”, “.”. Since the
encoder–decoder architecture forms the basis of different
sequence-to-sequence models in subsequent sections, this section will
convert this architecture into an interface that will be implemented
later.

```python
from torch import nn
from d2l import torch as d2l
```

## 10.6.1. Encoder

In the encoder interface, we just specify that the encoder takes
variable-length sequences as input `X`. The implementation will be
provided by any model that inherits this base `Encoder` class.

```python
class Encoder(nn.Module): #@save
 """The base encoder interface for the encoder--decoder architecture."""
 def __init__(self):
 super().__init__()

 # Later there can be additional arguments (e.g., length excluding padding)
 def forward(self, X, *args):
 raise NotImplementedError
```

## 10.6.2. Decoder

In the following decoder interface, we add an additional `init_state`
method to convert the encoder output (`enc_all_outputs`) into the
encoded state. Note that this step may require extra inputs, such as the
valid length of the input, which was explained in
Section 10.5. To generate a variable-length
sequence token by token, every time the decoder may map an input (e.g.,
the generated token at the previous time step) and the encoded state
into an output token at the current time step.

```python
class Decoder(nn.Module): #@save
 """The base decoder interface for the encoder--decoder architecture."""
 def __init__(self):
 super().__init__()

 # Later there can be additional arguments (e.g., length excluding padding)
 def init_state(self, enc_all_outputs, *args):
 raise NotImplementedError

 def forward(self, X, state):
 raise NotImplementedError
```

## 10.6.3. Putting the Encoder and Decoder Together

In the forward propagation, the output of the encoder is used to produce
the encoded state, and this state will be further used by the decoder as
one of its input.

```python
class EncoderDecoder(d2l.Classifier): #@save
 """The base class for the encoder--decoder architecture."""
 def __init__(self, encoder, decoder):
 super().__init__()
 self.encoder = encoder
 self.decoder = decoder

 def forward(self, enc_X, dec_X, *args):
 enc_all_outputs = self.encoder(enc_X, *args)
 dec_state = self.decoder.init_state(enc_all_outputs, *args)
 # Return decoder output only
 return self.decoder(dec_X, dec_state)[0]
```

In the next section, we will see how to apply RNNs to design
sequence-to-sequence models based on this encoder–decoder architecture.

## 10.6.4. Summary

Encoder-decoder architectures can handle inputs and outputs that both
consist of variable-length sequences and thus are suitable for
sequence-to-sequence problems such as machine translation. The encoder
takes a variable-length sequence as input and transforms it into a state
with a fixed shape. The decoder maps the encoded state of a fixed shape
to a variable-length sequence.
