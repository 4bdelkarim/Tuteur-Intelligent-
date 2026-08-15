---
source_url: https://d2l.ai/chapter_appendix-tools-for-deep-learning/d2l.html
title: 23.8. The d2l API Document
chapter: '23'
section_number: '23.8'
date: null
extractor: d2l
source_type: web
source: chapter_appendix-tools-for-deep-learning_d2l
---

# 23.8. The `d2l` API Document

This section displays classes and functions (sorted alphabetically) in
the `d2l` package, showing where they are defined in the book so you
can find more detailed implementations and explanations. See also the
source code on the GitHub
repository.

## 23.8.1. Classes

*class* d2l.torch.AdditiveAttention(*num\_hiddens*, *dropout*, *\*\*kwargs*)[source]
: Bases: `Module`

 Additive attention.

 Defined in Section 11.3.2.2

 forward(*queries*, *keys*, *values*, *valid\_lens*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.AddNorm(*norm\_shape*, *dropout*)[source]
: Bases: `Module`

 The residual connection followed by layer normalization.

 Defined in Section 11.7.2

 forward(*X*, *Y*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.AttentionDecoder[source]
: Bases: Decoder

 The base attention-based decoder interface.

 Defined in Section 11.4

 *property* attention\_weights

*class* d2l.torch.Classifier(*plot\_train\_per\_epoch=2*, *plot\_valid\_per\_epoch=1*)[source]
: Bases: Module

 The base class of classification models.

 Defined in Section 4.3

 accuracy(*Y\_hat*, *Y*, *averaged=True*)[source]
 : Compute the number of correct predictions.

 Defined in Section 4.3

 layer\_summary(*X\_shape*)[source]
 : Defined in Section 7.6

 loss(*Y\_hat*, *Y*, *averaged=True*)[source]
 : Defined in Section 4.5

 validation\_step(*batch*)[source]

*class* d2l.torch.DataModule(*root='../data'*, *num\_workers=4*)[source]
: Bases: HyperParameters

 The base class of data.

 Defined in Section 3.2.2

 get\_dataloader(*train*)[source]

 get\_tensorloader(*tensors*, *train*, *indices=slice(0, None, None)*)[source]
 : Defined in Section 3.3

 train\_dataloader()[source]

 val\_dataloader()[source]

*class* d2l.torch.Decoder[source]
: Bases: `Module`

 The base decoder interface for the encoder–decoder architecture.

 Defined in Section 10.6

 forward(*X*, *state*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

 init\_state(*enc\_all\_outputs*, *\*args*)[source]

*class* d2l.torch.DotProductAttention(*dropout*)[source]
: Bases: `Module`

 Scaled dot product attention.

 Defined in Section 11.3.2.2

 forward(*queries*, *keys*, *values*, *valid\_lens=None*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.Encoder[source]
: Bases: `Module`

 The base encoder interface for the encoder–decoder architecture.

 Defined in Section 10.6

 forward(*X*, *\*args*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.EncoderDecoder(*encoder*, *decoder*)[source]
: Bases: Classifier

 The base class for the encoder–decoder architecture.

 Defined in Section 10.6

 forward(*enc\_X*, *dec\_X*, *\*args*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

 predict\_step(*batch*, *device*, *num\_steps*, *save\_attention\_weights=False*)[source]
 : Defined in Section 10.7.6

*class* d2l.torch.FashionMNIST(*batch\_size=64*, *resize=(28, 28)*)[source]
: Bases: DataModule

 The Fashion-MNIST dataset.

 Defined in Section 4.2

 get\_dataloader(*train*)[source]
 : Defined in Section 4.2

 text\_labels(*indices*)[source]
 : Return text labels.

 Defined in Section 4.2

 visualize(*batch*, *nrows=1*, *ncols=8*, *labels=[]*)[source]
 : Defined in Section 4.2

*class* d2l.torch.GRU(*num\_inputs*, *num\_hiddens*, *num\_layers*, *dropout=0*)[source]
: Bases: RNN

 The multilayer GRU model.

 Defined in Section 10.3

*class* d2l.torch.HyperParameters[source]
: Bases: `object`

 The base class of hyperparameters.

 save\_hyperparameters(*ignore=[]*)[source]
 : Save function arguments into class attributes.

 Defined in Section 23.7

*class* d2l.torch.LeNet(*lr=0.1*, *num\_classes=10*)[source]
: Bases: Classifier

 The LeNet-5 model.

 Defined in Section 7.6

*class* d2l.torch.LinearRegression(*lr*)[source]
: Bases: Module

 The linear regression model implemented with high-level APIs.

 Defined in Section 3.5

 configure\_optimizers()[source]
 : Defined in Section 3.5

 forward(*X*)[source]
 : Defined in Section 3.5

 get\_w\_b()[source]
 : Defined in Section 3.5

 loss(*y\_hat*, *y*)[source]
 : Defined in Section 3.5

*class* d2l.torch.LinearRegressionScratch(*num\_inputs*, *lr*, *sigma=0.01*)[source]
: Bases: Module

 The linear regression model implemented from scratch.

 Defined in Section 3.4

 configure\_optimizers()[source]
 : Defined in Section 3.4

 forward(*X*)[source]
 : Defined in Section 3.4

 loss(*y\_hat*, *y*)[source]
 : Defined in Section 3.4

*class* d2l.torch.Module(*plot\_train\_per\_epoch=2*, *plot\_valid\_per\_epoch=1*)[source]
: Bases: `Module`, HyperParameters

 The base class of models.

 Defined in Section 3.2

 apply\_init(*inputs*, *init=None*)[source]
 : Defined in Section 6.4

 configure\_optimizers()[source]
 : Defined in Section 4.3

 forward(*X*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

 loss(*y\_hat*, *y*)[source]

 plot(*key*, *value*, *train*)[source]
 : Plot a point in animation.

 training\_step(*batch*)[source]

 validation\_step(*batch*)[source]

*class* d2l.torch.MTFraEng(*batch\_size*, *num\_steps=9*, *num\_train=512*, *num\_val=128*)[source]
: Bases: DataModule

 The English-French dataset.

 Defined in Section 10.5

 build(*src\_sentences*, *tgt\_sentences*)[source]
 : Defined in Section 10.5.3

 get\_dataloader(*train*)[source]
 : Defined in Section 10.5.3

*class* d2l.torch.MultiHeadAttention(*num\_hiddens*, *num\_heads*, *dropout*, *bias=False*, *\*\*kwargs*)[source]
: Bases: Module

 Multi-head attention.

 Defined in Section 11.5

 forward(*queries*, *keys*, *values*, *valid\_lens*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

 transpose\_output(*X*)[source]
 : Reverse the operation of transpose\_qkv.

 Defined in Section 11.5

 transpose\_qkv(*X*)[source]
 : Transposition for parallel computation of multiple attention heads.

 Defined in Section 11.5

*class* d2l.torch.PositionalEncoding(*num\_hiddens*, *dropout*, *max\_len=1000*)[source]
: Bases: `Module`

 Positional encoding.

 Defined in Section 11.6

 forward(*X*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.PositionWiseFFN(*ffn\_num\_hiddens*, *ffn\_num\_outputs*)[source]
: Bases: `Module`

 The positionwise feed-forward network.

 Defined in Section 11.7

 forward(*X*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.ProgressBoard(*xlabel=None*, *ylabel=None*, *xlim=None*, *ylim=None*, *xscale='linear'*, *yscale='linear'*, *ls=['-', '--', '-.', ':']*, *colors=['C0', 'C1', 'C2', 'C3']*, *fig=None*, *axes=None*, *figsize=(3.5, 2.5)*, *display=True*)[source]
: Bases: HyperParameters

 The board that plots data points in animation.

 Defined in Section 3.2

 draw(*x*, *y*, *label*, *every\_n=1*)[source]
 : Defined in Section 23.7

*class* d2l.torch.Residual(*num\_channels*, *use\_1x1conv=False*, *strides=1*)[source]
: Bases: `Module`

 The Residual block of ResNet models.

 Defined in Section 8.6

 forward(*X*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.ResNeXtBlock(*num\_channels*, *groups*, *bot\_mul*, *use\_1x1conv=False*, *strides=1*)[source]
: Bases: `Module`

 The ResNeXt block.

 Defined in Section 8.6.2

 forward(*X*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.RNN(*num\_inputs*, *num\_hiddens*)[source]
: Bases: Module

 The RNN model implemented with high-level APIs.

 Defined in Section 9.6

 forward(*inputs*, *H=None*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.RNNLM(*rnn*, *vocab\_size*, *lr=0.01*)[source]
: Bases: RNNLMScratch

 The RNN-based language model implemented with high-level APIs.

 Defined in Section 9.6

 init\_params()[source]

 output\_layer(*hiddens*)[source]
 : Defined in Section 9.5

*class* d2l.torch.RNNLMScratch(*rnn*, *vocab\_size*, *lr=0.01*)[source]
: Bases: Classifier

 The RNN-based language model implemented from scratch.

 Defined in Section 9.5

 forward(*X*, *state=None*)[source]
 : Defined in Section 9.5

 init\_params()[source]

 one\_hot(*X*)[source]
 : Defined in Section 9.5

 output\_layer(*rnn\_outputs*)[source]
 : Defined in Section 9.5

 predict(*prefix*, *num\_preds*, *vocab*, *device=None*)[source]
 : Defined in Section 9.5

 training\_step(*batch*)[source]

 validation\_step(*batch*)[source]

*class* d2l.torch.RNNScratch(*num\_inputs*, *num\_hiddens*, *sigma=0.01*)[source]
: Bases: Module

 The RNN model implemented from scratch.

 Defined in Section 9.5

 forward(*inputs*, *state=None*)[source]
 : Defined in Section 9.5

*class* d2l.torch.Seq2Seq(*encoder*, *decoder*, *tgt\_pad*, *lr*)[source]
: Bases: EncoderDecoder

 The RNN encoder–decoder for sequence to sequence learning.

 Defined in Section 10.7.3

 configure\_optimizers()[source]
 : Defined in Section 4.3

 validation\_step(*batch*)[source]

*class* d2l.torch.Seq2SeqEncoder(*vocab\_size*, *embed\_size*, *num\_hiddens*, *num\_layers*, *dropout=0*)[source]
: Bases: Encoder

 The RNN encoder for sequence-to-sequence learning.

 Defined in Section 10.7

 forward(*X*, *\*args*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.SGD(*params*, *lr*)[source]
: Bases: HyperParameters

 Minibatch stochastic gradient descent.

 Defined in Section 3.4

 step()[source]

 zero\_grad()[source]

*class* d2l.torch.SoftmaxRegression(*num\_outputs*, *lr*)[source]
: Bases: Classifier

 The softmax regression model.

 Defined in Section 4.5

 forward(*X*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.SyntheticRegressionData(*w*, *b*, *noise=0.01*, *num\_train=1000*, *num\_val=1000*, *batch\_size=32*)[source]
: Bases: DataModule

 Synthetic data for linear regression.

 Defined in Section 3.3

 get\_dataloader(*train*)[source]
 : Defined in Section 3.3

*class* d2l.torch.TimeMachine(*batch\_size*, *num\_steps*, *num\_train=10000*, *num\_val=5000*)[source]
: Bases: DataModule

 The Time Machine dataset.

 Defined in Section 9.2

 build(*raw\_text*, *vocab=None*)[source]
 : Defined in Section 9.2

 get\_dataloader(*train*)[source]
 : Defined in Section 9.3.3

*class* d2l.torch.Trainer(*max\_epochs*, *num\_gpus=0*, *gradient\_clip\_val=0*)[source]
: Bases: HyperParameters

 The base class for training models with data.

 Defined in Section 3.2.2

 clip\_gradients(*grad\_clip\_val*, *model*)[source]
 : Defined in Section 9.5

 fit(*model*, *data*)[source]

 fit\_epoch()[source]
 : Defined in Section 3.4

 prepare\_batch(*batch*)[source]
 : Defined in Section 6.7

 prepare\_data(*data*)[source]

 prepare\_model(*model*)[source]
 : Defined in Section 6.7

*class* d2l.torch.TransformerEncoder(*vocab\_size*, *num\_hiddens*, *ffn\_num\_hiddens*, *num\_heads*, *num\_blks*, *dropout*, *use\_bias=False*)[source]
: Bases: Encoder

 The Transformer encoder.

 Defined in Section 11.7.4

 forward(*X*, *valid\_lens*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.TransformerEncoderBlock(*num\_hiddens*, *ffn\_num\_hiddens*, *num\_heads*, *dropout*, *use\_bias=False*)[source]
: Bases: `Module`

 The Transformer encoder block.

 Defined in Section 11.7.2

 forward(*X*, *valid\_lens*)[source]
 : Defines the computation performed at every call.

 Should be overridden by all subclasses.

 Note

 Although the recipe for forward pass needs to be defined within
 this function, one should call the Module instance afterwards
 instead of this since the former takes care of running the
 registered hooks while the latter silently ignores them.

*class* d2l.torch.Vocab(*tokens=[]*, *min\_freq=0*, *reserved\_tokens=[]*)[source]
: Bases: `object`

 Vocabulary for text.

 to\_tokens(*indices*)[source]

 *property* unk

## 23.8.2. Functions

d2l.torch.add\_to\_class(*Class*)[source]
: Register functions as methods in created class.

 Defined in Section 3.2

d2l.torch.bleu(*pred\_seq*, *label\_seq*, *k*)[source]
: Compute the BLEU.

 Defined in Section 10.7.6

d2l.torch.check\_len(*a*, *n*)[source]
: Check the length of a list.

 Defined in Section 9.5

d2l.torch.check\_shape(*a*, *shape*)[source]
: Check the shape of a tensor.

 Defined in Section 9.5

d2l.torch.corr2d(*X*, *K*)[source]
: Compute 2D cross-correlation.

 Defined in Section 7.2

d2l.torch.cpu()[source]
: Get the CPU device.

 Defined in Section 6.7

d2l.torch.gpu(*i=0*)[source]
: Get a GPU device.

 Defined in Section 6.7

d2l.torch.init\_cnn(*module*)[source]
: Initialize weights for CNNs.

 Defined in Section 7.6

d2l.torch.init\_seq2seq(*module*)[source]
: Initialize weights for sequence-to-sequence learning.

 Defined in Section 10.7

d2l.torch.masked\_softmax(*X*, *valid\_lens*)[source]
: Perform softmax operation by masking elements on the last axis.

 Defined in Section 11.3

d2l.torch.num\_gpus()[source]
: Get the number of available GPUs.

 Defined in Section 6.7

d2l.torch.plot(*X*, *Y=None*, *xlabel=None*, *ylabel=None*, *legend=[]*, *xlim=None*, *ylim=None*, *xscale='linear'*, *yscale='linear'*, *fmts=('-', 'm--', 'g-.', 'r:')*, *figsize=(3.5, 2.5)*, *axes=None*)[source]
: Plot data points.

 Defined in Section 2.4

d2l.torch.set\_axes(*axes*, *xlabel*, *ylabel*, *xlim*, *ylim*, *xscale*, *yscale*, *legend*)[source]
: Set the axes for matplotlib.

 Defined in Section 2.4

d2l.torch.set\_figsize(*figsize=(3.5, 2.5)*)[source]
: Set the figure size for matplotlib.

 Defined in Section 2.4

d2l.torch.show\_heatmaps(*matrices*, *xlabel*, *ylabel*, *titles=None*, *figsize=(2.5, 2.5)*, *cmap='Reds'*)[source]
: Show heatmaps of matrices.

 Defined in Section 11.1

d2l.torch.show\_list\_len\_pair\_hist(*legend*, *xlabel*, *ylabel*, *xlist*, *ylist*)[source]
: Plot the histogram for list length pairs.

 Defined in Section 10.5

d2l.torch.try\_all\_gpus()[source]
: Return all available GPUs, or [cpu(),] if no GPU exists.

 Defined in Section 6.7

d2l.torch.try\_gpu(*i=0*)[source]
: Return gpu(i) if exists, otherwise return cpu().

 Defined in Section 6.7

d2l.torch.use\_svg\_display()[source]
: Use the svg format to display a plot in Jupyter.

 Defined in Section 2.4
