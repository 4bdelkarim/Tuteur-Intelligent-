---
source_url: https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
title: 11.8. Transformers for Vision
chapter: '11'
section_number: '11.8'
date: null
extractor: d2l
source_type: web
---

# 11.8. Transformers for Vision

The Transformer architecture was initially proposed for
sequence-to-sequence learning, with a focus on machine translation.
Subsequently, Transformers emerged as the model of choice in various
natural language processing tasks
(Brown et al., 2020, Devlin et al., 2018, Radford et al., 2018, Radford et al., 2019, Raffel et al., 2020).
However, in the field of computer vision the dominant architecture has
remained the CNN (Section 8). Naturally, researchers
started to wonder if it might be possible to do better by adapting
Transformer models to image data. This question sparked immense interest
in the computer vision community. Recently,
Ramachandran *et al.* (2019) proposed a scheme for replacing
convolution with self-attention. However, its use of specialized
patterns in attention makes it hard to scale up models on hardware
accelerators. Then, Cordonnier *et al.* (2020) theoretically
proved that self-attention can learn to behave similarly to convolution.
Empirically, $2 \times 2$ patches were taken from images as
inputs, but the small patch size makes the model only applicable to
image data with low resolutions.

Without specific constraints on patch size, *vision Transformers* (ViTs)
extract patches from images and feed them into a Transformer encoder to
obtain a global representation, which will finally be transformed for
classification (Dosovitskiy et al., 2021). Notably,
Transformers show better scalability than CNNs: and when training larger
models on larger datasets, vision Transformers outperform ResNets by a
significant margin. Similar to the landscape of network architecture
design in natural language processing, Transformers have also become a
game-changer in computer vision.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 11.8.1. Model

Fig. 11.8.1 depicts the model architecture of vision
Transformers. This architecture consists of a stem that patchifies
images, a body based on the multilayer Transformer encoder, and a head
that transforms the global representation into the output label.

Fig. 11.8.1 The vision Transformer architecture. In this example, an image is
split into nine patches. A special “<cls>” token and the nine
flattened image patches are transformed via patch embedding and
$\mathit{n}$ Transformer encoder blocks into ten
representations, respectively. The “<cls>” representation is further
transformed into the output label.

Consider an input image with height $h$, width $w$, and
$c$ channels. Specifying the patch height and width both as
$p$, the image is split into a sequence of $m = hw/p^2$
patches, where each patch is flattened to a vector of length
$cp^2$. In this way, image patches can be treated similarly to
tokens in text sequences by Transformer encoders. A special “<cls>”
(class) token and the $m$ flattened image patches are linearly
projected into a sequence of $m+1$ vectors, summed with learnable
positional embeddings. The multilayer Transformer encoder transforms
$m+1$ input vectors into the same number of output vector
representations of the same length. It works exactly the same way as the
original Transformer encoder in Fig. 11.7.1, only
differing in the position of normalization. Since the “<cls>” token
attends to all the image patches via self-attention (see
Fig. 11.6.1), its representation from the
Transformer encoder output will be further transformed into the output
label.

## 11.8.2. Patch Embedding

To implement a vision Transformer, let’s start with patch embedding in
Fig. 11.8.1. Splitting an image into patches and linearly
projecting these flattened patches can be simplified as a single
convolution operation, where both the kernel size and the stride size
are set to the patch size.

```python
class PatchEmbedding(nn.Module):
 def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
 super().__init__()
 def _make_tuple(x):
 if not isinstance(x, (list, tuple)):
 return (x, x)
 return x
 img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
 self.num_patches = (img_size[0] // patch_size[0]) * (
 img_size[1] // patch_size[1])
 self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
 stride=patch_size)

 def forward(self, X):
 # Output shape: (batch size, no. of patches, no. of channels)
 return self.conv(X).flatten(2).transpose(1, 2)
```

In the following example, taking images with height and width of
`img_size` as inputs, the patch embedding outputs
`(img_size//patch_size)**2` patches that are linearly projected to
vectors of length `num_hiddens`.

```python
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
 (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## 11.8.3. Vision Transformer Encoder

The MLP of the vision Transformer encoder is slightly different from the
positionwise FFN of the original Transformer encoder (see
Section 11.7.2). First, here the activation
function uses the Gaussian error linear unit (GELU), which can be
considered as a smoother version of the ReLU
(Hendrycks and Gimpel, 2016). Second, dropout is applied to the
output of each fully connected layer in the MLP for regularization.

```python
class ViTMLP(nn.Module):
 def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
 super().__init__()
 self.dense1 = nn.LazyLinear(mlp_num_hiddens)
 self.gelu = nn.GELU()
 self.dropout1 = nn.Dropout(dropout)
 self.dense2 = nn.LazyLinear(mlp_num_outputs)
 self.dropout2 = nn.Dropout(dropout)

 def forward(self, x):
 return self.dropout2(self.dense2(self.dropout1(self.gelu(
 self.dense1(x)))))
```

The vision Transformer encoder block implementation just follows the
pre-normalization design in Fig. 11.8.1, where normalization is
applied right *before* multi-head attention or the MLP. In contrast to
post-normalization (“add & norm” in Fig. 11.7.1), where
normalization is placed right *after* residual connections,
pre-normalization leads to more effective or efficient training for
Transformers
(Baevski and Auli, 2018, Wang et al., 2019, Xiong et al., 2020).

```python
class ViTBlock(nn.Module):
 def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
 num_heads, dropout, use_bias=False):
 super().__init__()
 self.ln1 = nn.LayerNorm(norm_shape)
 self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
 dropout, use_bias)
 self.ln2 = nn.LayerNorm(norm_shape)
 self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

 def forward(self, X, valid_lens=None):
 X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
 return X + self.mlp(self.ln2(X))
```

Just as in Section 11.7.4, no vision Transformer
encoder block changes its input shape.

```python
X = torch.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

## 11.8.4. Putting It All Together

The forward pass of vision Transformers below is straightforward. First,
input images are fed into an `PatchEmbedding` instance, whose output
is concatenated with the “<cls>” token embedding. They are summed with
learnable positional embeddings before dropout. Then the output is fed
into the Transformer encoder that stacks `num_blks` instances of the
`ViTBlock` class. Finally, the representation of the “<cls>” token is
projected by the network head.

```python
class ViT(d2l.Classifier):
 """Vision Transformer."""
 def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
 use_bias=False, num_classes=10):
 super().__init__()
 self.save_hyperparameters()
 self.patch_embedding = PatchEmbedding(
 img_size, patch_size, num_hiddens)
 self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
 num_steps = self.patch_embedding.num_patches + 1 # Add the cls token
 # Positional embeddings are learnable
 self.pos_embedding = nn.Parameter(
 torch.randn(1, num_steps, num_hiddens))
 self.dropout = nn.Dropout(emb_dropout)
 self.blks = nn.Sequential()
 for i in range(num_blks):
 self.blks.add_module(f"{i}", ViTBlock(
 num_hiddens, num_hiddens, mlp_num_hiddens,
 num_heads, blk_dropout, use_bias))
 self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
 nn.Linear(num_hiddens, num_classes))

 def forward(self, X):
 X = self.patch_embedding(X)
 X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
 X = self.dropout(X + self.pos_embedding)
 for blk in self.blks:
 X = blk(X)
 return self.head(X[:, 0])
```

## 11.8.5. Training

Training a vision Transformer on the Fashion-MNIST dataset is just like
how CNNs were trained in Section 8.

```python
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
 num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## 11.8.6. Summary and Discussion

You may have noticed that for small datasets like Fashion-MNIST, our
implemented vision Transformer does not outperform the ResNet in
Section 8.6. Similar observations can be made even on the
ImageNet dataset (1.2 million images). This is because Transformers
*lack* those useful principles in convolution, such as translation
invariance and locality (Section 7.1). However, the picture
changes when training larger models on larger datasets (e.g., 300
million images), where vision Transformers outperform ResNets by a large
margin in image classification, demonstrating intrinsic superiority of
Transformers in scalability
(Dosovitskiy et al., 2021). The introduction of
vision Transformers has changed the landscape of network design for
modeling image data. They were soon shown to be effective on the
ImageNet dataset with data-efficient training strategies of DeiT
(Touvron et al., 2021). However, the quadratic complexity of
self-attention (Section 11.6)
makes the Transformer architecture less suitable for higher-resolution
images. Towards a general-purpose backbone network in computer vision,
Swin Transformers addressed the quadratic computational complexity with
respect to image size (Section 11.6.2) and
reinstated convolution-like priors, extending the applicability of
Transformers to a range of computer vision tasks beyond image
classification with state-of-the-art results (Liu et al., 2021).
