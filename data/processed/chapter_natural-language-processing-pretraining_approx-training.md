---
source_url: https://d2l.ai/chapter_natural-language-processing-pretraining/approx-training.html
title: 15.2. Approximate Training
chapter: '15'
section_number: '15.2'
date: null
extractor: d2l
source_type: web
source: chapter_natural-language-processing-pretraining_approx-training
---

# 15.2. Approximate Training

Recall our discussions in Section 15.1. The main idea of the
skip-gram model is using softmax operations to calculate the conditional
probability of generating a context word $w_o$ based on the given
center word $w_c$ in (15.1.4), whose
corresponding logarithmic loss is given by the opposite of
(15.1.7).

Due to the nature of the softmax operation, since a context word may be
anyone in the dictionary $\mathcal{V}$, the opposite of
(15.1.7) contains the summation of items as many as
the entire size of the vocabulary. Consequently, the gradient
calculation for the skip-gram model in (15.1.8) and
that for the continuous bag-of-words model in
(15.1.15) both contain the summation. Unfortunately,
the computational cost for such gradients that sum over a large
dictionary (often with hundreds of thousands or millions of words) is
huge!

In order to reduce the aforementioned computational complexity, this
section will introduce two approximate training methods: *negative
sampling* and *hierarchical softmax*. Due to the similarity between the
skip-gram model and the continuous bag of words model, we will just take
the skip-gram model as an example to describe these two approximate
training methods.

## 15.2.1. Negative Sampling

Negative sampling modifies the original objective function. Given the
context window of a center word $w_c$, the fact that any (context)
word $w_o$ comes from this context window is considered as an
event with the probability modeled by

$$
(15.2.1)\[P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),
$$

where $\sigma$ uses the definition of the sigmoid activation
function:

$$
(15.2.2)\[\sigma(x) = \frac{1}{1+\exp(-x)}.
$$

Let’s begin by maximizing the joint probability of all such events in
text sequences to train word embeddings. Specifically, given a text
sequence of length $T$, denote by $w^{(t)}$ the word at time
step $t$ and let the context window size be $m$, consider
maximizing the joint probability

$$
(15.2.3)\[\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).
$$

However, (15.2.3) only considers those events
that involve positive examples. As a result, the joint probability in
(15.2.3) is maximized to 1 only if all the word
vectors are equal to infinity. Of course, such results are meaningless.
To make the objective function more meaningful, *negative sampling* adds
negative examples sampled from a predefined distribution.

Denote by $S$ the event that a context word $w_o$ comes from
the context window of a center word $w_c$. For this event
involving $w_o$, from a predefined distribution $P(w)$
sample $K$ *noise words* that are not from this context window.
Denote by $N_k$ the event that a noise word $w_k$
($k=1, \ldots, K$) does not come from the context window of
$w_c$. Assume that these events involving both the positive
example and negative examples $S, N_1, \ldots, N_K$ are mutually
independent. Negative sampling rewrites the joint probability (involving
only positive examples) in (15.2.3) as

$$
(15.2.4)\[\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),
$$

where the conditional probability is approximated through events
$S, N_1, \ldots, N_K$:

$$
(15.2.5)\[P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).
$$

Denote by $i_t$ and $h_k$ the indices of a word
$w^{(t)}$ at time step $t$ of a text sequence and a noise
word $w_k$, respectively. The logarithmic loss with respect to the
conditional probabilities in
(15.2.5) is

$$
(15.2.6)\[\begin{split}\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&- \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&- \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}\end{split}
$$

We can see that now the computational cost for gradients at each
training step has nothing to do with the dictionary size, but linearly
depends on $K$. When setting the hyperparameter $K$ to a
smaller value, the computational cost for gradients at each training
step with negative sampling is smaller.

## 15.2.2. Hierarchical Softmax

As an alternative approximate training method, *hierarchical softmax*
uses the binary tree, a data structure illustrated in
Fig. 15.2.1, where each leaf node of the tree represents
a word in dictionary $\mathcal{V}$.

Fig. 15.2.1 Hierarchical softmax for approximate training, where each leaf node
of the tree represents a word in the dictionary.

Denote by $L(w)$ the number of nodes (including both ends) on the
path from the root node to the leaf node representing word $w$ in
the binary tree. Let $n(w,j)$ be the $j^\textrm{th}$ node on
this path, with its context word vector being
$\mathbf{u}_{n(w, j)}$. For example, $L(w_3) = 4$ in
Fig. 15.2.1. Hierarchical softmax approximates the
conditional probability in (15.1.4) as

$$
(15.2.7)\[P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![ n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),
$$

where function $\sigma$ is defined in (15.2.2), and
$\textrm{leftChild}(n)$ is the left child node of node $n$:
if $x$ is true, $[\![x]\!] = 1$; otherwise
$[\![x]\!] = -1$.

To illustrate, let’s calculate the conditional probability of generating
word $w_3$ given word $w_c$ in Fig. 15.2.1.
This requires dot products between the word vector $\mathbf{v}_c$
of $w_c$ and non-leaf node vectors on the path (the path in bold
in Fig. 15.2.1) from the root to $w_3$, which is
traversed left, right, then left:

$$
(15.2.8)\[P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).
$$

Since $\sigma(x)+\sigma(-x) = 1$, it holds that the conditional
probabilities of generating all the words in dictionary
$\mathcal{V}$ based on any word $w_c$ sum up to one:

$$
(15.2.9)\[\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.
$$

Fortunately, since $L(w_o)-1$ is on the order of
$\mathcal{O}(\textrm{log}_2|\mathcal{V}|)$ due to the binary tree
structure, when the dictionary size $\mathcal{V}$ is huge, the
computational cost for each training step using hierarchical softmax is
significantly reduced compared with that without approximate training.

## 15.2.3. Summary

- Negative sampling constructs the loss function by considering
 mutually independent events that involve both positive and negative
 examples. The computational cost for training is linearly dependent
 on the number of noise words at each step.
- Hierarchical softmax constructs the loss function using the path from
 the root node to the leaf node in the binary tree. The computational
 cost for training is dependent on the logarithm of the dictionary
 size at each step.
