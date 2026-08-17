---
source_url: https://d2l.ai/chapter_recurrent-modern/beam-search.html
title: 10.8. Beam Search
chapter: '10'
section_number: '10.8'
date: null
extractor: d2l
source_type: web
---

# 10.8. Beam Search

In Section 10.7, we introduced the encoder–decoder
architecture, and the standard techniques for training them end-to-end.
However, when it came to test-time prediction, we mentioned only the
*greedy* strategy, where we select at each time step the token given the
highest predicted probability of coming next, until, at some time step,
we find that we have predicted the special end-of-sequence “<eos>”
token. In this section, we will begin by formalizing this *greedy
search* strategy and identifying some problems that practitioners tend
to run into. Subsequently, we compare this strategy with two
alternatives: *exhaustive search* (illustrative but not practical) and
*beam search* (the standard method in practice).

Let’s begin by setting up our mathematical notation, borrowing
conventions from Section 10.7. At any time step $t'$,
the decoder outputs predictions representing the probability of each
token in the vocabulary coming next in the sequence (the likely value of
$y_{t'+1}$), conditioned on the previous tokens
$y_1, \ldots, y_{t'}$ and the context variable $\mathbf{c}$,
produced by the encoder to represent the input sequence. To quantify
computational cost, denote by $\mathcal{Y}$ the output vocabulary
(including the special end-of-sequence token “<eos>”). Let’s also
specify the maximum number of tokens of an output sequence as
$T'$. Our goal is to search for an ideal output from all
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ possible output
sequences. Note that this slightly overestimates the number of distinct
outputs because there are no subsequent tokens once the “<eos>” token
occurs. However, for our purposes, this number roughly captures the size
of the search space.

## 10.8.1. Greedy Search

Consider the simple *greedy search* strategy from
Section 10.7. Here, at any time step $t'$, we simply
select the token with the highest conditional probability from
$\mathcal{Y}$, i.e.,

$$
(10.8.1)\[y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).
$$

Once our model outputs “<eos>” (or we reach the maximum length
$T'$) the output sequence is completed.

This strategy might look reasonable, and in fact it is not so bad!
Considering how computationally undemanding it is, you’d be hard pressed
to get more bang for your buck. However, if we put aside efficiency for
a minute, it might seem more reasonable to search for the *most likely
sequence*, not the sequence of (greedily selected) *most likely tokens*.
It turns out that these two objects can be quite different. The most
likely sequence is the one that maximizes the expression
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$.
In our machine translation example, if the decoder truly recovered the
probabilities of the underlying generative process, then this would give
us the most likely translation. Unfortunately, there is no guarantee
that greedy search will give us this sequence.

Let’s illustrate it with an example. Suppose that there are four tokens
“A”, “B”, “C”, and “<eos>” in the output dictionary. In
Fig. 10.8.1, the four numbers under each time step
represent the conditional probabilities of generating “A”, “B”, “C”, and
“<eos>” respectively, at that time step.

Fig. 10.8.1 At each time step, greedy search selects the token with the highest
conditional probability.

At each time step, greedy search selects the token with the highest
conditional probability. Therefore, the output sequence “A”, “B”, “C”,
and “<eos>” will be predicted (Fig. 10.8.1). The
conditional probability of this output sequence is
$0.5\times0.4\times0.4\times0.6 = 0.048$.

Next, let’s look at another example in Fig. 10.8.2. Unlike
in Fig. 10.8.1, at time step 2 we select the token “C”,
which has the *second* highest conditional probability.

Fig. 10.8.2 The four numbers under each time step represent the conditional
probabilities of generating “A”, “B”, “C”, and “<eos>” at that time
step. At time step 2, the token “C”, which has the second highest
conditional probability, is selected.

Since the output subsequences at time steps 1 and 2, on which time step
3 is based, have changed from “A” and “B” in Fig. 10.8.1
to “A” and “C” in Fig. 10.8.2, the conditional probability
of each token at time step 3 has also changed in
Fig. 10.8.2. Suppose that we choose the token “B” at time
step 3. Now time step 4 is conditional on the output subsequence at the
first three time steps “A”, “C”, and “B”, which has changed from “A”,
“B”, and “C” in Fig. 10.8.1. Therefore, the conditional
probability of generating each token at time step 4 in
Fig. 10.8.2 is also different from that in
Fig. 10.8.1. As a result, the conditional probability of
the output sequence “A”, “C”, “B”, and “<eos>” in
Fig. 10.8.2 is
$0.5\times0.3 \times0.6\times0.6=0.054$, which is greater than
that of greedy search in Fig. 10.8.1. In this example, the
output sequence “A”, “B”, “C”, and “<eos>” obtained by the greedy search
is not optimal.

## 10.8.2. Exhaustive Search

If the goal is to obtain the most likely sequence, we may consider using
*exhaustive search*: enumerate all the possible output sequences with
their conditional probabilities, and then output the one that scores the
highest predicted probability.

While this would certainly give us what we desire, it would come at a
prohibitive computational cost of
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$, exponential in the
sequence length and with an enormous base given by the vocabulary size.
For example, when $|\mathcal{Y}|=10000$ and $T'=10$, both
small numbers when compared with ones in real applications, we will need
to evaluate $10000^{10} = 10^{40}$ sequences, which is already
beyond the capabilities of any foreseeable computers. On the other hand,
the computational cost of greedy search is
$\mathcal{O}(\left|\mathcal{Y}\right|T')$: miraculously cheap but
far from optimal. For example, when $|\mathcal{Y}|=10000$ and
$T'=10$, we only need to evaluate $10000\times10=10^5$
sequences.

## 10.8.3. Beam Search

You could view sequence decoding strategies as lying on a spectrum, with
*beam search* striking a compromise between the efficiency of greedy
search and the optimality of exhaustive search. The most straightforward
version of beam search is characterized by a single hyperparameter, the
*beam size*, $k$. Let’s explain this terminology. At time step 1,
we select the $k$ tokens with the highest predicted probabilities.
Each of them will be the first token of $k$ candidate output
sequences, respectively. At each subsequent time step, based on the
$k$ candidate output sequences at the previous time step, we
continue to select $k$ candidate output sequences with the highest
predicted probabilities from $k\left|\mathcal{Y}\right|$ possible
choices.

Fig. 10.8.3 The process of beam search (beam size $=2$; maximum length of
an output sequence $=3$). The candidate output sequences are
$\mathit{A}$, $\mathit{C}$, $\mathit{AB}$,
$\mathit{CE}$, $\mathit{ABD}$, and $\mathit{CED}$.

Fig. 10.8.3 demonstrates the process of beam search with
an example. Suppose that the output vocabulary contains only five
elements: $\mathcal{Y} = \{A, B, C, D, E\}$, where one of them is
“<eos>”. Let the beam size be two and the maximum length of an output
sequence be three. At time step 1, suppose that the tokens with the
highest conditional probabilities $P(y_1 \mid \mathbf{c})$ are
$A$ and $C$. At time step 2, for all
$y_2 \in \mathcal{Y},$ we compute

$$
(10.8.2)\[\begin{split}\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}\end{split}
$$

and pick the largest two among these ten values, say
$P(A, B \mid \mathbf{c})$ and $P(C, E \mid \mathbf{c})$.
Then at time step 3, for all $y_3 \in \mathcal{Y}$, we compute

$$
(10.8.3)\[\begin{split}\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}\end{split}
$$

and pick the largest two among these ten values, say
$P(A, B, D \mid \mathbf{c})$ and
$P(C, E, D \mid \mathbf{c}).$ As a result, we get six candidates
output sequences: (i) $A$; (ii) $C$; (iii) $A$,
$B$; (iv) $C$, $E$; (v) $A$, $B$,
$D$; and (vi) $C$, $E$, $D$.

In the end, we obtain the set of final candidate output sequences based
on these six sequences (e.g., discard portions including and after
“<eos>”). Then we choose the output sequence which maximizes the
following score:

$$
(10.8.4)\[\frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c});
$$

here $L$ is the length of the final candidate sequence and
$\alpha$ is usually set to 0.75. Since a longer sequence has more
logarithmic terms in the summation of (10.8.4),
the term $L^\alpha$ in the denominator penalizes long sequences.

The computational cost of beam search is
$\mathcal{O}(k\left|\mathcal{Y}\right|T')$. This result is in
between that of greedy search and that of exhaustive search. Greedy
search can be treated as a special case of beam search arising when the
beam size is set to 1.

## 10.8.4. Summary

Sequence searching strategies include greedy search, exhaustive search,
and beam search. Beam search provides a trade-off between accuracy and
computational cost via the flexible choice of the beam size.
