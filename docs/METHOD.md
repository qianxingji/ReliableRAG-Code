# Released method definition

## Historical HGB signal

HGB is the frozen historical histogram-based gradient-boosting pairwise selector implemented in `src/mars/state_symmetric.py`. For answer `a_i` and evidence `E_j`, `L_ij` is the mean teacher-forced token log probability of the answer under that evidence. Each branch has eight features: own likelihood, cross likelihood, their difference, answer token length, added-document lexical compatibility, removed-document lexical compatibility, added-document sequence similarity, and removed-document sequence similarity.

The selector forms eight repaired-minus-original differences and interacts each with five pair-context variables: evidence-ID overlap, added-minus-removed retrieval score, new-document novelty, answer token F1, and saved BGE answer similarity. The resulting input has 48 dimensions. Dataset and retriever identity are not HGB inputs.

Historical fitting includes only pairs where exactly one answer is exact-match correct. The label is one for an incorrect original followed by a correct repair, and zero for the reverse transition. Every `(x,y)` is augmented with `(-x,1-y)`. At scoring time,

`h(x) = 0.5 * [p_H(1|x) + 1 - p_H(1|-x)]`.

A larger score favors the repaired answer. Fixed HGB parameters are 300 iterations, 15 maximum leaf nodes, learning rate 0.08, L2 regularization 1.0, and seed 20261834. The released source has SHA-256 `3724b5ac77722b70cabdf2589379d5584942f34ec81f3f5a04f88e0665f8f717`.

## Paired GbV signal and current heads

The GbV component assigns each branch the maximum entailment probability over its evidence chunks. The paired signal is repaired-branch score minus original-branch score. The fixed verifier is `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` at revision `5a4338ab2151dc8db04ad53b42b6153382bf4f99`.

Raw HGB ranks directly by the frozen score. `HGB_ONLY_R` is a separate calibrated L2-logistic Recovery head using HGB, its missingness indicator, and retriever indicators. `HGB_GBV_R` adds the paired GbV margin and its missingness indicator. Neither current head refits historical HGB.
