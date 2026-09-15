# Claim and evidence summary

## Study object

Each trace has a fixed original answer and a fixed repaired answer. A policy
selects Keep or Replace after both candidates and all policy scores have been
acquired. The accepted reader is Qwen2.5-3B-Instruct at revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`.

The population contains 6,000 question groups and 18,000 traces from
HotpotQA, 2WikiMultiHopQA and MuSiQue, each crossed with BM25, dense and hybrid
retrieval. The bounded source pools contain 19,352, 11,746 and 23,618 rows,
respectively. These are source-pool sizes rather than evaluation denominators.

## Policies and fairness

The complete policy set is Keep, HGB, GbV, ROA-FULL, ROA-NOGBV, HGB_GBV_R,
HGB_ONLY_R, GBV_ONLY_R and V2. HGB is an upstream signal and comparator. GbV is
a paired adaptation of Generate but Verify Post-Answering NLI. HGB_GBV_R is an
ordinary calibrated logistic head over HGB and GbV signals.

Five fitted heads use the same development identities, labels, fit budget,
eligibility mask and global top-K allocation. The study made 185 scientific
fits in total: 178 control/panel fits and seven historical reproduction fits.
Every non-Keep policy acts on 900 of 18,000 traces under the primary fixed
allocation.

## Outcomes and intervals

EM Recovery counts selected 0-to-1 transitions. Damage counts selected 1-to-0
transitions. Net equals Recovery minus Damage, and full-population EM change is
`100 * Net / 18,000` percentage points. Damage rate is
`100 * Damage / 18,000` percentage points.

The primary family comprises EM and Damage for three ordered comparisons.
Intervals use 20,000 dataset-stratified question-cluster bootstrap draws. Each
question's three retriever siblings share its multiplicity, and the global
900-action allocation is recomputed in every primary draw. Bonferroni-adjusted
percentile bounds use quantiles 0.05/12 and 1-0.05/12. Fixed original actions on
the same draws are secondary sensitivity results.

| Ordered comparison | EM difference, pp | Damage difference, pp | Joint rule |
|---|---:|---:|---|
| ROA-FULL - HGB_GBV_R | +0.0444 [-0.1333,+0.2037] | +0.0611 [+0.0000,+0.1278] | Not met |
| HGB_GBV_R - HGB_ONLY_R | +0.0556 [-0.1556,+0.3167] | -0.0833 [-0.1833,-0.0056] | Not met |
| HGB_GBV_R - GBV_ONLY_R | +0.2333 [+0.0722,+0.4333] | -0.0722 [-0.1444,-0.0278] | Met |

The joint rule requires a strictly positive adjusted EM range and a strictly
negative adjusted Damage range. A bound touching zero is inconclusive.

## Claim boundary

The evidence supports a comparison-dependent empirical result on one fixed
Qwen reader condition. It does not support method novelty, superiority over
HGB_ONLY_R, advancement of ROA-FULL, reader transfer, unseen-domain transfer,
causal or counterfactual identification, a formal risk guarantee, improved
faithfulness, or reduced end-to-end computation.

ID-level separation was checked for the recorded cohorts, but semantic overlap
and pretrained-model contamination were not audited. Phi evidence terminated
at a failed semantic validation gate, and the Mistral extension stopped before
engineering; neither is effect evidence.
