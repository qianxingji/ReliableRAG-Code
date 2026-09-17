# Reproducibility levels

## Publicly runnable

1. Install `requirements-lock.txt`, then install the package with `--no-deps`.
2. Run `python scripts/reproduce_all.py`.

The workflow reads a text-free development bundle and refits the five current
Recovery heads: five base logistic regressions and five disjoint Platt
calibrators. It verifies preprocessing values, coefficients, and calibration
parameters to a fixed `1e-10` absolute/relative tolerance across the tested
platforms. Iterations, target counts, design hashes, ranking metadata, and the
complete eligible probe ordering remain exact checks. The
development fit contains 2,572 eligible traces (557 Recovery positives and
2,015 negatives); calibration contains 630 eligible traces (132 positives and
498 negatives).

The workflow then reads the text-free evaluation bundle, rebuilds all nine-policy
point estimates, regenerates the fixed 20,000 dataset-stratified question-cluster
draws from seed `20260926`, repeats global top-K allocation within every primary
draw, computes the fixed-action sensitivity, and requires exact equality with
the sealed public point and interval files. It also authenticates repository
membership and reporting statements. On the tested environment the full
statistical reconstruction completes in seconds.

The unit suite also performs one small synthetic fit. The complete workflow
performs ten paper-head fits plus the synthetic test and zero neural forwards.

## Source inspection

`src/mars/state_symmetric.py` is a byte-identical copy of the authenticated historical HGB source. The public `src/evaluation/__init__.py` is a minimal compatibility export for the released answer-normalization helper; it is not part of the historical HGB source hash.

`src/arbitration/empirical_panel.py` and `src/arbitration/empirical_contract.py`
expose the current five-head fitting contract described in the manuscript. They
accept already assembled numeric rows and numeric development outcomes. The
fitted current-head parameters and their text-free development matrix are
released with opaque group identities. The evaluation rows likewise contain
only opaque group identities, numeric policy scores/actions, and numeric EM/F1
outcomes.

## Not publicly reproduced

The complete 18,000-trace neural acquisition depends on benchmark-derived pools, pretrained snapshots, generated answers, learned estimators, and private execution receipts that are not redistributed. A retained 180-trace replay is a bounded witness, not a second complete run. Seven historical upstream estimators also lack original fit-time ID/matrix receipts and an independent original-fit witness. A reconstructed HGB fit reproduced all 1,539 historical scores exactly, while its serialized file differed only in saved CPU-thread metadata.

The public command produces numerically equivalent current logistic heads and
exactly reproduces the reported statistical layer from the realized numeric
traces. It must not be
described as an end-to-end reproduction of retrieval, generation, neural
scoring, or historical training.
