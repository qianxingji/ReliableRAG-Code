# Reproducibility levels

## Publicly runnable

1. Install the Python package.
2. Run the unit tests.
3. Run `python scripts/verify_cas_q3_claim_statistics.py`.
4. Run `python scripts/verify_repository.py`.

These checks authenticate the released sources and aggregate records, recompute reporting arithmetic, and confirm the stated Claim boundary. The unit suite performs one small synthetic base/calibration fit to exercise the released current-head procedure; it does not refit any paper model or execute a neural network. The two standalone verifiers perform zero fits and zero neural forwards.

## Source inspection

`src/mars/state_symmetric.py` is a byte-identical copy of the authenticated historical HGB source. The public `src/evaluation/__init__.py` is a minimal compatibility export for the released answer-normalization helper; it is not part of the historical HGB source hash.

`src/arbitration/empirical_panel.py` and `src/arbitration/empirical_contract.py` expose the current five-head fitting contract described in the manuscript. They accept already assembled numeric rows and numeric development outcomes. Fitted paper parameters and the labeled rows from which they were learned are outside the public release.

## Not publicly reproduced

The complete 18,000-trace neural acquisition depends on benchmark-derived pools, pretrained snapshots, generated answers, learned estimators, and private execution receipts that are not redistributed. A retained 180-trace replay is a bounded witness, not a second complete run. Seven historical upstream estimators also lack original fit-time ID/matrix receipts and an independent original-fit witness. A reconstructed HGB fit reproduced all 1,539 historical scores exactly, while its serialized file differed only in saved CPU-thread metadata.

No public command should be described as an end-to-end reproduction of the historical execution.
