# ReliableRAG

Code and aggregate verification artifacts for **Supervision-Matched Selection of Paired RAG Repairs: An Empirical Study of Accuracy and Damage**.

ReliableRAG studies a fixed post-generation decision: after the same reader has produced an original answer and a repaired answer, should the system keep the original or switch to the repair? The accepted experiment uses 6,000 question groups, 18,000 traces, three multi-hop QA datasets, three retrievers, one Qwen2.5-3B-Instruct reader, and a fixed global budget of 900 replacements per non-Keep policy.

## What this repository contains

- the byte-identical historical state-symmetric HGB feature and selector source used by the study;
- the paired Generate-but-Verify NLI scoring component;
- the supervision-matched logistic fitting and disjoint calibration procedure used by all five current heads;
- fixed cohort, eligibility, top-K allocation, and aggregate analysis code;
- a text-free 18,000-trace numeric bundle with opaque question-group IDs;
- sealed aggregate point estimates and confidence-interval summaries;
- a standard-library verifier that checks 129 reporting statements; and
- tests that exercise the released code without downloading models or benchmark data.

The only supported joint positive comparison is `HGB_GBV_R` versus `GBV_ONLY_R`. The study does not establish a joint advantage over `HGB_ONLY_R`, advancement of `ROA-FULL`, reader transfer, unseen-domain transfer, a new selector architecture, or an end-to-end deployment claim.

## Full public statistical reproduction

Python 3.10 or newer is required. The lock file records the exact tested
non-neural environment.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/reproduce_all.py
```

The workflow rebuilds all nine-policy point estimates and the complete 20,000-draw
dataset-stratified question-cluster bootstrap from released trace-level numeric
records. The rebuilt point estimates and intervals must equal the sealed public
files exactly. It then runs the unit suite, the 129-statement reporting verifier,
and the repository-integrity verifier. No command performs a model fit or neural
forward pass.

The trace bundle contains scores, actions, and numeric EM/F1 outcomes. It uses
dataset-local `qNNNN` group identifiers that preserve the canonical tie order;
benchmark sample IDs and all question, context, reference, answer, and passage
text are absent.

## Repository map

- `src/mars/state_symmetric.py`: authenticated historical 48-feature HGB implementation.
- `src/verification/gbv_nli.py`: paired post-answering NLI scorer.
- `src/arbitration/empirical_contract.py`: fixed development and fresh-cohort membership rules.
- `src/arbitration/empirical_panel.py`: eligible-row preprocessing, L2-logistic fitting, and disjoint Platt calibration for the five current Recovery heads.
- `src/evaluation/`: answer normalization and resampled top-K allocation.
- `scripts/empirical_analysis_math.py`: frozen point-estimate and bootstrap arithmetic.
- `scripts/reproduce_paper_statistics.py`: exact point-estimate and full-bootstrap reconstruction.
- `outputs/cas_q2/empirical_analysis_v1/`: sealed aggregate results. The historical directory name is retained because its hashes are cited by the verifier; the paper's current target is CAS Q3.
- `outputs/reproduction_v1/`: compressed text-free trace-level numeric inputs and their release manifest.
- `docs/`: method, data, reproduction, and Claim boundaries.

## Data and model boundary

Benchmark questions, contexts, answers, generated candidates, original sample
identifiers, model weights, learned estimators, indexes, and private forensic
receipts are not redistributed. The public numeric bundle is sufficient to
reconstruct the reported selection statistics and bootstrap, but not to rerun
retrieval, generation, neural scoring, or model fitting. Download benchmark data
and pretrained models from their official sources under their respective terms.
See `DATA.md`, `THIRD_PARTY_NOTICES.md`, and `docs/REPRODUCIBILITY.md`.

The public package reproduces the statistical analysis from released numeric
traces and exposes the method code. It does not regenerate the accepted 18,000
neural traces from benchmark text and model assets.

## License

Project-authored source code is available under Apache License 2.0. Dataset, model, dependency, aggregate-result, manuscript, and third-party rights remain separate; see `LICENSE_SCOPE.md`.

## Citation

Citation metadata is in `CITATION.cff`. Replace its manuscript status with the final article DOI after publication and archive the exact release in a DOI-minting repository.
