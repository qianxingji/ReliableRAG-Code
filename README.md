# ReliableRAG

Code and aggregate verification artifacts for **Supervision-Matched Selection of Paired RAG Repairs: An Empirical Study of Accuracy and Damage**.

ReliableRAG studies a fixed post-generation decision: after the same reader has produced an original answer and a repaired answer, should the system keep the original or switch to the repair? The accepted experiment uses 6,000 question groups, 18,000 traces, three multi-hop QA datasets, three retrievers, one Qwen2.5-3B-Instruct reader, and a fixed global budget of 900 replacements per non-Keep policy.

## What this repository contains

- the byte-identical historical state-symmetric HGB feature and selector source used by the study;
- the paired Generate-but-Verify NLI scoring component;
- fixed cohort, eligibility, top-K allocation, and aggregate analysis code;
- sealed aggregate point estimates and confidence-interval summaries;
- a standard-library verifier that checks 129 reporting statements; and
- tests that exercise the released code without downloading models or benchmark data.

The only supported joint positive comparison is `HGB_GBV_R` versus `GBV_ONLY_R`. The study does not establish a joint advantage over `HGB_ONLY_R`, advancement of `ROA-FULL`, reader transfer, unseen-domain transfer, a new selector architecture, or an end-to-end deployment claim.

## Quick verification

Python 3.10 or newer is required.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/verify_cas_q3_claim_statistics.py
python scripts/verify_repository.py
```

The aggregate command must report `PASS_CAS_Q3_STATISTICAL_STATEMENT_VERIFICATION` with 129 checks. These commands perform no model fit and no neural forward pass.

## Repository map

- `src/mars/state_symmetric.py`: authenticated historical 48-feature HGB implementation.
- `src/verification/gbv_nli.py`: paired post-answering NLI scorer.
- `src/arbitration/empirical_contract.py`: fixed development and fresh-cohort membership rules.
- `src/evaluation/`: answer normalization and resampled top-K allocation.
- `scripts/empirical_analysis_math.py`: frozen point-estimate and bootstrap arithmetic.
- `outputs/cas_q2/empirical_analysis_v1/`: sealed aggregate results. The historical directory name is retained because its hashes are cited by the verifier; the paper's current target is CAS Q3.
- `docs/`: method, data, reproduction, and Claim boundaries.

## Data and model boundary

Benchmark questions, contexts, answers, generated candidates, per-question outcomes, action memberships, model weights, learned estimators, indexes, bootstrap multiplicities, and private forensic receipts are not redistributed. Download benchmark data and pretrained models from their official sources under their respective terms. See `DATA.md`, `THIRD_PARTY_NOTICES.md`, and `docs/REPRODUCIBILITY.md`.

The public package verifies the reporting layer and exposes the method code. It does not regenerate the accepted 18,000 neural traces from the private, hash-bound execution record.

## License

Project-authored source code is available under Apache License 2.0. Dataset, model, dependency, aggregate-result, manuscript, and third-party rights remain separate; see `LICENSE_SCOPE.md`.

## Citation

Citation metadata is in `CITATION.cff`. Replace its manuscript status with the final article DOI after publication and archive the exact release in a DOI-minting repository.
