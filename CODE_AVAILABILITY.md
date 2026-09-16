# Code availability statement

Project-authored method and reporting code is provided in this repository under
Apache License 2.0. The repository includes the authenticated historical HGB
implementation, paired GbV scorer, supervision-matched fitting and calibration
procedure, fixed selection and analysis kernels, a text-free 18,000-trace
numeric evaluation bundle, a text-free 13,500-trace development bundle, the five
accepted current-head parameter records, sealed aggregate outputs, and runnable
tests.

The development bundle permits exact refitting of the five current logistic
heads, while the evaluation bundle permits exact reconstruction of the reported
point estimates and 20,000-draw bootstrap. Both use opaque group IDs and numeric
fields only. The repository excludes benchmark text and original IDs, generated
answer text, retrieved passages, third-party model weights, historical learned
estimators, and private forensic evidence. It therefore does not reproduce the
complete neural execution from raw data.
Exact third-party model revisions and access boundaries are documented in
`THIRD_PARTY_NOTICES.md` and `docs/REPRODUCIBILITY.md`.
