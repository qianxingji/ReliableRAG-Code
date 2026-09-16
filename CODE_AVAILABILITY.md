# Code availability statement

Project-authored method and reporting code is provided in this repository under
Apache License 2.0. The repository includes the authenticated historical HGB
implementation, paired GbV scorer, supervision-matched fitting and calibration
procedure, fixed selection and analysis kernels, a text-free 18,000-trace
numeric reproduction bundle, sealed aggregate outputs, and runnable tests.

The numeric bundle permits exact reconstruction of the reported point estimates
and 20,000-draw bootstrap. It contains only opaque group IDs, numeric scores,
actions, and EM/F1 outcomes. The repository excludes benchmark text and original
IDs, generated answer text, retrieved passages, third-party model weights,
learned paper estimators, development labels, and private forensic evidence.
It therefore does not reproduce the complete neural execution from raw data.
Exact third-party model revisions and access boundaries are documented in
`THIRD_PARTY_NOTICES.md` and `docs/REPRODUCIBILITY.md`.
