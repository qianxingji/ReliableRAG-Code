# Code availability statement

Project-authored method and reporting code is provided in this repository under Apache License 2.0. The repository includes the authenticated historical HGB implementation, the paired GbV scorer, fixed selection and analysis kernels, sealed aggregate outputs, and runnable verification tests.

The repository excludes third-party model weights and benchmark payloads, generated answer text, per-question outcomes, and private forensic evidence. These exclusions mean that the public package verifies the reported aggregate layer but does not reproduce the complete neural execution from raw data. Exact third-party model revisions and access boundaries are documented in `THIRD_PARTY_NOTICES.md` and `docs/REPRODUCIBILITY.md`.
