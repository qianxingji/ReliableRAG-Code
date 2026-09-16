# Data access and redistribution

The study uses HotpotQA, 2WikiMultiHopQA, and MuSiQue. This repository does not redistribute their questions, contexts, annotations, or answers. Obtain each benchmark from its official project or repository and comply with its terms.

The JSON files under `outputs/cas_q2/empirical_analysis_v1/` are sealed aggregate
study results. The compressed
`outputs/reproduction_v1/TRACE_NUMERIC.jsonl.gz` file contains the 18,000
trace-level numeric analysis inputs: common eligibility, policy scores, realized
actions, and original/repaired EM and token-F1 outcomes. It replaces each
benchmark sample identifier with a dataset-local `qNNNN` ordinal. The ordinal
preserves the canonical tie order and the three-retriever sibling grouping.

The numeric bundle contains no question text, context text, reference answer,
generated answer, retrieved passage, annotation, or original benchmark sample
identifier. `outputs/reproduction_v1/MANIFEST.json` binds the bundle to the
sealed private inputs and accepted aggregate outputs by SHA-256.

The accepted experiment used fixed, dataset-specific pools derived from the
selected benchmark contexts. Those pools and the generated candidate pairs
remain outside this public repository. The public repository supports exact
point-estimate and full-bootstrap reconstruction, source inspection, and
aggregate-result verification. It does not support end-to-end retrieval,
generation, neural scoring, or head refitting.

Editors or reviewers who require deeper evidence may request access through a channel approved by the handling editor, subject to benchmark, model, institutional, and confidentiality constraints.
