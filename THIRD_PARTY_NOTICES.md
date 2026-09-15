# Third-party asset and dependency inventory

This inventory records the principal third-party materials used by the full
private study. The aggregate candidate does not contain any dataset payload,
model weight, third-party wheel or copied third-party source. Links point to
the upstream project or license record inspected for release planning.

## Benchmarks

| Material | Study use | Upstream declaration | Release action |
|---|---|---|---|
| [HotpotQA](https://github.com/hotpotqa/hotpot) | Benchmark-supplied source frames and labels | Dataset: CC BY-SA 4.0; repository code: Apache-2.0 | Cite and link upstream; exclude questions, contexts and answers from this candidate. Any later data redistribution must preserve attribution and applicable share-alike terms. |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | Benchmark-supplied source frames and labels | Repository declares Apache-2.0. The inspected README does not state a separate license for the externally downloaded dataset archive. | Cite and link upstream; exclude all dataset payloads. Do not infer that the repository code license independently authorizes redistribution of every downloaded data file. |
| [MuSiQue](https://github.com/stonybrooknlp/musique) | Benchmark-supplied source frames and labels | Dataset: CC BY 4.0 | Cite and link upstream; exclude questions, contexts and answers. Retain the upstream warning about overlap with its seed single-hop datasets in the contamination discussion. |

## Pretrained models

| Exact study asset | Upstream declaration | Release action |
|---|---|---|
| [Qwen/Qwen2.5-3B-Instruct@aa8e725](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE) | Qwen Research License Agreement; non-commercial research/evaluation grant with separate redistribution conditions | Exclude all weights, tokenizer/config payloads and answer text. Retain exact model/revision attribution. A future weight package would need the upstream agreement and required notice and is outside the current release. |
| [BAAI/bge-base-en-v1.5@a5beb1e](https://huggingface.co/BAAI/bge-base-en-v1.5/tree/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a) | The upstream model card declares MIT and says the released model can be used commercially | Exclude weights and tokenizer payloads; retain exact model/revision attribution and upstream link. |
| [MoritzLaurer/deberta-v3-large-zeroshot-v2.0@5a4338a](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0/tree/5a4338ab2151dc8db04ad53b42b6153382bf4f99) | The upstream model card labels the foundation model MIT but warns that this non-`-c` checkpoint was trained on data with varying licenses, including non-commercial licenses, and that legal views differ on downstream effect | Exclude weights and tokenizer payloads; retain exact model/revision attribution and upstream link. Do not reduce the training-data caveat to an unconditional MIT redistribution statement; require owner/institutional review before any broader release or commercial reuse. |

## Principal pinned software

The exact private environment lock files record wheel hashes. This table covers
the software named in the study's release audit; it is not a replacement for
the license files and third-party notices bundled inside any redistributed
binary wheel.

| Pinned package | Upstream license record | Declared license |
|---|---|---|
| torch 2.7.1+cu128 | [PyTorch v2.7.1 LICENSE](https://github.com/pytorch/pytorch/blob/v2.7.1/LICENSE) | BSD-style 3-clause terms; bundled third-party notices also apply |
| transformers 4.53.2 | [Transformers v4.53.2 LICENSE](https://github.com/huggingface/transformers/blob/v4.53.2/LICENSE) | Apache-2.0 |
| NumPy 2.2.6 | [NumPy v2.2.6 LICENSE](https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt) | BSD-3-Clause |
| SciPy 1.15.3 | [SciPy v1.15.3 LICENSE](https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt) | BSD-3-Clause |
| scikit-learn 1.7.2 | [scikit-learn 1.7.2 COPYING](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING) | BSD-3-Clause |
| joblib 1.5.3 | [joblib 1.5.3 LICENSE](https://github.com/joblib/joblib/blob/1.5.3/LICENSE.txt) | BSD-3-Clause |
| safetensors 0.8.0 | [safetensors v0.8.0 LICENSE](https://github.com/huggingface/safetensors/blob/v0.8.0/LICENSE) | Apache-2.0 |
| sentencepiece 0.2.1 | [SentencePiece v0.2.1 LICENSE](https://github.com/google/sentencepiece/blob/v0.2.1/LICENSE) | Apache-2.0 |
| PyArrow 20.0.0 | [Apache Arrow 20.0.0 LICENSE](https://github.com/apache/arrow/blob/apache-arrow-20.0.0/LICENSE.txt) | Apache-2.0 plus bundled notices |

The aggregate verification command uses only the Python standard library. No
third-party package above is copied into the candidate archive. The archived
statistical source refers to NumPy for transparency but is read as text by the
aggregate verifier and is not executed by that command.
