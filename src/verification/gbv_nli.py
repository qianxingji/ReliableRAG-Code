from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

GBV_MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
GBV_MODEL_REVISION = "5a4338ab2151dc8db04ad53b42b6153382bf4f99"
GBV_HYPOTHESIS_TEMPLATE = 'The answer to the question "{question}" is: "{answer}"'
GBV_WORD_OVERLAP = 20


def format_hypothesis(question: str, answer: str) -> str:
    return GBV_HYPOTHESIS_TEMPLATE.format(question=str(question), answer=str(answer))


def resolve_entailment_index(id2label: dict[Any, str]) -> int:
    """Resolve the unique positive entailment class and fail closed otherwise.

    The pinned GbV model uses the binary labels ``entailment`` and
    ``not_entailment``. Substring matching is therefore unsafe because the negative
    label also contains ``entailment``. After the existing normalization, require an
    exact positive ``entailment`` label.
    """
    matches: list[int] = []
    for raw_index, raw_label in id2label.items():
        label = re.sub(r"[^a-z]", "", str(raw_label).lower())
        if label == "entailment":
            matches.append(int(raw_index))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one entailment label, found {matches}")
    return matches[0]


def pair_token_length(tokenizer: Any, premise: str, hypothesis: str) -> int:
    encoded = tokenizer(
        premise,
        hypothesis,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("pair_token_length expects one premise/hypothesis pair")
        ids = ids[0]
    return len(ids)


def split_passage_to_fit(
    tokenizer: Any,
    passage: str,
    hypothesis: str,
    *,
    max_length: int,
    overlap_words: int = GBV_WORD_OVERLAP,
) -> list[str]:
    """Split an overlength passage into fit-checked word windows.

    The published GbV recipe uses 20-word overlap for overlength premises. No
    scientific truncation is allowed here: every emitted premise+hypothesis pair is
    checked against the model context window, and an impossible pair fails closed.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    passage = "" if passage is None else str(passage)
    hypothesis = str(hypothesis)
    if pair_token_length(tokenizer, "", hypothesis) > max_length:
        raise ValueError("hypothesis does not fit the NLI context window")
    if pair_token_length(tokenizer, passage, hypothesis) <= max_length:
        return [passage]

    words = passage.split()
    if not words:
        return [""]

    chunks: list[str] = []
    start = 0
    n_words = len(words)
    while start < n_words:
        lo, hi = start + 1, n_words
        best: int | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(words[start:mid])
            if pair_token_length(tokenizer, candidate, hypothesis) <= max_length:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise ValueError(
                f"single-word premise at position {start} does not fit the NLI context window"
            )
        chunk = " ".join(words[start:best])
        if pair_token_length(tokenizer, chunk, hypothesis) > max_length:
            raise AssertionError("chunking emitted an overlength pair")
        chunks.append(chunk)
        if best >= n_words:
            break
        next_start = max(start + 1, best - overlap_words)
        if next_start <= start:
            raise AssertionError("chunking did not make forward progress")
        start = next_start
    return chunks


def effective_model_max_length(tokenizer: Any, model_config: Any) -> int:
    candidates: list[int] = []
    token_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(token_limit, int) and 0 < token_limit < 1_000_000:
        candidates.append(token_limit)
    model_limit = getattr(model_config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        candidates.append(model_limit)
    if not candidates:
        raise ValueError("could not determine a finite NLI context length")
    return min(candidates)


@dataclass(frozen=True)
class BranchScore:
    score: float
    premise_count: int
    chunk_count: int


class GBVPostAnsweringNLI:
    """Generate-but-Verify Post-Answering NLI scorer for completed RAG branches."""

    def __init__(
        self,
        *,
        model_id: str = GBV_MODEL_ID,
        revision: str = GBV_MODEL_REVISION,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: str = "float32",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "GbV scoring requires torch and transformers; install the fresh-baseline dependencies"
            ) from exc

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"unsupported torch_dtype: {torch_dtype}")

        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.requested_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=False,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype_map[torch_dtype],
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.model.to(device)
        self.entailment_index = resolve_entailment_index(self.model.config.id2label)
        self.max_length = effective_model_max_length(self.tokenizer, self.model.config)

    @property
    def resolved_dtype(self) -> str:
        try:
            return str(next(self.model.parameters()).dtype)
        except StopIteration:  # pragma: no cover
            return "unknown"

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        torch = self.torch
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises = [premise for premise, _ in batch]
            hypotheses = [hypothesis for _, hypothesis in batch]
            encoded = self.tokenizer(
                premises,
                hypotheses,
                add_special_tokens=True,
                truncation=False,
                padding=True,
                return_tensors="pt",
            )
            if int(encoded["input_ids"].shape[1]) > self.max_length:
                raise RuntimeError("overlength NLI pair reached model scoring")
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits.float(), dim=-1)
            scores.extend(
                float(value)
                for value in probabilities[:, self.entailment_index].detach().cpu().tolist()
            )
        return scores

    def score_branch(
        self,
        question: str,
        answer: str,
        evidence_passages: Iterable[str],
    ) -> BranchScore:
        hypothesis = format_hypothesis(question, answer)
        premises = [str(passage) for passage in evidence_passages]
        if not premises:
            raise ValueError("branch contains no evidence passages")
        pairs: list[tuple[str, str]] = []
        for passage in premises:
            chunks = split_passage_to_fit(
                self.tokenizer,
                passage,
                hypothesis,
                max_length=self.max_length,
                overlap_words=GBV_WORD_OVERLAP,
            )
            pairs.extend((chunk, hypothesis) for chunk in chunks)
        if not pairs:
            raise ValueError("branch produced no scorable premise/hypothesis pairs")
        scores = self._score_pairs(pairs)
        return BranchScore(
            score=max(scores),
            premise_count=len(premises),
            chunk_count=len(pairs),
        )
