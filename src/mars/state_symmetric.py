"""State-symmetric pairwise MARS feature representation and selectors."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation import normalize_answer


def _tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text.casefold())


def _token_f1(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2.0 * precision * recall / (precision + recall)


def _sequence_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, normalize_answer(left), normalize_answer(right)).ratio())


def _evidence_change(trace: dict[str, Any]) -> tuple[str, str, float, float, float, float]:
    e0, e1 = trace["E0"], trace["E1"]
    ids0 = {str(row["document_id"]) for row in e0}
    ids1 = {str(row["document_id"]) for row in e1}
    added = [row for row in e1 if str(row["document_id"]) not in ids0]
    removed = [row for row in e0 if str(row["document_id"]) not in ids1]
    added_text = " ".join(str(row.get("text", "")) for row in added)
    removed_text = " ".join(str(row.get("text", "")) for row in removed)
    added_score = float(added[0].get("score", 0.0)) if added else 0.0
    removed_score = float(removed[0].get("score", 0.0)) if removed else 0.0
    overlap = len(ids0 & ids1) / len(ids0 | ids1) if ids0 | ids1 else 1.0
    added_tokens, existing_tokens = set(_tokens(added_text)), [set(_tokens(str(row.get("text", "")))) for row in e0]
    novelty = 1.0 - max((len(added_tokens & item) / len(added_tokens | item) if added_tokens | item else 1.0 for item in existing_tokens), default=0.0)
    return added_text, removed_text, added_score, removed_score, overlap, novelty


PHI_NAMES = (
    "own_likelihood", "cross_likelihood", "own_minus_cross", "answer_token_length",
    "added_lexical_compatibility", "removed_lexical_compatibility",
    "added_semantic_compatibility", "removed_semantic_compatibility",
)

INVARIANT_NAMES = (
    "evidence_id_overlap", "added_document_score", "removed_document_score",
    "added_minus_removed_score", "new_document_novelty", "answer_exact_agreement",
    "answer_token_similarity", "answer_semantic_similarity",
)

DELTA_NAMES = tuple(f"delta_{name}" for name in PHI_NAMES)
INTERACTION_BASES = (
    "evidence_id_overlap", "added_minus_removed_score", "new_document_novelty",
    "answer_token_similarity", "answer_semantic_similarity",
)
SYMMETRIC_FEATURE_NAMES = DELTA_NAMES + tuple(
    f"{delta}_x_{invariant}" for delta in DELTA_NAMES for invariant in INTERACTION_BASES
)
NO_CROSS_FEATURE_NAMES = tuple(
    name for name in SYMMETRIC_FEATURE_NAMES if "cross_likelihood" not in name and "own_minus_cross" not in name
)


def build_pair_record(trace: dict[str, Any], cells: dict[str, dict[str, Any]], *, schema_version: str) -> dict[str, Any]:
    likelihood = {name: float(cells[name]["mean_log_probability"]) for name in ("L00", "L01", "L10", "L11")}
    a0, a1 = str(trace["a0"]), str(trace["a1"])
    added, removed, added_score, removed_score, evidence_overlap, novelty = _evidence_change(trace)
    phi0 = {
        "own_likelihood": likelihood["L00"], "cross_likelihood": likelihood["L01"],
        "own_minus_cross": likelihood["L00"] - likelihood["L01"],
        "answer_token_length": float(len(_tokens(a0))),
        "added_lexical_compatibility": _token_f1(a0, added),
        "removed_lexical_compatibility": _token_f1(a0, removed),
        "added_semantic_compatibility": _sequence_similarity(a0, added),
        "removed_semantic_compatibility": _sequence_similarity(a0, removed),
    }
    phi1 = {
        "own_likelihood": likelihood["L11"], "cross_likelihood": likelihood["L10"],
        "own_minus_cross": likelihood["L11"] - likelihood["L10"],
        "answer_token_length": float(len(_tokens(a1))),
        "added_lexical_compatibility": _token_f1(a1, added),
        "removed_lexical_compatibility": _token_f1(a1, removed),
        "added_semantic_compatibility": _sequence_similarity(a1, added),
        "removed_semantic_compatibility": _sequence_similarity(a1, removed),
    }
    invariant = {
        "evidence_id_overlap": evidence_overlap, "added_document_score": added_score,
        "removed_document_score": removed_score, "added_minus_removed_score": added_score - removed_score,
        "new_document_novelty": novelty,
        "answer_exact_agreement": float(normalize_answer(a0) == normalize_answer(a1)),
        "answer_token_similarity": _token_f1(a0, a1),
        "answer_semantic_similarity": float(trace.get("answer_semantic_agreement", _sequence_similarity(a0, a1))),
    }
    delta = {f"delta_{name}": phi1[name] - phi0[name] for name in PHI_NAMES}
    features = dict(delta)
    for delta_name, value in delta.items():
        for invariant_name in INTERACTION_BASES:
            features[f"{delta_name}_x_{invariant_name}"] = value * invariant[invariant_name]
    if tuple(features) != SYMMETRIC_FEATURE_NAMES or not all(math.isfinite(float(value)) for value in features.values()):
        raise RuntimeError("Invalid state-symmetric feature record")
    return {
        "sample_id": str(trace["sample_id"]), "dataset": str(trace["dataset"]),
        "split": str(trace["split"]), "retriever": str(trace["retriever"]),
        "schema_version": schema_version, "features": features, "invariant_features": invariant,
        "diagnostics": {
            **likelihood,
            "m0": likelihood["L10"] - likelihood["L00"],
            "m1": likelihood["L11"] - likelihood["L01"],
            "B_repair": (likelihood["L11"] - likelihood["L01"]) - (likelihood["L10"] - likelihood["L00"]),
        },
    }


def matrix(records: list[dict[str, Any]], feature_names: tuple[str, ...] = SYMMETRIC_FEATURE_NAMES) -> np.ndarray:
    values = np.asarray([[float(row["features"][name]) for name in feature_names] for row in records], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("Non-finite state-symmetric feature matrix")
    return values


@dataclass(slots=True)
class SymmetricSelector:
    family: str
    feature_names: tuple[str, ...]
    scaler: StandardScaler | None
    model: Any

    def decision_function(self, records: list[dict[str, Any]]) -> np.ndarray:
        x = matrix(records, self.feature_names)
        if self.family == "logistic":
            transformed = self.scaler.transform(x) if self.scaler is not None else x
            return np.asarray(self.model.decision_function(transformed), dtype=np.float64)
        positive = np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)
        negative = np.asarray(self.model.predict_proba(-x)[:, 1], dtype=np.float64)
        probability = 0.5 * (positive + 1.0 - negative)
        eps = np.finfo(np.float64).eps
        probability = np.clip(probability, eps, 1.0 - eps)
        return np.log(probability / (1.0 - probability))

    def scores(self, records: list[dict[str, Any]]) -> np.ndarray:
        logits = self.decision_function(records)
        return 1.0 / (1.0 + np.exp(-logits))

    def swapped_scores(self, records: list[dict[str, Any]]) -> np.ndarray:
        logits = -self.decision_function(records)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_symmetric_selector(
    records: list[dict[str, Any]], labels: list[int], *, family: str, seed: int,
    feature_names: tuple[str, ...] = SYMMETRIC_FEATURE_NAMES,
    logistic_max_iter: int = 4000, hgb_max_iter: int = 300, hgb_max_leaf_nodes: int = 15,
) -> SymmetricSelector:
    x = matrix(records, feature_names)
    y = np.asarray(labels, dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("State-symmetric fitting requires both preference classes")
    augmented_x = np.concatenate([x, -x], axis=0)
    augmented_y = np.concatenate([y, 1 - y], axis=0)
    if family == "logistic":
        scaler = StandardScaler(with_mean=False).fit(augmented_x)
        transformed = scaler.transform(augmented_x)
        model = LogisticRegression(
            fit_intercept=False, class_weight="balanced", max_iter=logistic_max_iter,
            random_state=seed, solver="lbfgs",
        ).fit(transformed, augmented_y)
        return SymmetricSelector(family, feature_names, scaler, model)
    if family == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=hgb_max_iter, max_leaf_nodes=hgb_max_leaf_nodes,
            learning_rate=0.08, l2_regularization=1.0, random_state=seed,
        ).fit(augmented_x, augmented_y)
        return SymmetricSelector(family, feature_names, None, model)
    raise ValueError(family)


def maximum_antisymmetry_error(selector: SymmetricSelector, records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(np.max(np.abs(selector.scores(records) + selector.swapped_scores(records) - 1.0)))
