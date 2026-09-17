#!/usr/bin/env python3
"""Refit and verify the five reported current Recovery heads.

The released development bundle contains only numeric features, binary EM
outcomes, opaque group identifiers, and the frozen fit/calibration role. It
contains no benchmark text, answers, passages, references, or original IDs.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.arbitration.empirical_contract import FIELDS, RETRIEVERS  # noqa: E402
from src.arbitration.empirical_panel import fit_panel  # noqa: E402
BUNDLE = ROOT / "outputs/reproduction_v1/DEVELOPMENT_NUMERIC.jsonl.gz"
ACCEPTED = ROOT / "outputs/reproduction_v1/CURRENT_HEADS.json"
PARAMETER_ATOL = 1e-10
PARAMETER_RTOL = 1e-10


def load() -> tuple[dict, dict, dict]:
    rows = {}
    outcomes = {}
    parts = {"fit": set(), "cal": set(), "probe": set()}
    with gzip.open(BUNDLE, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["dataset"], record["retriever"], record["group_id"])
            if key in rows:
                raise AssertionError("duplicate development key")
            rows[key] = {"eligible": record["eligible"], "numeric": record["numeric"]}
            outcomes[key] = {"a0_em": record["a0_em"], "a1_em": record["a1_em"]}
            parts[record["role"]].add(key)
            parts["probe"].add(key)
    return rows, outcomes, parts


def array_close(actual, expected, name: str) -> dict:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"parameter shape mismatch: {name}: "
            f"{actual_array.shape} != {expected_array.shape}"
        )
    absolute = np.abs(actual_array - expected_array)
    relative = absolute / np.maximum(np.abs(expected_array), np.finfo(np.float64).tiny)
    maximum_absolute = float(absolute.max(initial=0.0))
    maximum_relative = float(relative.max(initial=0.0))
    if not np.allclose(
        actual_array,
        expected_array,
        atol=PARAMETER_ATOL,
        rtol=PARAMETER_RTOL,
        equal_nan=False,
    ):
        raise AssertionError(
            f"parameter mismatch: {name}; max_abs={maximum_absolute:.17g}; "
            f"max_rel={maximum_relative:.17g}; atol={PARAMETER_ATOL}; "
            f"rtol={PARAMETER_RTOL}"
        )
    return {"max_abs": maximum_absolute, "max_rel": maximum_relative}


def scalar_close(actual, expected, name: str) -> dict:
    actual_value = float(actual)
    expected_value = float(expected)
    absolute = abs(actual_value - expected_value)
    relative = absolute / max(abs(expected_value), np.finfo(np.float64).tiny)
    if not math.isclose(
        actual_value,
        expected_value,
        abs_tol=PARAMETER_ATOL,
        rel_tol=PARAMETER_RTOL,
    ):
        raise AssertionError(
            f"model scalar mismatch: {name}; abs={absolute:.17g}; "
            f"rel={relative:.17g}; atol={PARAMETER_ATOL}; rtol={PARAMETER_RTOL}"
        )
    return {"max_abs": absolute, "max_rel": relative}


def verify_model(actual: dict, expected: dict, variant: str) -> dict:
    differences = []
    for field in ("variant", "head", "feature_dimension", "base_iterations", "platt_iterations", "target_counts"):
        if actual[field] != expected[field]:
            raise AssertionError(f"model field mismatch: {variant}.{field}")
    for field in ("median", "mean", "std"):
        differences.append(
            array_close(
                actual["preprocessing"][field],
                expected["preprocessing"][field],
                f"{variant}.preprocessing.{field}",
            )
        )
    differences.append(array_close(actual["coef"], expected["coef"], f"{variant}.coef"))
    for field in ("intercept", "platt_slope", "platt_intercept"):
        differences.append(scalar_close(actual[field], expected[field], f"{variant}.{field}"))
    if actual["design_hashes"] != expected["design_hashes"]:
        raise AssertionError(f"design hash mismatch: {variant}")
    return {
        "max_abs": max(item["max_abs"] for item in differences),
        "max_rel": max(item["max_rel"] for item in differences),
    }


def accepted_predictions(selected: dict, parts: dict, model: dict) -> dict:
    """Apply one accepted record to every eligible probe row."""
    keys = sorted(key for key in parts["probe"] if selected[key]["eligible"])
    values = np.asarray(
        [
            [np.nan if value is None else value for value in selected[key]["numeric"]]
            for key in keys
        ],
        dtype=np.float64,
    )
    median = np.asarray(model["preprocessing"]["median"], dtype=np.float64)
    mean = np.asarray(model["preprocessing"]["mean"], dtype=np.float64)
    std = np.asarray(model["preprocessing"]["std"], dtype=np.float64)
    missing = ~np.isfinite(values)
    standardized = (np.where(missing, median, values) - mean) / std
    retrievers = np.asarray(
        [[float(key[1] == retriever) for retriever in RETRIEVERS] for key in keys],
        dtype=np.float64,
    )
    design = np.concatenate([standardized, missing.astype(float), retrievers], axis=1)
    raw = design @ np.asarray(model["coef"], dtype=np.float64) + float(model["intercept"])
    probability = expit(
        raw * float(model["platt_slope"]) + float(model["platt_intercept"])
    )
    return {
        key: {"logit_R": float(raw[index]), "pR": float(probability[index])}
        for index, key in enumerate(keys)
    }


def verify_ranking(
    actual_predictions: dict,
    actual_ranking: dict,
    expected_predictions: dict,
    expected_ranking: dict,
    variant: str,
) -> None:
    scalar_close(
        actual_ranking["platt_slope"],
        expected_ranking["platt_slope"],
        f"{variant}.ranking.platt_slope",
    )
    for field in (
        "strict_order_reversal_blocks",
        "calibrated_tie_blocks",
        "ranking_unchanged_except_ties",
        "calibrated_ranking_always_used",
    ):
        if actual_ranking[field] != expected_ranking[field]:
            raise AssertionError(f"ranking metadata mismatch: {variant}.{field}")
    actual_order = sorted(
        actual_predictions,
        key=lambda key: (actual_predictions[key]["pR"], key),
    )
    expected_order = sorted(
        expected_predictions,
        key=lambda key: (expected_predictions[key]["pR"], key),
    )
    if actual_order != expected_order:
        first = next(
            index
            for index, pair in enumerate(zip(actual_order, expected_order))
            if pair[0] != pair[1]
        )
        raise AssertionError(
            f"accepted/refit prediction order mismatch: {variant}; "
            f"first_index={first}; actual={actual_order[first]}; "
            f"expected={expected_order[first]}"
        )


def main() -> int:
    rows, outcomes, parts = load()
    if (len(rows), len(parts["fit"]), len(parts["cal"])) != (13_500, 10_800, 2_700):
        raise AssertionError("development population mismatch")
    eligible = {name: sum(rows[key]["eligible"] for key in parts[name]) for name in ("fit", "cal")}
    if eligible != {"fit": 2_572, "cal": 630}:
        raise AssertionError("eligible sample counts mismatch")

    accepted = json.loads(ACCEPTED.read_text(encoding="utf-8"))["heads"]
    events = []
    parameter_differences = {}
    with threadpool_limits(limits=1):
        for variant, indices in FIELDS.items():
            selected = {
                key: {"eligible": row["eligible"], "numeric": [row["numeric"][index] for index in indices]}
                for key, row in rows.items()
            }
            model, predictions, ranking = fit_panel(
                selected,
                outcomes,
                parts,
                variant,
                "public_refit_" + variant,
                events.append,
            )
            parameter_differences[variant] = verify_model(model, accepted[variant], variant)
            verify_ranking(
                predictions,
                ranking,
                accepted_predictions(selected, parts, accepted[variant]),
                accepted[variant]["ranking"],
                variant,
            )

    completed = [event for event in events if event["event"] == "fit_completed"]
    if len(completed) != 10:
        raise AssertionError("expected ten completed fit calls")
    print(json.dumps({
        "status": "PASS_NUMERICALLY_EQUIVALENT_CURRENT_HEAD_REFIT",
        "parameter_atol": PARAMETER_ATOL,
        "parameter_rtol": PARAMETER_RTOL,
        "complete_probe_ranking_match": True,
        "parameter_differences": parameter_differences,
        "development_traces": len(rows),
        "fit_eligible": 2_572,
        "fit_positive": 557,
        "calibration_eligible": 630,
        "calibration_positive": 132,
        "scientific_fit_calls": 10,
        "heads": list(FIELDS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
