#!/usr/bin/env python3
"""Refit and verify the five reported current Recovery heads.

The released development bundle contains only numeric features, binary EM
outcomes, opaque group identifiers, and the frozen fit/calibration role. It
contains no benchmark text, answers, passages, references, or original IDs.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from src.arbitration.empirical_contract import FIELDS  # noqa: E402
from src.arbitration.empirical_panel import fit_panel  # noqa: E402
BUNDLE = ROOT / "outputs/reproduction_v1/DEVELOPMENT_NUMERIC.jsonl.gz"
ACCEPTED = ROOT / "outputs/reproduction_v1/CURRENT_HEADS.json"


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


def exact_array(actual, expected, name: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise AssertionError(f"parameter mismatch: {name}")


def verify_model(actual: dict, expected: dict, variant: str) -> None:
    for field in ("variant", "head", "feature_dimension", "base_iterations", "platt_iterations", "target_counts"):
        if actual[field] != expected[field]:
            raise AssertionError(f"model field mismatch: {variant}.{field}")
    for field in ("median", "mean", "std"):
        exact_array(actual["preprocessing"][field], expected["preprocessing"][field], f"{variant}.preprocessing.{field}")
    exact_array(actual["coef"], expected["coef"], f"{variant}.coef")
    for field in ("intercept", "platt_slope", "platt_intercept"):
        if actual[field] != expected[field]:
            raise AssertionError(f"model scalar mismatch: {variant}.{field}")
    if actual["design_hashes"] != expected["design_hashes"]:
        raise AssertionError(f"design hash mismatch: {variant}")


def main() -> int:
    rows, outcomes, parts = load()
    if (len(rows), len(parts["fit"]), len(parts["cal"])) != (13_500, 10_800, 2_700):
        raise AssertionError("development population mismatch")
    eligible = {name: sum(rows[key]["eligible"] for key in parts[name]) for name in ("fit", "cal")}
    if eligible != {"fit": 2_572, "cal": 630}:
        raise AssertionError("eligible sample counts mismatch")

    accepted = json.loads(ACCEPTED.read_text(encoding="utf-8"))["heads"]
    events = []
    with threadpool_limits(limits=1):
        for variant, indices in FIELDS.items():
            selected = {
                key: {"eligible": row["eligible"], "numeric": [row["numeric"][index] for index in indices]}
                for key, row in rows.items()
            }
            model, _, ranking = fit_panel(selected, outcomes, parts, variant, "public_refit_" + variant, events.append)
            verify_model(model, accepted[variant], variant)
            if ranking != accepted[variant]["ranking"]:
                raise AssertionError(f"ranking metadata mismatch: {variant}")

    completed = [event for event in events if event["event"] == "fit_completed"]
    if len(completed) != 10:
        raise AssertionError("expected ten completed fit calls")
    print(json.dumps({
        "status": "PASS_EXACT_CURRENT_HEAD_REFIT",
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
