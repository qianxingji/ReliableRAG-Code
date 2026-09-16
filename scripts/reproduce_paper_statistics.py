#!/usr/bin/env python3
"""Recompute the paper point estimates and 20,000-draw intervals.

This command uses the public text-free numeric trace bundle. It performs no
model fit, neural forward pass, benchmark download, or answer-text access.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.empirical_analysis_math import (  # noqa: E402
    COMPARISONS,
    Panel,
    intervals,
    point_estimates,
    question_weights,
)
from src.evaluation.batch_allocation import weighted_top_k  # noqa: E402


BUNDLE_DIR = ROOT / "outputs/reproduction_v1"
BUNDLE = BUNDLE_DIR / "TRACE_NUMERIC.jsonl.gz"
BUNDLE_MANIFEST = BUNDLE_DIR / "MANIFEST.json"
ACCEPTED_POINTS = ROOT / "outputs/cas_q2/empirical_analysis_v1/POINT_ESTIMATES.json"
ACCEPTED_INTERVALS = ROOT / "outputs/cas_q2/empirical_analysis_v1/INTERVALS.json"
EXPECTED_SCHEMA = {
    "dataset", "retriever", "group_id", "eligible", "forced_keep_reason",
    "scores", "actions", "a0_em", "a1_em", "a0_f1", "a1_f1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def load_panel() -> Panel:
    manifest = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    record = manifest["files"][0]
    require(record["path"] == BUNDLE.name, "unexpected bundle path")
    require(BUNDLE.stat().st_size == record["size_bytes"] and sha256(BUNDLE) == record["sha256"], "numeric bundle identity mismatch")
    require(manifest["records"] == 18000 and manifest["question_groups"] == 6000, "numeric bundle population mismatch")

    actions, outcomes = [], []
    with gzip.open(BUNDLE, "rt", encoding="utf-8", newline="") as stream:
        for number, line in enumerate(stream, 1):
            require(line.endswith("\n") and bool(line.strip()), f"invalid bundle JSONL line {number}")
            row = json.loads(line)
            require(set(row) == EXPECTED_SCHEMA, f"unexpected bundle schema at line {number}")
            require(isinstance(row["group_id"], str) and len(row["group_id"]) == 5 and row["group_id"].startswith("q") and row["group_id"][1:].isdigit(), f"invalid opaque group ID at line {number}")
            identity = {"dataset": row["dataset"], "retriever": row["retriever"], "sample_id": row["group_id"]}
            actions.append({
                **identity,
                "eligible": row["eligible"],
                "forced_keep_reason": row["forced_keep_reason"],
                "scores": row["scores"],
                "actions": row["actions"],
            })
            outcomes.append({
                **identity,
                "a0_em": row["a0_em"],
                "a1_em": row["a1_em"],
                "a0_f1": row["a0_f1"],
                "a1_f1": row["a1_f1"],
            })
    require(len(actions) == 18000, "numeric bundle row count mismatch")
    return Panel(actions, outcomes)


def bootstrap_records(panel: Panel) -> list[dict]:
    required_policies = tuple(dict.fromkeys(policy for pair in COMPARISONS for policy in pair))
    records = []
    for draw, weights in enumerate(question_weights(panel)):
        row_weights = weights.astype(np.int64, copy=False)[panel.siblings]
        require(int(row_weights.sum()) == panel.n, "bootstrap denominator mismatch")
        analyses = {}
        for mode in ("reallocated", "fixed_action"):
            totals = {}
            for policy in required_policies:
                selected = (
                    weighted_top_k(panel.orders[policy], row_weights, panel.cap)
                    if mode == "reallocated"
                    else panel.actions[policy] * row_weights
                )
                totals[policy] = {
                    "net": int(selected @ panel.gain),
                    "damage": int(selected @ panel.damage),
                }
            analyses[mode] = {
                "paired_event_differences": [
                    [totals[left][field] - totals[right][field] for field in ("net", "damage")]
                    for left, right in COMPARISONS
                ]
            }
        records.append({"draw": draw, "N_all": panel.n, "cap": panel.cap, **analyses})
        if (draw + 1) % 2000 == 0:
            print(f"bootstrap {draw + 1}/20000", flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()

    panel = load_panel()
    rebuilt_points = canonical(point_estimates(panel))
    accepted_points = json.loads(ACCEPTED_POINTS.read_text(encoding="utf-8"))
    require(rebuilt_points == accepted_points, "point estimates differ from the sealed public result")

    records = bootstrap_records(panel)
    rebuilt_intervals = canonical(intervals(panel, records))
    accepted_intervals = json.loads(ACCEPTED_INTERVALS.read_text(encoding="utf-8"))
    require(rebuilt_intervals == accepted_intervals, "bootstrap intervals differ from the sealed public result")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    point_path = output / "POINT_ESTIMATES_REBUILT.json"
    interval_path = output / "INTERVALS_REBUILT.json"
    point_path.write_text(json.dumps(rebuilt_points, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    interval_path.write_text(json.dumps(rebuilt_intervals, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "status": "PASS_FULL_PUBLIC_STATISTICAL_REPRODUCTION",
        "traces": panel.n,
        "question_groups": len(panel.groups),
        "eligible_traces": panel.eligible_count,
        "bootstrap_draws": len(records),
        "bootstrap_seed": 20260926,
        "point_estimates_exact": True,
        "intervals_exact": True,
        "scientific_fits": 0,
        "model_forwards": 0,
        "outputs": {
            point_path.name: sha256(point_path),
            interval_path.name: sha256(interval_path),
        },
    }
    receipt = output / "REPRODUCTION_RECEIPT.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
