"""Verify CAS Q3 statistical statements from sealed aggregate evidence only.

This verifier does not read question outcomes, actions, Gold, bootstrap weights,
or model artifacts. It checks the reporting layer against the already accepted
and independently validated empirical-D aggregates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs/cas_q2/empirical_analysis_v1"
EXPECTED_HASHES = {
    "POINT_ESTIMATES.json": "b03ddad8fa35f582a63403c029942104c3f5da1a961110edc2a62f09871f4d3b",
    "INTERVALS.json": "6d454afeec7c125c0cc4d182556af6db214a867aa4f62f7a6fbd1e6e22b09331",
    "SHA256_MANIFEST.json": "498b83e75538032de3711bd6ce190eac049631d8393c875744d83aa0318097f6",
}
POLICIES = (
    "Keep", "HGB", "GbV", "ROA-FULL", "ROA-NOGBV", "HGB_GBV_R",
    "HGB_ONLY_R", "GBV_ONLY_R", "V2",
)
COMPARISONS = (
    ("ROA-FULL", "HGB_GBV_R"),
    ("HGB_GBV_R", "HGB_ONLY_R"),
    ("HGB_GBV_R", "GBV_ONLY_R"),
)


class Audit:
    def __init__(self) -> None:
        self.checks = 0

    def equal(self, actual, expected, label: str) -> None:
        self.checks += 1
        if actual != expected:
            raise ValueError(f"{label}: {actual!r} != {expected!r}")

    def true(self, value: bool, label: str) -> None:
        self.checks += 1
        if not value:
            raise ValueError(label)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction(bounds: list[float]) -> str:
    lower, upper = bounds
    return "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive"


def main() -> None:
    audit = Audit()
    for name, expected in EXPECTED_HASHES.items():
        audit.equal(sha256(ANALYSIS / name), expected, f"sealed hash {name}")

    points = json.loads((ANALYSIS / "POINT_ESTIMATES.json").read_text(encoding="utf-8"))
    intervals = json.loads((ANALYSIS / "INTERVALS.json").read_text(encoding="utf-8"))
    audit.equal(points["N_all"], 18000, "trace denominator")
    audit.equal(points["question_groups"], 6000, "question clusters")
    audit.equal(points["N_eligible"], 4267, "common eligible rows")
    audit.equal(points["primary_global_cap"], 900, "global action cap")
    audit.equal(tuple(points["policies"]), POLICIES, "complete policy order")

    keep = points["policies"]["Keep"]
    audit.equal(keep["replacements"], 0, "Keep actions")
    for name in POLICIES:
        row = points["policies"][name]
        audit.equal(row["recovery"] + row["damage"] + row["neutral"], row["replacements"], f"{name} event partition")
        audit.equal(row["net"], row["recovery"] - row["damage"], f"{name} net")
        audit.equal(sum(row["realized_em_transition_counts"].values()), 18000, f"{name} transition denominator")
        audit.true(abs(row["em_rate"] - row["em_correct"] / 18000) < 1e-15, f"{name} EM rate")
        audit.true(abs(row["damage_rate_pp"] - 100 * row["damage"] / 18000) < 1e-15, f"{name} Damage rate")
        audit.true(abs(row["delta_em_pp"] - 100 * row["net"] / 18000) < 1e-15, f"{name} EM delta")
        if name != "Keep":
            audit.equal(row["replacements"], 900, f"{name} fixed action count")

    audit.equal(intervals["draws"], 20000, "bootstrap draws")
    audit.equal(intervals["interval_family_size"], 6, "primary endpoint family")
    audit.equal(intervals["adjusted_quantiles"], [0.05 / 12, 1 - 0.05 / 12], "Bonferroni percentile quantiles")
    audit.equal(intervals["unadjusted_quantiles"], [0.025, 0.975], "secondary percentile quantiles")
    audit.equal(intervals["quantile_method"], "linear", "quantile method")
    audit.equal(intervals["denominator"], 18000, "interval denominator")
    audit.equal(intervals["reallocated"]["interval_role"], "primary", "reallocated role")
    audit.equal(intervals["fixed_action"]["interval_role"], "secondary_sensitivity_same_draws", "fixed-action role")

    point_comparisons = {(x["left"], x["right"]): x for x in points["comparisons"]}
    for mode in ("reallocated", "fixed_action"):
        observed = intervals[mode]["comparisons"]
        audit.equal(tuple((x["left"], x["right"]) for x in observed), COMPARISONS, f"{mode} comparison family")
        for comparison in observed:
            pair = comparison["left"], comparison["right"]
            base = point_comparisons[pair]
            expected_joint = True
            for endpoint in ("em_difference_pp", "damage_rate_difference_pp"):
                item = comparison["endpoints"][endpoint]
                audit.equal(item["point_pp"], base[endpoint], f"{mode} {pair} {endpoint} point")
                audit.equal(item["adjusted_direction"], direction(item["adjusted_percentile_range_pp"]), f"{mode} {pair} {endpoint} direction")
                audit.equal(len(item["secondary_unadjusted_95_range_pp"]), 2, f"{mode} {pair} {endpoint} secondary range")
                wanted = "positive" if endpoint == "em_difference_pp" else "negative"
                expected_joint &= item["adjusted_direction"] == wanted
            audit.equal(comparison["joint_em_improvement_and_damage_reduction"], expected_joint, f"{mode} {pair} joint rule")

    source = (ROOT / "scripts/empirical_analysis_math.py").read_text(encoding="utf-8")
    for literal in (
        "SEED, DRAWS = 20260926, 20000",
        "sampled = rng.choice(stratum, size=panel.q_per_dataset, replace=True)",
        "row_weights = weights.astype(np.int64, copy=False)[panel.siblings]",
        "weighted_top_k(panel.orders[policy], row_weights, cap)",
        "panel.actions[policy] * row_weights",
        "[.05 / 12, 1 - .05 / 12]",
    ):
        audit.true(literal in source, f"frozen grouped-bootstrap source binding: {literal}")

    receipt = {
        "status": "PASS_CAS_Q3_STATISTICAL_STATEMENT_VERIFICATION",
        "checks": audit.checks,
        "sealed_inputs_sha256": EXPECTED_HASHES,
        "population": {"question_groups": 6000, "traces": 18000, "dataset_strata": 3},
        "action_allocation": {"global_k": 900, "fraction": 0.05, "reallocated_each_primary_draw": True},
        "bootstrap": {
            "draws": 20000,
            "seed": 20260926,
            "unit": "dataset-stratified question cluster with three retriever siblings",
            "primary_family_endpoints": 6,
            "adjusted_quantiles": intervals["adjusted_quantiles"],
            "fixed_action_role": "secondary_sensitivity_same_draws",
        },
        "primary_joint_results": [
            {
                "left": x["left"],
                "right": x["right"],
                "joint_favorable": x["joint_em_improvement_and_damage_reduction"],
            }
            for x in intervals["reallocated"]["comparisons"]
        ],
        "scope": "sealed aggregate reporting verification; no raw-outcome or bootstrap recomputation",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
