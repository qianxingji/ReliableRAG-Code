"""Frozen D arithmetic on numeric outcomes and immutable prelabel ledgers only.

This pure module neither opens files nor authorizes Gold access or execution.
"""
from collections import Counter
import math

import numpy as np

from src.evaluation.batch_allocation import weighted_top_k

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
RETRIEVERS = ("bm25", "dense", "hybrid")
POLICIES = ("Keep", "HGB", "GbV", "ROA-FULL", "ROA-NOGBV", "HGB_GBV_R", "HGB_ONLY_R", "GBV_ONLY_R", "V2")
COMPARISONS = (("ROA-FULL", "HGB_GBV_R"), ("HGB_GBV_R", "HGB_ONLY_R"), ("HGB_GBV_R", "GBV_ONLY_R"))
ENDPOINTS = ("em_difference_pp", "damage_rate_difference_pp")
KEYS = ("dataset", "retriever", "sample_id")
SEED, DRAWS = 20260926, 20000


def require(value, message):
    if not value:
        raise ValueError("Frozen D analysis: " + message)


def key(row):
    require(all(type(row[k]) is str and row[k] for k in KEYS), "nonempty exact identity strings")
    k = tuple(row[k] for k in KEYS)
    require(k[0] in DATASETS and k[1] in RETRIEVERS, "fixed trace stratum")
    return k


class Panel:
    """Authenticated callers supply complete rows; small sizes are for toy tests."""
    def __init__(self, actions, outcomes, *, questions_per_dataset=2000):
        require(type(questions_per_dataset) is int and questions_per_dataset > 0, "fixed positive question count")
        action_rows, outcome_rows = {}, {}
        for row in actions:
            require(type(row) is dict and set(row) == set(KEYS) | {"eligible", "scores", "actions", "forced_keep_reason"}, "exact prelabel row schema")
            k = key(row)
            require(k not in action_rows, "unique prelabel identity")
            require(type(row["eligible"]) is bool and set(row["scores"]) == set(POLICIES[1:]) and set(row["actions"]) == set(POLICIES), "fixed policy/mask schema")
            reason = row["forced_keep_reason"]
            require(reason is None if row["eligible"] else type(reason) is str and bool(reason), "common eligibility reason")
            for score in row["scores"].values():
                require(type(score) in (int, float) and math.isfinite(score) if row["eligible"] else score is None, "complete finite common score mask")
            require(all(type(value) is str and value in {"KEEP", "REPLACE"} for value in row["actions"].values()), "exact original action values")
            action_rows[k] = row
        for row in outcomes:
            require(type(row) is dict and set(row) == set(KEYS) | {"a0_em", "a1_em", "a0_f1", "a1_f1"}, "numeric-only outcome schema")
            k = key(row)
            require(k not in outcome_rows, "unique numeric outcome identity")
            for field in ("a0_em", "a1_em"):
                require(type(row[field]) is int and row[field] in (0, 1), "binary integer EM")
            for field in ("a0_f1", "a1_f1"):
                require(type(row[field]) in (int, float) and math.isfinite(row[field]) and 0 <= row[field] <= 1, "finite bounded F1")
            outcome_rows[k] = row
        require(set(action_rows) == set(outcome_rows), "complete action/outcome identity equality")
        self.keys = sorted(action_rows)
        self.n = len(self.keys)
        self.q_per_dataset = questions_per_dataset
        require(self.n == 9 * questions_per_dataset, "complete balanced all-trace population")
        self.groups = sorted({(k[0], k[2]) for k in self.keys})
        require(Counter(ds for ds, _ in self.groups) == Counter({d: questions_per_dataset for d in DATASETS}), "balanced selected question groups")
        require(Counter((k[0], k[2]) for k in self.keys) == Counter({g: 3 for g in self.groups}), "three distinct siblings per question")
        index = {g: i for i, g in enumerate(self.groups)}
        self.siblings = np.array([index[(k[0], k[2])] for k in self.keys], dtype=np.int64)
        self.strata = [np.array([i for i, g in enumerate(self.groups) if g[0] == d], dtype=np.int64) for d in DATASETS]
        self.outcomes = [outcome_rows[k] for k in self.keys]
        self.em = np.array([[r["a0_em"], r["a1_em"]] for r in self.outcomes], dtype=np.int64)
        self.gain = self.em[:, 1] - self.em[:, 0]
        self.damage = ((self.em[:, 0] == 1) & (self.em[:, 1] == 0)).astype(np.int64)
        self.cap = round(.05 * self.n)
        self.actions, self.orders = {}, {}
        eligible = [i for i, k in enumerate(self.keys) if action_rows[k]["eligible"]]
        self.eligible_count = len(eligible)
        unit_weights = np.ones(self.n, dtype=np.int64)
        for policy in POLICIES:
            actual = np.array([int(action_rows[k]["actions"][policy] == "REPLACE") for k in self.keys], dtype=np.int64)
            order = np.array([] if policy == "Keep" else sorted(eligible, key=lambda i: (-action_rows[self.keys[i]]["scores"][policy], self.keys[i])), dtype=np.int64)
            expected = weighted_top_k(order, unit_weights, self.cap)
            require(np.array_equal(actual, expected), "sealed primary actions are immutable and must match global allocation")
            self.actions[policy], self.orders[policy] = actual, order


def _point(panel, indices):
    require(bool(indices), "nonempty fixed analysis cell")
    n = len(indices)
    before_em = sum(int(panel.em[i, 0]) for i in indices)
    before_f1 = math.fsum(panel.outcomes[i]["a0_f1"] for i in indices)
    result = {}
    for policy in POLICIES:
        selected = panel.actions[policy]
        changes = [i for i in indices if selected[i]]
        recovery = sum(panel.gain[i] == 1 for i in changes)
        damage = sum(panel.damage[i] for i in changes)
        after_em = sum(int(panel.em[i, int(selected[i])]) for i in indices)
        after_f1 = math.fsum(panel.outcomes[i]["a1_f1" if selected[i] else "a0_f1"] for i in indices)
        table = {a + b: sum(int(panel.em[i, 0]) == int(a) and int(panel.em[i, int(selected[i])]) == int(b) for i in indices)
            for a in "01" for b in "01"}
        result[policy] = dict(N_all=n, replacements=len(changes), recovery=int(recovery), damage=int(damage),
            neutral=int(len(changes) - recovery - damage), net=int(recovery - damage), em_correct=after_em,
            em_rate=after_em / n, token_f1=after_f1 / n, delta_em_pp=100. * (after_em - before_em) / n,
            delta_f1_pp=100. * (after_f1 - before_f1) / n, damage_rate_pp=100. * int(damage) / n,
            realized_em_transition_counts=table)
    return result


def point_estimates(panel):
    all_rows = _point(panel, list(range(panel.n)))
    breakdown = {}
    for kind, labels in (("dataset", DATASETS), ("retriever", RETRIEVERS),
        ("dataset_x_retriever", [(d, r) for d in DATASETS for r in RETRIEVERS])):
        cells = []
        for label in labels:
            indices = [i for i, k in enumerate(panel.keys) if (k[0] if kind == "dataset" else k[1] if kind == "retriever" else k[:2]) == label]
            cells.append(dict(cell=label, policies=_point(panel, indices)))
        breakdown[kind] = cells
    comparison = []
    for left, right in COMPARISONS:
        net = all_rows[left]["net"] - all_rows[right]["net"]
        damage = all_rows[left]["damage"] - all_rows[right]["damage"]
        comparison.append(dict(left=left, right=right, event_differences=[net, damage],
            **{field: 100. * value / panel.n for field, value in zip(ENDPOINTS, (net, damage))}))
    return dict(N_all=panel.n, question_groups=len(panel.groups), N_eligible=panel.eligible_count, primary_global_cap=panel.cap,
        policies=all_rows, comparisons=comparison, fixed_global_action_breakdowns=breakdown)


def question_weights(panel, *, draws=DRAWS, seed=SEED):
    """Actual executors use the constants; optional small settings are toy tests."""
    require(type(draws) is int and draws > 0 and type(seed) is int, "fixed draw count and seed")
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        weights = np.zeros(len(panel.groups), dtype=np.int64)
        for stratum in panel.strata:
            sampled = rng.choice(stratum, size=panel.q_per_dataset, replace=True)
            weights += np.bincount(sampled, minlength=len(panel.groups))
        yield weights


def draw_record(panel, weights, draw):
    weights = np.asarray(weights)
    require(type(draw) is int and draw >= 0 and weights.ndim == 1 and weights.dtype.kind in "iu" and
        len(weights) == len(panel.groups) and np.all(weights >= 0) and np.all(weights <= panel.q_per_dataset), "integer canonical question multiplicities")
    require(all(int(weights[s].sum()) == panel.q_per_dataset for s in panel.strata), "each draw preserves every dataset size")
    row_weights = weights.astype(np.int64, copy=False)[panel.siblings]
    count = int(row_weights.sum())
    require(count == panel.n, "all siblings and ineligible copies in denominator")
    cap = round(.05 * count)
    analyses = {}
    for mode in ("reallocated", "fixed_action"):
        totals = {}
        for policy in POLICIES:
            chosen = weighted_top_k(panel.orders[policy], row_weights, cap) if mode == "reallocated" else panel.actions[policy] * row_weights
            totals[policy] = dict(net=int(chosen @ panel.gain), damage=int(chosen @ panel.damage), replacements=int(chosen.sum()))
        differences = [[totals[a][f] - totals[b][f] for f in ("net", "damage")] for a, b in COMPARISONS]
        analyses[mode] = dict(policy_totals=totals, paired_event_differences=differences)
    return dict(draw=draw, N_all=count, cap=cap, **analyses)


def intervals(panel, records):
    records = list(records)
    require(bool(records) and all(r["draw"] == i and r["N_all"] == panel.n and r["cap"] == panel.cap for i, r in enumerate(records)), "complete ordered draw records")
    points = point_estimates(panel)["comparisons"]
    reports = {}
    for mode in ("reallocated", "fixed_action"):
        counts = np.array([r[mode]["paired_event_differences"] for r in records])
        require(counts.dtype.kind in "iu" and counts.shape == (len(records), 3, 2), "integer six-endpoint draws")
        values = counts * (100. / panel.n)
        adjusted = np.quantile(values, [.05 / 12, 1 - .05 / 12], axis=0, method="linear")
        ordinary = np.quantile(values, [.025, .975], axis=0, method="linear")
        comparisons = []
        for i, (left, right) in enumerate(COMPARISONS):
            endpoints = {}
            for j, endpoint in enumerate(ENDPOINTS):
                lower, upper = map(float, adjusted[:, i, j])
                direction = "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive"
                endpoints[endpoint] = dict(point_pp=points[i][endpoint], adjusted_percentile_range_pp=[lower, upper],
                    secondary_unadjusted_95_range_pp=ordinary[:, i, j].tolist(), adjusted_direction=direction)
            em, damage = [endpoints[e]["adjusted_direction"] for e in ENDPOINTS]
            comparisons.append(dict(left=left, right=right, endpoints=endpoints,
                joint_em_improvement_and_damage_reduction=em == "positive" and damage == "negative"))
        reports[mode] = dict(comparisons=comparisons, interval_role="primary" if mode == "reallocated" else "secondary_sensitivity_same_draws")
    return dict(draws=len(records), interval_family_size=6, adjusted_quantiles=[.05 / 12, 1 - .05 / 12],
        unadjusted_quantiles=[.025, .975], quantile_method="linear", denominator=panel.n, **reports)
