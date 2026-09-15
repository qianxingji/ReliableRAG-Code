"""Supervision-matched fitting for the five current Recovery heads.

This public module preserves the accepted fit-only preprocessing, L2-logistic
base fit, disjoint Platt calibration, and calibrated ranking logic. It consumes
already constructed numeric rows and numeric training outcomes; it does not
load benchmark text, generate candidates, or access fresh evaluation labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .empirical_contract import BASE, PLATT, RETRIEVERS, recovery_target


WIDTHS = {
    "ROA-FULL": 11,
    "ROA-NOGBV": 10,
    "HGB_GBV_R": 2,
    "HGB_ONLY_R": 1,
    "GBV_ONLY_R": 1,
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _key_hash(keys):
    digest = hashlib.sha256()
    for key in sorted(keys):
        payload = json.dumps(list(key), ensure_ascii=False, separators=(",", ":")) + "\n"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def _numeric(rows, keys, width):
    return np.array(
        [[np.nan if value is None else value for value in rows[key]["numeric"][:width]] for key in keys],
        dtype=np.float64,
    )


def _matrix(values, keys, median, mean, std):
    missing = ~np.isfinite(values)
    retriever_indicators = np.array(
        [[float(key[1] == retriever) for retriever in RETRIEVERS] for key in keys]
    )
    standardized = (np.where(missing, median, values) - mean) / std
    return np.concatenate([standardized, missing.astype(float), retriever_indicators], axis=1)


def fit_panel(rows, training_outcomes, partitions, variant, job_id, event):
    """Fit one base head and one disjoint Platt calibrator.

    ``partitions`` must contain ``fit``, ``cal``, and ``probe`` key sets. Only
    eligible rows enter numerical fitting. ``probe`` may overlap development
    data and is used for saved-parameter checking, not performance estimation.
    """
    _require(variant in WIDTHS, "UNKNOWN_VARIANT")
    eligible_keys = {
        name: sorted(key for key in keys if rows[key]["eligible"])
        for name, keys in partitions.items()
    }
    _require(
        set(training_outcomes) == set(partitions["fit"]) | set(partitions["cal"]),
        "TRAINING_LABEL_SCOPE",
    )
    width = WIDTHS[variant]
    _require(all(len(row["numeric"]) == width for row in rows.values()), "CONTROL_FEATURE_WIDTH")

    numeric = {
        name: _numeric(rows, keys, width) for name, keys in eligible_keys.items()
    }
    _require(np.isfinite(numeric["fit"]).any(axis=0).all(), "ALL_MISSING_FIT_COLUMN")
    median = np.nanmedian(numeric["fit"], axis=0)
    filled = np.where(np.isfinite(numeric["fit"]), numeric["fit"], median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    design = {
        name: _matrix(numeric[name], eligible_keys[name], median, mean, std)
        for name in eligible_keys
    }
    targets = {
        name: np.array(
            [recovery_target(training_outcomes[key]) for key in eligible_keys[name]],
            dtype=int,
        )
        for name in ("fit", "cal")
    }
    _require(all(set(values) == {0, 1} for values in targets.values()), "MISSING_CLASS")

    def run(stage, values, labels, parameters, keys):
        entry = dict(
            job_id=job_id,
            variant=variant,
            head="R",
            stage=stage,
            started_utc=_now(),
            parameters=parameters,
            sample_count=len(labels),
            class_counts=np.bincount(labels, minlength=2).tolist(),
            design_sha256=_array_hash(values),
            target_sha256=hashlib.sha256(np.asarray(labels, dtype=np.uint8).tobytes()).hexdigest(),
            keys_sha256=_key_hash(keys),
        )
        event(dict(event="fit_started", **entry))
        model = LogisticRegression(**parameters)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(values, labels)
        _require(
            not any(issubclass(item.category, ConvergenceWarning) for item in captured),
            "FIT_NONCONVERGENCE",
        )
        _require(
            np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),
            "NONFINITE_COEFFICIENTS",
        )
        event(
            dict(
                event="fit_completed",
                **entry,
                completed_utc=_now(),
                iterations=model.n_iter_.tolist(),
                warnings=[str(item.message) for item in captured],
            )
        )
        return model

    base = run("base", design["fit"], targets["fit"], BASE, eligible_keys["fit"])
    calibration_logits = design["cal"] @ base.coef_[0] + base.intercept_[0]
    platt = run(
        "platt",
        calibration_logits.reshape(-1, 1),
        targets["cal"],
        PLATT,
        eligible_keys["cal"],
    )

    raw = design["probe"] @ base.coef_[0] + base.intercept_[0]
    probability = expit(raw * platt.coef_[0, 0] + platt.intercept_[0])
    model_record = dict(
        job_id=job_id,
        variant=variant,
        head="R",
        feature_dimension=2 * width + 3,
        preprocessing=dict(median=median.tolist(), mean=mean.tolist(), std=std.tolist()),
        coef=base.coef_[0].tolist(),
        intercept=float(base.intercept_[0]),
        platt_slope=float(platt.coef_[0, 0]),
        platt_intercept=float(platt.intercept_[0]),
        base_iterations=base.n_iter_.tolist(),
        platt_iterations=platt.n_iter_.tolist(),
        partition_key_hashes={name: _key_hash(eligible_keys[name]) for name in eligible_keys},
        design_hashes={name: _array_hash(design[name]) for name in eligible_keys},
        target_counts={
            name: np.bincount(targets[name], minlength=2).tolist() for name in targets
        },
    )
    predictions = {
        key: dict(logit_R=float(raw[index]), pR=float(probability[index]))
        for index, key in enumerate(eligible_keys["probe"])
    }

    ordered = sorted(predictions, key=lambda key: (predictions[key]["pR"], key))
    previous_max = float("-inf")
    reversals = 0
    tied_blocks = 0
    for _, group in itertools.groupby(ordered, key=lambda key: predictions[key]["pR"]):
        block = list(group)
        logits = [predictions[key]["logit_R"] for key in block]
        if len(block) > 1:
            tied_blocks += 1
        if min(logits) < previous_max:
            reversals += 1
        previous_max = max(previous_max, max(logits))
    ranking = dict(
        platt_slope=model_record["platt_slope"],
        strict_order_reversal_blocks=reversals,
        calibrated_tie_blocks=tied_blocks,
        ranking_unchanged_except_ties=reversals == 0,
        calibrated_ranking_always_used=True,
    )
    return model_record, predictions, ranking
