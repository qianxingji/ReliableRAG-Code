from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np

from src.arbitration.empirical_contract import DATASETS, RETRIEVERS, split_development
from src.arbitration.empirical_panel import fit_panel
from src.evaluation import assess_pair_eligibility, normalize_answer
from src.evaluation.batch_allocation import weighted_top_k
from src.mars.state_symmetric import SYMMETRIC_FEATURE_NAMES, build_pair_record, matrix
from src.verification.gbv_nli import format_hypothesis, resolve_entailment_index, split_passage_to_fit

ROOT = Path(__file__).resolve().parents[1]


class DummyTokenizer:
    def __call__(self, premise, hypothesis, **kwargs):
        return {"input_ids": list(range(len(str(premise).split()) + len(str(hypothesis).split()) + 3))}


class PublicCoreTests(unittest.TestCase):
    def test_historical_hgb_source_is_exact(self):
        payload = (ROOT / "src/mars/state_symmetric.py").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), "3724b5ac77722b70cabdf2589379d5584942f34ec81f3f5a04f88e0665f8f717")

    def test_hgb_feature_record_has_48_finite_features(self):
        trace = {
            "sample_id": "toy", "dataset": "hotpotqa", "split": "dev", "retriever": "bm25",
            "a0": "Ada Lovelace", "a1": "Ada Byron",
            "E0": [{"document_id": "d0", "text": "Ada wrote notes", "score": 2.0}],
            "E1": [{"document_id": "d1", "text": "Ada Byron wrote notes", "score": 3.0}],
            "answer_semantic_agreement": 0.75,
        }
        cells = {name: {"mean_log_probability": value} for name, value in zip(("L00", "L01", "L10", "L11"), (-1.0, -1.2, -1.3, -0.8))}
        row = build_pair_record(trace, cells, schema_version="toy")
        self.assertEqual(len(SYMMETRIC_FEATURE_NAMES), 48)
        self.assertEqual(matrix([row]).shape, (1, 48))
        self.assertTrue(np.isfinite(matrix([row])).all())

    def test_normalization_and_eligibility(self):
        self.assertEqual(normalize_answer("The, Eiffel Tower!"), "eiffel tower")
        self.assertFalse(assess_pair_eligibility("The Cat", "cat").eligible)
        self.assertTrue(assess_pair_eligibility("cat", "dog").eligible)

    def test_weighted_top_k(self):
        order = np.array([2, 0, 1], dtype=np.int64)
        weights = np.array([2, 1, 3], dtype=np.int64)
        np.testing.assert_array_equal(weighted_top_k(order, weights, 4), [1, 0, 3])

    def test_gbv_helpers(self):
        self.assertEqual(format_hypothesis("Who?", "Ada"), 'The answer to the question "Who?" is: "Ada"')
        self.assertEqual(resolve_entailment_index({0: "entailment", 1: "not_entailment"}), 0)
        chunks = split_passage_to_fit(DummyTokenizer(), "one two three four five six seven eight", "h1 h2", max_length=10, overlap_words=2)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[-1].split()[-1], "eight")

    def test_development_split_keeps_retriever_siblings(self):
        keys = {(dataset, retriever, f"q{i:04d}") for dataset in DATASETS for retriever in RETRIEVERS for i in range(1500)}
        parts = split_development(keys)
        self.assertEqual(len(parts["fit"]), 10800)
        self.assertEqual(len(parts["cal"]), 2700)
        for part in ("fit", "cal"):
            groups = {(d, q) for d, _, q in parts[part]}
            self.assertTrue(all({(d, r, q) for r in RETRIEVERS} <= parts[part] for d, q in groups))

    def test_current_recovery_head_fits_and_calibrates_disjoint_partitions(self):
        keys = [("hotpotqa", "bm25", f"q{i}") for i in range(8)]
        rows = {
            key: {"eligible": True, "numeric": [float(i % 2), float(i % 2)]}
            for i, key in enumerate(keys)
        }
        outcomes = {
            key: {"a0_em": int(i % 2 == 0), "a1_em": int(i % 2 == 1)}
            for i, key in enumerate(keys)
        }
        partitions = {"fit": set(keys[:4]), "cal": set(keys[4:]), "probe": set(keys)}
        events = []
        model, predictions, ranking = fit_panel(
            rows, outcomes, partitions, "HGB_GBV_R", "synthetic", events.append
        )
        self.assertEqual(model["feature_dimension"], 7)
        self.assertEqual(model["target_counts"], {"fit": [2, 2], "cal": [2, 2]})
        self.assertEqual(len(predictions), 8)
        self.assertEqual([event["stage"] for event in events if event["event"] == "fit_completed"], ["base", "platt"])
        self.assertTrue(ranking["ranking_unchanged_except_ties"])


if __name__ == "__main__":
    unittest.main()
