
from pathlib import Path
import unittest

from experiments.pu_dcgp_v2.benchmark_runner import (
    aggregate_v2_benchmark_records,
)
from experiments.pu_dcgp_v21 import (
    PUDCGPV21Config,
    build_v21_development_summary,
    evaluate_v21_development_hypotheses,
    load_v21_dataset_checkpoints,
    pu_dcgp_v21_development_contract,
)


class V21DevelopmentHypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = pu_dcgp_v21_development_contract()
        cls.config = PUDCGPV21Config(
            quantile_grid=cls.contract.quantile_grid
        )
        checkpoint_path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v21"
            / "data"
            / "development_v21_records.jsonl"
        )
        checkpoints = load_v21_dataset_checkpoints(
            checkpoint_path,
            cls.contract,
            cls.config,
        )
        records = tuple(
            record
            for checkpoint in checkpoints
            for record in checkpoint.records
        )
        cls.decisions = evaluate_v21_development_hypotheses(
            aggregate_v2_benchmark_records(records),
            cls.contract,
            cls.config,
        )
        cls.summary = build_v21_development_summary(
            checkpoints,
            cls.contract,
            cls.config,
        )

    def test_supported_development_meets_s1_h2_and_h4(self):
        by_id = {
            decision.hypothesis_id: decision
            for decision in self.decisions
        }

        for hypothesis_id in ("S1", "H2-v2", "H4-v2"):
            self.assertTrue(by_id[hypothesis_id].threshold_met)
            self.assertEqual(
                by_id[hypothesis_id].status,
                "development_threshold_met",
            )
        self.assertLessEqual(
            by_id["S1"].evidence[
                "maximum_calibrated_to_diagonal_shape_irmse_ratio"
            ],
            self.config.shape_non_regression_ratio,
        )
        self.assertGreaterEqual(
            by_id["H4-v2"].evidence[
                "supported_active_existence_power"
            ],
            self.config.active_existence_power_min,
        )

    def test_unrun_failure_scenarios_keep_h3_not_evaluable(self):
        h3 = next(
            decision
            for decision in self.decisions
            if decision.hypothesis_id == "H3-v2"
        )

        self.assertIsNone(h3.threshold_met)
        self.assertEqual(h3.status, "not_evaluable")

    def test_summary_carries_complete_integrity_and_structure_audit(self):
        self.assertTrue(self.summary["integrity"]["integrity_passed"])
        self.assertEqual(
            self.summary["integrity"]["dataset_count"],
            120,
        )
        self.assertEqual(
            self.summary["integrity"]["method_record_count"],
            960,
        )
        self.assertEqual(
            self.summary["structure_selection_counts"],
            {"diagonal": 2160},
        )
