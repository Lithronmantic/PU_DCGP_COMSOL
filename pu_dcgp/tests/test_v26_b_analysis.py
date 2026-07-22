
import unittest
import json
from pathlib import Path

import numpy as np

from experiments.pu_dcgp_v26.b_group_auxiliary_analysis import (
    build_b_group_auxiliary_analysis,
    load_b_run_means,
)
from experiments.pu_dcgp_v26.b_group_qc import audit_b_group_data
from experiments.pu_dcgp_v26.comsol_b_observation_validation import (
    build_validation,
)


class BGroupAuxiliaryAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.frames = build_b_group_auxiliary_analysis()
        cls.comsol_b = build_validation()

    def test_outcome_blind_qc_retains_balanced_30_run_design(self):
        audit, flags = audit_b_group_data()

        self.assertEqual(audit.raw_run_count, 30)
        self.assertEqual(audit.factorial_cell_count, 10)
        self.assertEqual(audit.cell_replicates, (3,) * 10)
        self.assertEqual(audit.primary_included_run_count, 30)
        self.assertEqual(audit.primary_excluded_run_count, 0)
        self.assertTrue(audit.outcome_blind)
        self.assertTrue(all(row["primary_include"] for row in flags))

    def test_physical_coordinates_and_fixed_inputs_are_frozen(self):
        _, frame = load_b_run_means()

        self.assertEqual(len(frame), 30)
        self.assertEqual(
            tuple(sorted(frame["measurement_position_mm"].unique())),
            (90.0, 110.0),
        )
        self.assertEqual(
            tuple(sorted(frame["nominal_spray_distance_mm"].unique())),
            (80.0, 90.0, 100.0, 110.0, 120.0),
        )
        self.assertTrue(np.allclose(frame["current_a"], 700.0))
        self.assertTrue(np.allclose(frame["argon_flow_scfh"], 100.0))
        self.assertTrue(np.allclose(frame["powder_feed_g_min"], 20.0))

    def test_velocity_position_difference_has_frozen_monotonic_pattern(self):
        velocity = self.result["position_effects"]["velocity_m_s"]

        np.testing.assert_allclose(
            velocity["far_minus_near_by_nominal_distance"],
            (-7.9339511, 0.4940523, 2.8430307, 3.0014704, 5.6709805),
            atol=1e-6,
        )
        self.assertAlmostEqual(velocity["distance_difference_spearman_rho"], 1.0)
        self.assertAlmostEqual(velocity["exact_permutation_p"], 1.0 / 60.0)
        self.assertGreater(velocity["slope_bootstrap_95_interval"][0], 0.0)
        self.assertTrue(velocity["leave_one_slope_sign_consistent"])

    def test_hc3_sequence_adjusted_result_is_reported_conservatively(self):
        model = self.result["sequence_adjusted_models"]["velocity_m_s"]

        self.assertEqual(model["categorical_design_rank"], 11)
        self.assertEqual(model["categorical_parameter_count"], 11)
        self.assertLess(model["categorical_hc3_anova_p"]["block_by_position"], 0.05)
        self.assertGreater(
            model["categorical_hc3_anova_p"]["block_by_position"],
            model["sequence_adjusted_classical_anova_p"]["block_by_position"],
        )

    def test_raw_prt_sensitivity_is_retained_without_deletion(self):
        sensitivity = self.result["raw_prt_velocity_sensitivity"]

        self.assertEqual(
            sensitivity["largest_source_discrepancy_run_id"], "UCE-R131-B1"
        )
        self.assertGreater(sensitivity["largest_source_discrepancy_m_s"], 39.0)
        self.assertEqual(
            sensitivity["conclusion"],
            "robust_rank_trend_with_raw_mean_export_sensitivity",
        )
        self.assertEqual(len(self.frames["velocity_source_comparison"]), 30)

    def test_claim_stops_before_workpiece_distance_causality(self):
        self.assertFalse(self.result["physical_coordinates"]["workpiece_present"])
        self.assertIn("not a workpiece spray-distance causal effect", self.result["claim_boundary"])

    def test_comsol_two_plane_extraction_passes_numerical_gates(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "simulator_v2"
            / "phase_h"
            / "h11_outputs"
            / "b_observation_planes"
            / "h11_b_observation_plane_extraction.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "pass_two_plane_comsol_extraction")
        self.assertTrue(all(payload["numerical_gates"].values()))
        self.assertEqual([plane["crossing_particle_count"] for plane in payload["planes"]], [7161, 7161])
        self.assertFalse(payload["calibrated_on_b"])

    def test_current_comsol_fails_all_frozen_b_consistency_channels(self):
        audit = self.comsol_b

        self.assertEqual(audit["status"], "fail_current_comsol_for_B_external_consistency")
        self.assertEqual(audit["admitted_outcomes"], [])
        self.assertAlmostEqual(
            audit["outcomes"]["temperature_c"]["simulated_far_minus_near"],
            -49.525154742381346,
        )
        self.assertAlmostEqual(
            audit["outcomes"]["velocity_m_s"]["simulated_far_minus_near"],
            -4.17708791607518,
        )
        self.assertAlmostEqual(
            audit["outcomes"]["particle_diameter_um"]["simulated_far_minus_near"],
            0.022400687153869114,
        )
        for values in audit["outcomes"].values():
            self.assertEqual(values["status"], "fail_B_external_consistency")

    def test_b_absolute_scale_is_not_used_to_tune_a_exit_state(self):
        comparison = self.comsol_b["a_b_campaign_comparison"]["velocity_m_s"]

        self.assertEqual(comparison["a_same_process_run_count"], 49)
        self.assertEqual(comparison["a_same_process_distance_100_run_count"], 27)
        self.assertLess(comparison["b_vs_a_same_process_relative_shift"], -0.20)
        assessment = self.comsol_b["a_b_observation_configuration_assessment"]
        self.assertEqual(
            assessment["decision"],
            "B_absolute_scale_not_admitted_for_A_exit_calibration",
        )
        self.assertIn("90/110", assessment["position_adjustment"])
        self.assertIn(
            "not_supported",
            assessment["position_only_explanation"],
        )
        self.assertFalse(self.comsol_b["calibrated_on_b"])


if __name__ == "__main__":
    unittest.main()
