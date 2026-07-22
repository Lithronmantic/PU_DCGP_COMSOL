
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.pu_dcgp import ManifestDataSource
from experiments.pu_dcgp_v23 import H8GaussianProcessMediatorSurrogate
from experiments.pu_dcgp_v25 import (
    FixedPlaneH8MediatorSurrogate,
    V25ExactGPComparisonPlan,
    a_group_free_jet_contract,
    a_group_free_jet_estimands,
    aps_ysz_a_free_jet_causal_graph,
    audit_a_group_free_jet_support,
    audit_a_free_jet_run_batch,
    audit_a_free_jet_physics_validity,
    candidate_one_factor_rules,
    make_a_free_jet_run_batch,
    validate_v25_evidence_manifest,
)


class AFreeJetContractTests(unittest.TestCase):
    @staticmethod
    def _v25_data_path(file_name: str) -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v25"
            / "data"
            / file_name
        )

    def test_contract_removes_nominal_distance_from_physical_treatments(self):
        contract = a_group_free_jet_contract()

        self.assertEqual(contract.measurement_position_mm, 100.0)
        self.assertEqual(
            contract.workpiece_status,
            "absent_during_DPV_acquisition",
        )
        self.assertNotIn(
            "spray_distance_mm",
            contract.physical_treatment_names,
        )
        self.assertFalse(contract.spray_distance_causal_estimand_allowed)

    def test_free_jet_estimands_contain_only_three_physical_treatments(self):
        estimands = a_group_free_jet_estimands()

        self.assertEqual(len(estimands), 3)
        self.assertEqual(
            {estimand.treatment_name for estimand in estimands},
            {"current_a", "argon_flow_scfh", "powder_feed_g_min"},
        )

    def test_free_jet_graph_has_no_nominal_distance_physical_path(self):
        graph = aps_ysz_a_free_jet_causal_graph()

        self.assertTrue(graph.is_acyclic())
        self.assertFalse(
            any(
                edge.source == "spray_distance_mm"
                for edge in graph.edges
            )
        )
        self.assertTrue(
            any(
                edge.source == "measurement_position_mm"
                and edge.target == "in_flight_particle_state_at_100_mm"
                for edge in graph.edges
            )
        )
        self.assertTrue(
            any(
                edge.source == "execution_order"
                and edge.target == "dpv_measurement_state"
                for edge in graph.edges
            )
        )

    def test_a_group_collapses_to_46_free_jet_settings(self):
        runs = ManifestDataSource(groups=("A",)).load()

        audit = audit_a_group_free_jet_support(runs)

        self.assertEqual(audit.run_count, 150)
        self.assertEqual(audit.nominal_four_factor_setting_count, 66)
        self.assertEqual(audit.physical_three_factor_setting_count, 46)
        self.assertEqual(audit.repeated_physical_setting_count, 23)
        self.assertEqual(audit.runs_in_repeated_physical_settings, 127)
        self.assertEqual(audit.maximum_physical_setting_repeats, 49)
        self.assertEqual(
            audit.physical_settings_with_multiple_spray_labels,
            17,
        )
        self.assertEqual(audit.measurement_positions_mm, (100.0,))

    def test_free_jet_validity_audit_supersedes_temperature_claim(self):
        runs = ManifestDataSource(groups=("A",)).load()

        audit = audit_a_free_jet_physics_validity(runs)
        temperature = next(
            item
            for item in audit.h8_fixed_plane_sensitivities
            if item.mediator == "particle_impact_temp_K"
        )

        self.assertEqual(temperature.changed_run_count, 76)
        self.assertGreater(temperature.mean_absolute_change, 40.0)
        self.assertEqual(
            temperature.validation_settings_changed_gt_one_unit,
            6,
        )
        self.assertEqual(
            audit.v24_issued_setting_correction.physical_issued_setting_count,
            7,
        )
        self.assertEqual(
            audit.claim_dispositions[
                "v24_temperature_multifidelity_gain"
            ],
            "superseded_pending_fixed_100mm_three_factor_rerun",
        )
        self.assertEqual(
            audit.claim_dispositions[
                "conditional_powder_to_detected_diameter_sign"
            ],
            "retained_with_nominal_distance_as_schedule_block",
        )

    def test_saved_validity_artifact_records_claim_dispositions(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v25"
            / "data"
            / "v25_a_free_jet_validity_audit.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        dispositions = payload["audit"]["claim_dispositions"]

        self.assertEqual(
            payload["schema"],
            "pu_dcgp_v25_a_free_jet_validity_audit_v1",
        )
        self.assertEqual(
            dispositions["spray_distance_to_DPV_outcome_effect"],
            "retired_noncausal_negative_control_only",
        )
        self.assertEqual(
            dispositions[
                "conditional_powder_to_detected_diameter_sign"
            ],
            "retained_with_nominal_distance_as_schedule_block",
        )

    def test_corrected_run_view_has_three_physical_inputs(self):
        historical = ManifestDataSource(groups=("A",)).load()

        corrected = make_a_free_jet_run_batch(historical)
        audit = audit_a_free_jet_run_batch(corrected)

        self.assertEqual(
            corrected.treatment_names,
            ("current_a", "argon_flow_scfh", "powder_feed_g_min"),
        )
        self.assertEqual(corrected.treatment_values.shape, (150, 3))
        self.assertEqual(audit.physical_setting_count, 46)
        self.assertFalse(audit.nominal_schedule_label_in_model_inputs)
        self.assertEqual(audit.measurement_positions_mm, (100.0,))
        self.assertEqual(corrected.run_ids, historical.run_ids)
        self.assertIs(
            corrected.particle_samples,
            historical.particle_samples,
        )

    def test_fixed_plane_adapter_matches_direct_h8_at_100_mm(self):
        historical = ManifestDataSource(groups=("A",)).load()
        corrected = make_a_free_jet_run_batch(historical)
        run_ids = corrected.run_ids[:4]
        physical = corrected.treatment_values[:4]
        direct = H8GaussianProcessMediatorSurrogate().fit()
        expected = direct.predict(
            run_ids,
            np.column_stack(
                (physical, np.full((len(run_ids), 1), 100.0))
            ),
        )

        actual = FixedPlaneH8MediatorSurrogate(
            delegate=direct,
        ).fit().predict(run_ids, physical)

        self.assertEqual(
            actual.factor_names,
            (
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
        )
        np.testing.assert_allclose(
            actual.factor_values[:, -1],
            100.0,
        )
        for mediator in expected.mediator_means:
            np.testing.assert_allclose(
                actual.mediator_means[mediator],
                expected.mediator_means[mediator],
            )
            np.testing.assert_allclose(
                actual.mediator_variances[mediator],
                expected.mediator_variances[mediator],
            )
        np.testing.assert_array_equal(
            actual.boundary_extrapolation,
            expected.boundary_extrapolation,
        )

    def test_exact_gp_plan_freezes_corrected_physical_domain(self):
        plan = V25ExactGPComparisonPlan()

        plan.validate()

        self.assertEqual(
            plan.grouping_unit,
            "exact_three_factor_physical_setting",
        )
        self.assertEqual(plan.fixed_h8_plane_mm, 100.0)
        self.assertEqual(
            plan.evidence_role,
            "a_reuse_development_diagnostic",
        )

    def test_nested_rule_library_uses_only_physical_factors(self):
        runs = make_a_free_jet_run_batch(
            ManifestDataSource(groups=("A",)).load()
        )

        rules = candidate_one_factor_rules(
            runs.treatment_names,
            runs.treatment_values,
        )

        self.assertEqual(len(rules), 39)
        self.assertFalse(
            any(
                "spray_distance_mm" in rule.label
                for rule in rules
            )
        )
        self.assertTrue(
            all(len(rule.clauses) == 1 for rule in rules)
        )

    def test_saved_corrected_comparison_is_negative_globally(self):
        payload = json.loads(
            self._v25_data_path(
                "v25_exact_gp_comparison.json"
            ).read_text(encoding="utf-8")
        )
        result = payload["result"]
        ratios = result["outcome_loss_ratios"][
            "multifidelity_exact_gp"
        ]

        self.assertEqual(result["physical_setting_count"], 46)
        self.assertTrue(result["all_folds_leakage_free"])
        self.assertTrue(result["all_predictions_finite"])
        self.assertGreater(ratios["temperature_c"], 1.0)
        self.assertAlmostEqual(
            ratios["velocity_m_s"],
            0.9995791319589565,
        )
        self.assertLess(
            result["setting_mean_gains"][
                "multifidelity_exact_gp"
            ],
            0.0,
        )

    def test_saved_nested_domains_fail_outer_reproduction(self):
        payload = json.loads(
            self._v25_data_path(
                "v25_nested_domain_audit.json"
            ).read_text(encoding="utf-8")
        )
        outcomes = {
            item["outcome"]: item
            for item in payload["audit"]["outcome_results"]
        }

        self.assertEqual(
            outcomes["temperature_c"][
                "issued_physical_setting_count"
            ],
            3,
        )
        self.assertLess(
            outcomes["temperature_c"]["mean_outer_setting_gain"],
            0.0,
        )
        self.assertEqual(
            outcomes["velocity_m_s"][
                "issued_physical_setting_count"
            ],
            2,
        )
        self.assertLess(
            outcomes["velocity_m_s"][
                "outer_gain_confidence_interval"
            ][1],
            0.0,
        )

    def test_saved_exchangeability_audit_records_block_boundary(self):
        payload = json.loads(
            self._v25_data_path(
                "v25_exchangeability_audit.json"
            ).read_text(encoding="utf-8")
        )
        audit = payload["audit"]
        feasibility = {
            (item["block_size"], item["nominal_coverage"]): item
            for item in audit["block_conformal_feasibility"]
        }

        self.assertTrue(
            audit[
                "single_setting_exchangeability_rejected_for_temperature"
            ]
        )
        self.assertFalse(
            audit["paired_block_exchangeability_justified"]
        )
        self.assertEqual(
            audit["current_a_certificate_status"],
            "abstain_no_independent_exchangeable_future_block",
        )
        self.assertTrue(
            feasibility[(2, 0.9)]["minimum_training_budget_met"]
        )
        self.assertFalse(
            feasibility[(2, 0.95)]["minimum_training_budget_met"]
        )

    def test_v25_evidence_manifest_hashes_validate(self):
        self.assertTrue(
            validate_v25_evidence_manifest(
                self._v25_data_path(
                    "v25_evidence_manifest.json"
                ).parent
            )
        )


if __name__ == "__main__":
    unittest.main()
