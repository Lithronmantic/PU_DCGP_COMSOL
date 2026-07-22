"""Contract tests for PU-DCGP v2.4 diagnostics."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.pu_dcgp import ManifestDataSource, a_group_doe_estimands
from experiments.pu_dcgp_v24 import (
    BGroupDesignAudit,
    ProcessRule,
    VelocitySubdomainSpec,
    V24EvidenceRole,
    V24ExactGPComparisonPlan,
    audit_b_velocity_source_sensitivity,
    build_b_guided_multidiameter_plan,
    H1aDiameterSmokeResult,
    H1aMultiDiameterSmokeAudit,
    CenterlineVelocityField,
    integrate_particle_states,
    load_centerline_velocity_field,
    run_nonuniform_velocity_observation_audit,
    run_v24_exact_gp_fold_smoke,
    audit_conditional_causal_sign,
)
from experiments.pu_dcgp_v24.exact_gp_comparison import (
    _balanced_setting_folds,
)


class V24ExactGPComparisonContractTests(unittest.TestCase):
    def test_a_reuse_is_explicitly_development_only(self):
        plan = V24ExactGPComparisonPlan()

        plan.validate()
        self.assertIs(
            plan.evidence_role,
            V24EvidenceRole.DEVELOPMENT_DIAGNOSTIC,
        )
        self.assertEqual(
            plan.grouping_unit,
            "exact_four_factor_setting",
        )
        self.assertEqual(
            plan.support_gate_source,
            "frozen_h8_factor_envelope",
        )
        self.assertFalse(plan.outer_targets_used_by_support_gate)

    def test_support_gate_cannot_use_outer_targets(self):
        plan = replace(
            V24ExactGPComparisonPlan(),
            outer_targets_used_by_support_gate=True,
        )

        with self.assertRaisesRegex(ValueError, "outer targets"):
            plan.validate()

    def test_outer_folds_balance_settings_not_repeated_runs(self):
        treatments = np.array(
            [[0.0]] * 20
            + [[1.0], [2.0], [3.0], [4.0], [5.0]],
            dtype=float,
        )

        folds = _balanced_setting_folds(treatments, 2, 74026)
        test_setting_counts = [
            len(np.unique(treatments[indices], axis=0))
            for indices in folds
        ]

        self.assertLessEqual(
            max(test_setting_counts) - min(test_setting_counts),
            1,
        )
        self.assertEqual(
            set(np.concatenate(folds)),
            set(range(len(treatments))),
        )

    def test_process_rule_uses_inputs_only(self):
        rule = ProcessRule(
            (
                ("current_a", "==", 700.0),
                ("powder_feed_g_min", "<=", 20.0),
            )
        )
        mask = rule.mask(
            ("current_a", "powder_feed_g_min"),
            np.array(
                [[700.0, 15.0], [700.0, 25.0], [750.0, 15.0]]
            ),
        )

        np.testing.assert_array_equal(
            mask,
            np.array([True, False, False]),
        )

    def test_velocity_subdomain_contract_imports(self):
        domain = VelocitySubdomainSpec(
            name="h8_interior_powder_le20",
            h8_region="interior",
            factor_name="powder_feed_g_min",
            operator="<=",
            value=20.0,
        )

        self.assertEqual(domain.h8_region, "interior")
        self.assertEqual(domain.operator, "<=")

    def test_b_group_observation_contract_imports(self):
        audit = BGroupDesignAudit(
            run_count=30,
            setting_count=5,
            spray_distances_mm=(80.0, 90.0, 100.0, 110.0, 120.0),
            measurement_positions_mm=(90.0, 110.0),
            runs_per_cell=(3,) * 10,
        )

        self.assertEqual(audit.run_count, 30)
        self.assertEqual(set(audit.runs_per_cell), {3})

    def test_b_guided_multidiameter_plan_is_diagnostic_only(self):
        plan = build_b_guided_multidiameter_plan(
            particle_diameters_um=(30.0, 40.0, 50.0),
            diameter_basis="provisional_sensitivity_probes",
        )

        self.assertEqual(len(plan.cells), 6)
        self.assertTrue(plan.ready_to_execute)
        self.assertFalse(plan.ready_for_population_fusion)
        self.assertFalse(plan.may_train_main_a_model)
        self.assertEqual(
            plan.nominal_spray_distance_role,
            "experimental_block_only_not_physics_input",
        )
        self.assertEqual(
            {
                cell.measurement_position_label_mm
                for cell in plan.cells
            },
            {90.0, 110.0},
        )

    def test_saved_b_plan_is_free_jet_six_output_contract(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v24"
            / "data"
            / "v24_b_guided_multidiameter_plan.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = payload["plan"]

        self.assertEqual(
            payload["schema"],
            "pu_dcgp_v24_b_guided_free_jet_plan_v2",
        )
        self.assertEqual(len(plan["cells"]), 6)
        self.assertTrue(plan["ready_to_execute"])
        self.assertFalse(plan["ready_for_population_fusion"])
        self.assertTrue(
            all(
                "spray_distance_mm" not in cell
                for cell in plan["cells"]
            )
        )

    def test_b_velocity_source_sensitivity_preserves_raw_anomaly(self):
        runs = ManifestDataSource(groups=("B",)).load()

        audit = audit_b_velocity_source_sensitivity(runs)

        self.assertEqual(
            audit.largest_source_discrepancy_run_id,
            "UCE-R131-B1",
        )
        self.assertEqual(
            audit.conclusion,
            "robust_interaction_with_documented_raw_export_anomaly",
        )
        raw_mean, raw_median = audit.source_summaries[1:]
        self.assertLess(
            raw_mean.distance_difference_spearman_rho,
            0.8,
        )
        self.assertGreaterEqual(
            raw_median.distance_difference_spearman_rho,
            0.8,
        )

    def test_h1a_multidiameter_smoke_contract_imports(self):
        result = H1aDiameterSmokeResult(
            particle_diameter_um=40.0,
            particle_count=60,
            hit_count=60,
            wall_hit_fraction=1.0,
            impact_velocity_m_s_mean=20.0,
            impact_velocity_m_s_median=20.0,
            residence_time_us_median=5000.0,
            runtime_sec=1.0,
        )
        audit = H1aMultiDiameterSmokeAudit(
            evidence_role="COMSOL_H1a_representation_smoke_only",
            spray_distance_mm=100.0,
            inlet_velocity_m_s=20.0,
            results=(result,),
            impact_velocity_range_m_s=0.0,
            diameter_signal_present=False,
            diagnosis="uniform_coflow_eliminates_diameter_velocity_signal",
        )

        self.assertFalse(audit.diameter_signal_present)
        self.assertEqual(audit.results[0].particle_diameter_um, 40.0)

    def test_h1a_multidiameter_smoke_artifact_records_zero_signal(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v24"
            / "data"
            / "v24_h1a_multidiameter_smoke.json"
        )
        audit = json.loads(path.read_text(encoding="utf-8"))["audit"]

        self.assertFalse(audit["diameter_signal_present"])
        self.assertEqual(audit["impact_velocity_range_m_s"], 0.0)
        self.assertEqual(
            {
                result["particle_diameter_um"]
                for result in audit["results"]
            },
            {30.0, 40.0, 50.0},
        )

    def test_nonuniform_velocity_field_skeleton_loads_l5(self):
        field = load_centerline_velocity_field("L5_komega")

        self.assertEqual(field.model_tag, "L5_komega")
        self.assertEqual(field.z_m.shape, field.gas_velocity_m_s.shape)
        self.assertLess(field.z_m[0], 0.09)
        self.assertGreater(field.z_m[-1], 0.11)
        self.assertGreater(
            float(np.interp(0.09, field.z_m, field.gas_velocity_m_s)),
            float(np.interp(0.11, field.z_m, field.gas_velocity_m_s)),
        )

    def test_particle_integrator_preserves_uniform_coflow(self):
        field = CenterlineVelocityField(
            model_tag="test_uniform",
            source_path="synthetic_test_field",
            z_m=np.array([0.0, 0.12]),
            gas_velocity_m_s=np.array([20.0, 20.0]),
            nozzle_velocity_m_s=20.0,
            effective_gas_density_kg_m3=0.6,
            effective_gas_viscosity_pa_s=1e-3,
        )

        states = integrate_particle_states(
            field,
            particle_diameters_um=(30.0, 40.0, 50.0),
            measurement_positions_mm=(90.0, 110.0),
            injection_velocity_m_s=20.0,
        )

        self.assertEqual(len(states), 6)
        self.assertTrue(
            np.allclose(
                [state.particle_velocity_m_s for state in states],
                20.0,
            )
        )

    def test_nonuniform_velocity_audit_recovers_diameter_signal(self):
        audit = run_nonuniform_velocity_observation_audit()

        self.assertEqual(len(audit.field_audits), 2)
        self.assertTrue(
            audit.representation_gates[
                "diameter_velocity_signal_resolved"
            ]
        )
        self.assertTrue(
            audit.representation_gates[
                "L4_L5_plane_effect_sign_agreement"
            ]
        )
        self.assertFalse(
            audit.representation_gates[
                "B_distance_by_position_interaction_represented"
            ]
        )
        for field_audit in audit.field_audits:
            self.assertLess(
                field_audit.step_refinement_max_abs_difference_m_s,
                1e-4,
            )

    def test_nonuniform_velocity_artifact_records_free_jet_contract(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v24"
            / "data"
            / "v24_nonuniform_velocity_observation.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = payload["audit"]

        self.assertEqual(
            payload["schema"],
            "pu_dcgp_v24_nonuniform_velocity_observation_v2",
        )
        self.assertTrue(
            audit["representation_gates"][
                "diameter_velocity_signal_resolved"
            ]
        )
        self.assertFalse(
            audit["representation_gates"][
                "nominal_spray_distance_is_physical_free_jet_input"
            ]
        )
        self.assertTrue(
            audit["representation_gates"][
                "workpiece_absent_during_B_DPV_acquisition"
            ]
        )


class V24ExactGPFoldSmokeTests(unittest.TestCase):
    def test_gpu_paths_share_one_leakage_free_fold(self):
        audit = run_v24_exact_gp_fold_smoke()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.backend, "gpu")
        self.assertEqual(audit.setting_overlap_count, 0)
        self.assertTrue(audit.all_predictions_finite)
        self.assertTrue(audit.unrepresented_prediction_identical)
        self.assertEqual(
            set(audit.data_only_structures),
            {
                "temperature_c",
                "velocity_m_s",
                "particle_diameter_um",
            },
        )

    def test_conditional_powder_diameter_sign_audit(self):
        runs = ManifestDataSource(groups=("A",)).load()
        estimand = next(
            item
            for item in a_group_doe_estimands()
            if item.estimand_id == "powder_10_to_30"
        )

        audit = audit_conditional_causal_sign(
            runs,
            estimand,
            "particle_diameter_um",
            direction="negative",
            alpha=0.1,
        )

        self.assertEqual(audit.matched_strata, 9)
        self.assertEqual(audit.direction_consistent_strata, 7)
        self.assertAlmostEqual(
            audit.one_sided_exact_sign_p,
            46 / 512,
        )
        self.assertTrue(audit.finite_sample_sign_statement_issued)


if __name__ == "__main__":
    unittest.main()
