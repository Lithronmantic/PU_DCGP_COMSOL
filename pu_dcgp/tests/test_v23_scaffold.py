
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from experiments.pu_dcgp import ManifestDataSource
from experiments.pu_dcgp.contracts import (
    DistributionRepresentation,
    RunBatch,
)
from experiments.pu_dcgp_v2.contracts import (
    FullScoreUncertaintyRepresentation,
    JointScorePrediction,
    PreparedV2Data,
)
from experiments.pu_dcgp_v21 import PUDCGPV21Config
from experiments.pu_dcgp_v23 import (
    AffinePhysicsScoreObservationOperator,
    H8GaussianProcessMediatorSurrogate,
    LowFidelityMediatorPrediction,
    PhysicsOutcomeStatus,
    RidgeScoreDiscrepancyModel,
    SelectedMultifidelityDistributionModel,
    align_comsol_mediators_to_dpv_units,
    audit_h8_low_fidelity_surrogate,
    audit_v23_observation_operator,
    finalize_v23_multifidelity_evidence,
    run_v23_multifidelity_support_pilot,
    run_v23_a_group_model_smoke,
    run_v23_synthetic_multifidelity_benchmark,
    validate_v23_multifidelity_evidence_release,
    v23_mechanism_enriched_plan,
    v23_nested_mechanism_plan,
    v23_outcome_selective_plan,
    v23_outcome_contracts,
)


class V23ObservationOperatorTests(unittest.TestCase):
    def test_outcome_contract_excludes_fixed_diameter(self):
        contracts = {
            contract.outcome: contract
            for contract in v23_outcome_contracts()
        }

        self.assertIs(
            contracts["temperature_c"].status,
            PhysicsOutcomeStatus.SUPPORTED_WITH_DISCREPANCY,
        )
        self.assertIs(
            contracts["velocity_m_s"].status,
            PhysicsOutcomeStatus.SUPPORTED_WITH_DISCREPANCY,
        )
        self.assertIs(
            contracts["particle_diameter_um"].status,
            PhysicsOutcomeStatus.UNREPRESENTED,
        )
        self.assertIsNone(contracts["particle_diameter_um"].mediator)

    def test_unit_operator_maps_temperature_and_velocity_only(self):
        prediction = LowFidelityMediatorPrediction(
            run_ids=("run-1", "run-2"),
            factor_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            factor_values=np.asarray(
                [[700.0, 100.0, 20.0, 100.0]] * 2
            ),
            mediator_means={
                "particle_impact_temp_K": np.asarray([3000.0, 3100.0]),
                "particle_impact_velocity_m_s": np.asarray([20.0, 22.0]),
            },
            mediator_variances={
                "particle_impact_temp_K": np.asarray([100.0, 121.0]),
                "particle_impact_velocity_m_s": np.asarray([1.0, 1.44]),
            },
            boundary_extrapolation=np.asarray([False, True]),
        )

        aligned = align_comsol_mediators_to_dpv_units(prediction)

        np.testing.assert_allclose(
            aligned.outcome_means["temperature_c"],
            [2726.85, 2826.85],
        )
        np.testing.assert_allclose(
            aligned.outcome_means["velocity_m_s"],
            [20.0, 22.0],
        )
        self.assertNotIn(
            "particle_diameter_um",
            aligned.outcome_means,
        )
        np.testing.assert_array_equal(
            aligned.boundary_extrapolation,
            [False, True],
        )

    def test_real_a_h8_sources_are_ready_for_discrepancy_fit(self):
        runs = ManifestDataSource(groups=("A",)).load()

        audit = audit_v23_observation_operator(runs)

        self.assertTrue(audit.ready_for_discrepancy_fit)
        self.assertEqual(audit.low_fidelity_row_count, 96)
        self.assertEqual(audit.a_run_count, 150)
        self.assertEqual(audit.a_setting_count, 66)
        self.assertEqual(
            tuple(item.factor for item in audit.factor_support),
            runs.treatment_names,
        )
        self.assertTrue(
            all(item.outside_run_count > 0 for item in audit.factor_support)
        )
        self.assertTrue(
            all(len(digest) == 64 for digest in (
                audit.low_fidelity_sha256,
                audit.heat_model_sha256,
                audit.particle_model_sha256,
            ))
        )

    def test_affine_score_operator_maps_supported_scores_and_variance(self):
        run_ids = tuple(f"run-{index}" for index in range(6))
        mediator_temperature = np.linspace(2800.0, 3300.0, 6)
        mediator_velocity = np.linspace(18.0, 24.0, 6)
        runs = RunBatch(
            run_ids=run_ids,
            groups=("A",) * 6,
            doe_modules=("test",) * 6,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.tile(
                [700.0, 100.0, 20.0, 100.0],
                (6, 1),
            ),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (6, 1)),
            context_names=(
                "execution_order",
                "measurement_position_mm",
            ),
            context_values=np.zeros((6, 2)),
            particle_samples={
                outcome: tuple(np.ones(3) for _ in run_ids)
                for outcome in (
                    "temperature_c",
                    "velocity_m_s",
                    "particle_diameter_um",
                )
            },
        )
        distributions = DistributionRepresentation(
            run_ids=run_ids,
            outcome_names=(
                "temperature_c",
                "velocity_m_s",
                "particle_diameter_um",
            ),
            quantile_grid=np.asarray([0.25, 0.75]),
            scores={
                "temperature_c": np.column_stack(
                    (
                        1.0 + 0.01 * mediator_temperature,
                        -2.0 + 0.02 * mediator_temperature,
                    )
                ),
                "velocity_m_s": (
                    3.0 + 0.5 * mediator_velocity
                )[:, None],
                "particle_diameter_um": np.zeros((6, 2)),
            },
            score_variances={
                "temperature_c": np.zeros((6, 2)),
                "velocity_m_s": np.zeros((6, 1)),
                "particle_diameter_um": np.zeros((6, 2)),
            },
        )
        low_fidelity = LowFidelityMediatorPrediction(
            run_ids=run_ids,
            factor_names=runs.treatment_names,
            factor_values=runs.treatment_values,
            mediator_means={
                "particle_impact_temp_K": mediator_temperature,
                "particle_impact_velocity_m_s": mediator_velocity,
            },
            mediator_variances={
                "particle_impact_temp_K": np.full(6, 4.0),
                "particle_impact_velocity_m_s": np.full(6, 0.25),
            },
            boundary_extrapolation=np.zeros(6, dtype=bool),
        )

        prior = AffinePhysicsScoreObservationOperator(
            ridge_alpha=1e-8
        ).fit(runs, distributions, low_fidelity).transform(low_fidelity)

        self.assertEqual(
            prior.supported_outcomes,
            ("temperature_c", "velocity_m_s"),
        )
        self.assertEqual(
            prior.unrepresented_outcomes,
            ("particle_diameter_um",),
        )
        np.testing.assert_allclose(
            prior.score_means["temperature_c"],
            distributions.scores["temperature_c"],
            rtol=1e-7,
        )
        self.assertTrue(
            np.all(prior.score_variances["temperature_c"] > 0.0)
        )
        fused = RidgeScoreDiscrepancyModel(
            ridge_alpha=1e-8
        ).fit(
            runs,
            distributions,
            prior,
        ).predict(runs, prior)
        np.testing.assert_allclose(
            fused.means["temperature_c"],
            distributions.scores["temperature_c"],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            fused.means["particle_diameter_um"],
            distributions.scores["particle_diameter_um"],
            atol=1e-12,
        )
        self.assertEqual(
            fused.data_only_outcomes,
            ("particle_diameter_um",),
        )

    def test_selected_exact_gp_wrapper_residualizes_supported_only(self):
        run_ids = tuple(f"train-{index}" for index in range(5))
        treatment_values = np.asarray(
            [
                [600.0 + 50.0 * index, 80.0 + 10.0 * index, 20.0, 100.0]
                for index in range(5)
            ]
        )
        runs = RunBatch(
            run_ids=run_ids,
            groups=("A",) * 5,
            doe_modules=("test",) * 5,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=treatment_values,
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (5, 1)),
            context_names=(
                "execution_order",
                "measurement_position_mm",
            ),
            context_values=np.zeros((5, 2)),
            particle_samples={
                outcome: tuple(np.ones(3) for _ in run_ids)
                for outcome in (
                    "temperature_c",
                    "velocity_m_s",
                    "particle_diameter_um",
                )
            },
        )
        scores = {
            "temperature_c": (0.01 * treatment_values[:, 0])[:, None],
            "velocity_m_s": (0.1 * treatment_values[:, 1])[:, None],
            "particle_diameter_um": np.column_stack(
                (
                    np.linspace(0.0, 1.0, 5),
                    np.linspace(1.0, 0.0, 5),
                )
            ),
        }
        covariances = {
            outcome: np.tile(
                np.eye(values.shape[1])[None, :, :] * 0.01,
                (5, 1, 1),
            )
            for outcome, values in scores.items()
        }
        representation = FullScoreUncertaintyRepresentation(
            run_ids=run_ids,
            outcome_names=tuple(scores),
            quantile_grid=np.asarray([0.25, 0.75]),
            scores=scores,
            score_covariances=covariances,
        )

        class FakeLowFidelity:
            def fit(self):
                return self

            def predict(self, requested_ids, factors):
                return LowFidelityMediatorPrediction(
                    run_ids=requested_ids,
                    factor_names=runs.treatment_names,
                    factor_values=factors,
                    mediator_means={
                        "particle_impact_temp_K": factors[:, 0],
                        "particle_impact_velocity_m_s": factors[:, 1],
                    },
                    mediator_variances={
                        "particle_impact_temp_K": np.full(len(factors), 0.5),
                        "particle_impact_velocity_m_s": np.full(len(factors), 0.2),
                    },
                    boundary_extrapolation=np.zeros(len(factors), dtype=bool),
                )

        class FakeResidualModel:
            def fit(self, data):
                self.data = data

            def predict(self, treatments, contexts):
                count = len(treatments)
                means = {
                    outcome: np.zeros(
                        (count, values.shape[1])
                    )
                    for outcome, values
                    in self.data.distributions.scores.items()
                }
                covariances = {
                    outcome: np.zeros(
                        (
                            count,
                            values.shape[1],
                            count,
                            values.shape[1],
                        )
                    )
                    for outcome, values
                    in self.data.distributions.scores.items()
                }
                return JointScorePrediction(means, covariances)

        residual_model = FakeResidualModel()
        model = SelectedMultifidelityDistributionModel(
            PUDCGPV21Config(),
            {
                "temperature_c": ("particle_impact_temp_K",),
                "velocity_m_s": ("particle_impact_velocity_m_s",),
            },
            low_fidelity_surrogate=FakeLowFidelity(),
            observation_operator=AffinePhysicsScoreObservationOperator(
                ridge_alpha=1e-8
            ),
            residual_model=residual_model,
        )
        model.fit(PreparedV2Data(runs, representation))

        np.testing.assert_array_equal(
            residual_model.data.distributions.scores[
                "particle_diameter_um"
            ],
            scores["particle_diameter_um"],
        )
        self.assertLess(
            np.max(
                np.abs(
                    residual_model.data.distributions.scores[
                        "temperature_c"
                    ]
                )
            ),
            1e-7,
        )
        prediction = model.predict(
            treatment_values[:2],
            np.zeros((2, 2)),
        )
        self.assertTrue(
            np.all(
                np.diagonal(
                    prediction.covariances["temperature_c"],
                    axis1=0,
                    axis2=2,
                )
                >= 0.0
            )
        )
        np.testing.assert_array_equal(
            prediction.means["particle_diameter_um"],
            np.zeros((2, 2)),
        )


class V23LowFidelitySurrogateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = ManifestDataSource(groups=("A",)).load()

    def test_surrogate_returns_aligned_moments_and_boundary_flags(self):
        prediction = H8GaussianProcessMediatorSurrogate().fit().predict(
            self.runs.run_ids,
            self.runs.treatment_values,
        )

        self.assertEqual(prediction.factor_names, self.runs.treatment_names)
        self.assertEqual(
            set(prediction.mediator_means),
            {
                "particle_impact_temp_K",
                "particle_impact_velocity_m_s",
            },
        )
        self.assertEqual(
            int(np.sum(prediction.boundary_extrapolation)),
            93,
        )
        for mediator in prediction.mediator_means:
            self.assertTrue(
                np.all(np.isfinite(prediction.mediator_means[mediator]))
            )
            self.assertTrue(
                np.all(prediction.mediator_variances[mediator] >= 0.0)
            )

    def test_surrogate_cv_is_reliable_on_frozen_h8_rows(self):
        audit = audit_h8_low_fidelity_surrogate(
            self.runs.run_ids,
            self.runs.treatment_values,
        )
        r2 = {
            item.mediator: item.cv_r2 for item in audit.target_audits
        }

        self.assertEqual(audit.row_count, 96)
        self.assertEqual(audit.a_boundary_run_count, 93)
        self.assertEqual(audit.a_interior_run_count, 57)
        self.assertGreater(r2["particle_impact_temp_K"], 0.8)
        self.assertGreater(r2["particle_impact_velocity_m_s"], 0.99)


class V23MultifidelitySupportPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = ManifestDataSource(groups=("A",)).load()
        cls.result, cls.predictions = (
            run_v23_multifidelity_support_pilot(cls.runs)
        )

    def test_repeated_settings_never_cross_folds(self):
        for setting_group in np.unique(self.predictions.setting_groups):
            folds = np.unique(
                self.predictions.fold_indices[
                    self.predictions.setting_groups == setting_group
                ]
            )
            self.assertEqual(len(folds), 1)

    def test_unrepresented_diameter_path_is_exactly_data_only(self):
        self.assertTrue(self.result.diameter_path_identical)
        np.testing.assert_array_equal(
            self.predictions.multifidelity["particle_diameter_um"],
            self.predictions.data_only["particle_diameter_um"],
        )

    def test_two_scalar_mediator_gate_fails_without_overclaim(self):
        self.assertFalse(self.result.passed)
        self.assertLess(self.result.aggregate_loss_reduction, 0.0)
        self.assertIn(
            "aggregate_loss_not_reduced",
            self.result.reasons,
        )
        self.assertIn(
            "cluster_bootstrap_lower_bound_not_positive",
            self.result.reasons,
        )

    def test_mechanism_enriched_fixed_residual_gate_also_fails(self):
        plan = v23_mechanism_enriched_plan()
        result, predictions = run_v23_multifidelity_support_pilot(
            self.runs,
            plan,
        )

        self.assertEqual(plan.pilot_id, "mechanism_v2")
        self.assertFalse(result.passed)
        self.assertLess(result.aggregate_loss_reduction, 0.0)
        self.assertTrue(result.diameter_path_identical)
        for setting_group in np.unique(predictions.setting_groups):
            self.assertEqual(
                len(
                    np.unique(
                        predictions.fold_indices[
                            predictions.setting_groups == setting_group
                        ]
                    )
                ),
                1,
            )

    def test_nested_and_outcome_selective_variants_do_not_rescue_a(self):
        nested_result, _ = run_v23_multifidelity_support_pilot(
            self.runs,
            v23_nested_mechanism_plan(),
        )
        selective_result, selective_predictions = (
            run_v23_multifidelity_support_pilot(
                self.runs,
                v23_outcome_selective_plan(),
            )
        )

        self.assertFalse(nested_result.passed)
        self.assertFalse(selective_result.passed)
        self.assertTrue(selective_result.diameter_path_identical)
        selected_paths = {
            selection.selected_path
            for selection
            in selective_predictions.hyperparameter_selections
        }
        self.assertEqual(selected_paths, {"data_only", "multifidelity"})


class V23KnownTruthMultifidelityTests(unittest.TestCase):
    def test_concrete_score_fusion_improves_when_physics_is_informative(self):
        result = run_v23_synthetic_multifidelity_benchmark()

        self.assertTrue(result.passed)
        self.assertEqual(result.replicate_count, 20)
        self.assertGreater(result.mean_loss_reduction_fraction, 0.20)
        self.assertGreaterEqual(result.replicate_pass_rate, 0.9)
        self.assertTrue(result.diameter_path_identical)
        self.assertLess(
            result.mean_multifidelity_loss,
            result.mean_physics_only_loss,
        )
        self.assertLess(
            result.mean_physics_only_loss,
            result.mean_data_only_loss,
        )
        self.assertEqual(result.outer_orbit_coverage_rate, 0.95)
        self.assertEqual(result.outer_orbit_miscovered_count, 1)
        self.assertTrue(result.outer_orbit_passed)

    def test_real_a_group_exact_gp_wrapper_is_numerically_valid(self):
        audit = run_v23_a_group_model_smoke()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.run_count, 150)
        self.assertEqual(audit.setting_count, 66)
        self.assertTrue(audit.all_predictions_finite)
        self.assertTrue(audit.diameter_residual_scores_identical)
        self.assertEqual(
            set(audit.selected_structures),
            {
                "temperature_c",
                "velocity_m_s",
                "particle_diameter_um",
            },
        )


class V23EvidenceReleaseTests(unittest.TestCase):
    DATA_DIRECTORY = (
        Path(__file__).resolve().parents[2] / "pu_dcgp_v23" / "data"
    )

    def test_release_seals_positive_algorithm_and_negative_a_evidence(self):
        release = validate_v23_multifidelity_evidence_release(
            self.DATA_DIRECTORY
        )

        self.assertEqual(len(release.artifact_sha256), 13)
        self.assertEqual(release.low_fidelity_row_count, 96)
        self.assertEqual(release.synthetic_outer_orbit_coverage, 0.95)
        self.assertTrue(release.synthetic_algorithm_passed)
        self.assertTrue(release.a_group_exact_gp_smoke_passed)
        self.assertEqual(
            release.application_status,
            "implemented_but_not_admitted_for_a_group",
        )
        self.assertEqual(
            release.a_group_pilot_decisions,
            (
                ("scalar_v1", False),
                ("mechanism_v2", False),
                ("mechanism_nested_v3", False),
                ("outcome_selective_v4", False),
            ),
        )

    def test_release_detects_artifact_byte_change(self):
        source = validate_v23_multifidelity_evidence_release(
            self.DATA_DIRECTORY
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            for name, _ in source.artifact_sha256:
                shutil.copy2(self.DATA_DIRECTORY / name, destination / name)
            finalized = finalize_v23_multifidelity_evidence(destination)
            self.assertEqual(finalized.artifact_sha256, source.artifact_sha256)

            changed = destination / source.artifact_sha256[0][0]
            changed.write_bytes(changed.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_v23_multifidelity_evidence_release(destination)


if __name__ == "__main__":
    unittest.main()
