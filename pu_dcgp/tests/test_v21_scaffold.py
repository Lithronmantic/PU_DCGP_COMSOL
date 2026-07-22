
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pu_dcgp_v21 import (
    BICCoregionalizationStructureSelector,
    CoregionalizationCandidateFit,
    CoregionalizationCandidateEvidence,
    CoregionalizationSelectionDecision,
    CoregionalizationStructureSelector,
    DiagonalTraceNormalizedICMParameterization,
    CoregionalizationCandidateFitter,
    FoldStructureSelectionAudit,
    PUDCGPV21Config,
    PUDCGPV21Workflow,
    SelectedJointGaussianProcessDistributionModel,
    SelectedSettingGroupedResidualBandCalibrator,
    StructuredExactICMGaussianProcessRegressor,
    V21DatasetCheckpoint,
    V21BenchmarkDatasetResult,
    V21SelectedMethodAudit,
    append_v21_dataset_checkpoint,
    bic_score,
    coregionalization_parameter_count,
    fit_aligned_v21_benchmark_methods,
    flatten_v21_structure_audit,
    load_v21_dataset_checkpoints,
    pu_dcgp_v21_development_contract,
    run_checkpointed_v21_benchmark,
    run_v21_benchmark_dataset,
    v21_benchmark_run_signature,
)
from experiments.pu_dcgp.benchmark_generator import (
    generate_identified_balanced_dataset,
)
from experiments.pu_dcgp.benchmark_metrics import evaluate_benchmark_method
from experiments.pu_dcgp.contracts import RunBatch
from experiments.pu_dcgp_v2 import (
    ExactICMGaussianProcessRegressor,
    DistributionEffectEstimate,
    FullCovarianceBootstrapWassersteinFPCAEncoder,
    FullScoreUncertaintyRepresentation,
    PreparedV2Data,
    V2BenchmarkReplicateRecord,
)


@lru_cache(maxsize=1)
def _v21_smoke_artifacts():
    contract = pu_dcgp_v21_development_contract()
    dataset = generate_identified_balanced_dataset(
        contract,
        sample_size=48,
        replicate_index=0,
    )
    config = PUDCGPV21Config(
        quantile_grid=contract.quantile_grid,
        particle_bootstrap_replicates=20,
        posterior_band_draws=500,
        calibration_folds=3,
        calibration_band_draws=100,
        optimize_joint_hyperparameters=False,
    )
    fit = fit_aligned_v21_benchmark_methods(
        dataset,
        contract,
        config,
    )
    return contract, dataset, config, fit


class _Selector(CoregionalizationStructureSelector):
    def select(
        self,
        outcome,
        run_count,
        component_count,
        candidates,
    ):
        selected = min(
            candidates,
            key=lambda candidate: candidate.negative_log_likelihood,
        )
        evidence = {
            candidate.structure: CoregionalizationCandidateEvidence(
                structure=candidate.structure,
                negative_log_likelihood=candidate.negative_log_likelihood,
                parameter_count=coregionalization_parameter_count(
                    candidate.structure,
                    component_count,
                ),
                bic=bic_score(
                    candidate.negative_log_likelihood,
                    coregionalization_parameter_count(
                        candidate.structure,
                        component_count,
                    ),
                    run_count,
                    component_count,
                ),
            )
            for candidate in candidates
        }
        return CoregionalizationSelectionDecision(
            outcome=outcome,
            run_count=run_count,
            component_count=component_count,
            selected_structure=selected.structure,
            candidate_evidence=evidence,
        )


class V21ScaffoldTests(unittest.TestCase):
    def test_config_separates_development_from_formal_randomness(self):
        config = PUDCGPV21Config()

        self.assertEqual(
            config.coregionalization_candidates,
            ("diagonal", "full"),
        )
        self.assertEqual(config.development_benchmark_random_seed, 32026)
        self.assertEqual(config.formal_benchmark_random_seed, 22026)
        self.assertEqual(config.shape_non_regression_ratio, 1.05)
        self.assertEqual(config.active_existence_power_min, 0.80)

    def test_parameter_counts_match_trace_normalized_contract(self):
        self.assertEqual(coregionalization_parameter_count("diagonal", 8), 11)
        self.assertEqual(coregionalization_parameter_count("full", 8), 39)
        self.assertEqual(coregionalization_parameter_count("diagonal", 2), 5)
        self.assertEqual(coregionalization_parameter_count("full", 2), 6)

    def test_thin_workflow_delegates_training_only_evidence(self):
        workflow = PUDCGPV21Workflow(PUDCGPV21Config(), _Selector())
        candidates = (
            CoregionalizationCandidateFit("diagonal", 90.0),
            CoregionalizationCandidateFit("full", 100.0),
        )

        decision = workflow.select_structure(
            "temperature_c",
            96,
            8,
            candidates,
        )

        self.assertEqual(decision.selected_structure, "diagonal")
        self.assertEqual(tuple(decision.candidate_evidence), ("diagonal", "full"))


class DiagonalTraceNormalizedICMParameterizationTests(unittest.TestCase):
    def setUp(self):
        self.parameterization = DiagonalTraceNormalizedICMParameterization(
            PUDCGPV21Config()
        )

    def test_encode_decode_returns_positive_trace_normalized_diagonal(self):
        initial = np.diag([0.5, 1.0, 2.0])
        parameters = self.parameterization.encode(
            2.0,
            8.0,
            1.5,
            0.2,
            initial,
        )
        decoded = self.parameterization.decode(parameters, 3)

        self.assertEqual(len(parameters), 6)
        np.testing.assert_allclose(
            decoded.coregionalization,
            np.diag(np.diag(decoded.coregionalization)),
        )
        self.assertAlmostEqual(
            float(np.trace(decoded.coregionalization)),
            3.0,
        )
        self.assertGreater(
            float(np.min(np.diag(decoded.coregionalization))),
            0.0,
        )
        np.testing.assert_allclose(
            decoded.coregionalization,
            initial / (np.trace(initial) / 3.0),
        )

    def test_analytic_derivatives_match_centered_finite_difference(self):
        parameters = np.array(
            [np.log(2.0), np.log(8.0), 0.0, np.log(0.2), -0.3, 0.4]
        )
        analytic = self.parameterization.coregionalization_derivatives(
            parameters,
            3,
        )
        step = 1e-6

        self.assertEqual(len(analytic), 2)
        for offset, derivative in enumerate(analytic, start=4):
            upper = parameters.copy()
            lower = parameters.copy()
            upper[offset] += step
            lower[offset] -= step
            finite_difference = (
                self.parameterization.decode(upper, 3).coregionalization
                - self.parameterization.decode(lower, 3).coregionalization
            ) / (2.0 * step)
            np.testing.assert_allclose(
                derivative,
                finite_difference,
                rtol=1e-5,
                atol=1e-7,
            )

    def test_bounds_have_one_latent_parameter_per_component_after_first(self):
        bounds = self.parameterization.bounds(8)

        self.assertEqual(len(bounds), 11)
        self.assertEqual(
            bounds[4:],
            (bounds[4],) * 7,
        )


class BICCoregionalizationStructureSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = BICCoregionalizationStructureSelector()

    def test_bic_uses_run_times_component_observation_count(self):
        expected = 2.0 * 90.0 + 39 * np.log(96 * 8)

        self.assertAlmostEqual(
            bic_score(90.0, 39, 96, 8),
            expected,
        )

    def test_selector_scores_fits_and_selects_lower_bic(self):
        decision = self.selector.select(
            outcome="temperature_c",
            run_count=96,
            component_count=8,
            candidates=(
                CoregionalizationCandidateFit("full", 90.0),
                CoregionalizationCandidateFit("diagonal", 100.0),
            ),
        )

        self.assertEqual(decision.selected_structure, "diagonal")
        self.assertEqual(
            decision.candidate_evidence["diagonal"].parameter_count,
            11,
        )
        self.assertEqual(
            decision.candidate_evidence["full"].parameter_count,
            39,
        )

    def test_exact_bic_tie_selects_diagonal(self):
        diagonal_nll = 100.0
        full_nll = diagonal_nll - (
            (39 - 11) * np.log(96 * 8) / 2.0
        )

        decision = self.selector.select(
            outcome="temperature_c",
            run_count=96,
            component_count=8,
            candidates=(
                CoregionalizationCandidateFit("full", full_nll),
                CoregionalizationCandidateFit("diagonal", diagonal_nll),
            ),
        )

        self.assertEqual(decision.selected_structure, "diagonal")


class CoregionalizationCandidateFitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        process = np.linspace(-1.0, 1.0, 12)
        context = np.tile([-1.0, 0.0, 1.0], 4)
        cls.predictors = np.column_stack([process, context])
        cls.targets = np.column_stack(
            [
                np.sin(1.5 * process) + 0.1 * context,
                0.7 * np.sin(1.5 * process) - 0.15 * context,
            ]
        )
        cls.observation_covariances = np.tile(
            np.array([[0.04, 0.015], [0.015, 0.09]]),
            (len(process), 1, 1),
        )
        cls.config = PUDCGPV21Config(
            optimizer_max_iterations=20,
            optimizer_ftol=1e-8,
        )
        cls.result = CoregionalizationCandidateFitter(
            cls.config,
            n_process_features=1,
        ).fit(
            "temperature_c",
            cls.predictors,
            cls.targets,
            cls.observation_covariances,
        )

    def test_two_candidates_fit_and_bic_decision_selects_one_model(self):
        self.assertEqual(
            tuple(self.result.models),
            ("diagonal", "full"),
        )
        self.assertIs(
            self.result.selected_model,
            self.result.models[self.result.decision.selected_structure],
        )
        for structure, model in self.result.models.items():
            self.assertAlmostEqual(
                self.result.decision.candidate_evidence[
                    structure
                ].negative_log_likelihood,
                model.negative_log_likelihood,
            )

    def test_diagonal_candidate_changes_only_latent_output_structure(self):
        diagonal = self.result.models["diagonal"]
        full = self.result.models["full"]

        np.testing.assert_allclose(
            diagonal.coregionalization,
            np.diag(np.diag(diagonal.coregionalization)),
            atol=1e-12,
        )
        self.assertGreater(
            float(np.linalg.eigvalsh(full.coregionalization).min()),
            0.0,
        )
        self.assertAlmostEqual(float(np.trace(diagonal.coregionalization)), 2.0)
        self.assertAlmostEqual(float(np.trace(full.coregionalization)), 2.0)
        for model in (diagonal, full):
            self.assertTrue(
                np.all(
                    np.abs(
                        model.standardized_observation_covariances[:, 0, 1]
                    )
                    > 0.0
                )
            )

    def test_structured_full_candidate_matches_unchanged_v2_default(self):
        legacy = ExactICMGaussianProcessRegressor(
            self.config,
            n_process_features=1,
        )
        legacy.fit(
            self.predictors,
            self.targets,
            self.observation_covariances,
        )
        structured = self.result.models["full"]

        np.testing.assert_allclose(
            structured.coregionalization,
            legacy.coregionalization,
        )
        self.assertAlmostEqual(
            structured.negative_log_likelihood,
            legacy.negative_log_likelihood,
        )
        self.assertIsInstance(
            structured,
            StructuredExactICMGaussianProcessRegressor,
        )


class SelectedJointGaussianProcessDistributionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        process = np.linspace(-1.0, 1.0, 12)
        context = np.tile([-1.0, 0.0, 1.0], 4)
        run_ids = tuple(f"run-{index}" for index in range(len(process)))
        runs = RunBatch(
            run_ids=run_ids,
            groups=("A",) * len(process),
            doe_modules=("M1",) * len(process),
            treatment_names=("current_a",),
            treatment_values=process[:, None],
            controlled_process_names=(),
            controlled_process_values=np.empty((len(process), 0)),
            context_names=("order",),
            context_values=context[:, None],
            particle_samples={},
        )
        two_scores = np.column_stack(
            [
                np.sin(1.5 * process) + 0.1 * context,
                0.7 * np.sin(1.5 * process) - 0.15 * context,
            ]
        )
        three_scores = np.column_stack(
            [
                np.cos(process),
                process + 0.2 * context,
                0.5 * process**2 - 0.1 * context,
            ]
        )
        two_covariance = np.tile(
            np.array([[0.04, 0.015], [0.015, 0.09]]),
            (len(process), 1, 1),
        )
        three_covariance = np.tile(
            np.array(
                [
                    [0.05, 0.01, -0.005],
                    [0.01, 0.07, 0.012],
                    [-0.005, 0.012, 0.06],
                ]
            ),
            (len(process), 1, 1),
        )
        representation = FullScoreUncertaintyRepresentation(
            run_ids=run_ids,
            outcome_names=("temperature_c", "velocity_m_s"),
            quantile_grid=np.array([0.25, 0.5, 0.75]),
            scores={
                "temperature_c": two_scores,
                "velocity_m_s": three_scores,
            },
            score_covariances={
                "temperature_c": two_covariance,
                "velocity_m_s": three_covariance,
            },
        )
        cls.data = PreparedV2Data(runs, representation)
        cls.model = SelectedJointGaussianProcessDistributionModel(
            PUDCGPV21Config(
                optimizer_max_iterations=20,
                optimizer_ftol=1e-8,
            )
        )
        cls.model.fit(cls.data)

    def test_each_outcome_has_its_own_auditable_selection(self):
        self.assertEqual(
            tuple(self.model.selection_decisions),
            ("temperature_c", "velocity_m_s"),
        )
        self.assertEqual(
            self.model.selection_decisions[
                "temperature_c"
            ].component_count,
            2,
        )
        self.assertEqual(
            self.model.selection_decisions[
                "velocity_m_s"
            ].component_count,
            3,
        )
        for outcome, result in self.model.candidate_results.items():
            self.assertIs(
                self.model.models[outcome],
                result.selected_model,
            )
            self.assertEqual(
                tuple(result.models),
                ("diagonal", "full"),
            )

    def test_prediction_preserves_joint_score_contract(self):
        prediction = self.model.predict(
            np.array([[-0.5], [0.5]]),
            np.array([[0.0], [1.0]]),
        )

        self.assertEqual(prediction.means["temperature_c"].shape, (2, 2))
        self.assertEqual(prediction.means["velocity_m_s"].shape, (2, 3))
        self.assertEqual(
            prediction.covariances["temperature_c"].shape,
            (2, 2, 2, 2),
        )
        self.assertEqual(
            prediction.covariances["velocity_m_s"].shape,
            (2, 3, 2, 3),
        )
        for covariance in prediction.covariances.values():
            flattened = covariance.reshape(
                covariance.shape[0] * covariance.shape[1],
                covariance.shape[2] * covariance.shape[3],
            )
            self.assertGreaterEqual(
                float(np.linalg.eigvalsh(flattened).min()),
                -1e-9,
            )


class SelectedSettingGroupedResidualBandCalibratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = np.repeat(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                ]
            ),
            2,
            axis=0,
        )
        base = np.linspace(-1.0, 1.0, 31)
        samples = tuple(
            0.4 * row[0]
            - 0.2 * row[1]
            + (1.0 + 0.1 * row[1]) * base
            + 0.02 * (index % 2)
            for index, row in enumerate(settings)
        )
        runs = RunBatch(
            run_ids=tuple(
                f"cal-{index:02d}" for index in range(len(settings))
            ),
            groups=("synthetic",) * len(settings),
            doe_modules=("synthetic",) * len(settings),
            treatment_names=("x1", "x2"),
            treatment_values=settings,
            controlled_process_names=(),
            controlled_process_values=np.empty((len(settings), 0)),
            context_names=("execution_order",),
            context_values=np.arange(len(settings), dtype=float)[:, None],
            particle_samples={"temperature_c": samples},
        )
        cls.config = PUDCGPV21Config(
            particle_bootstrap_replicates=20,
            optimize_joint_hyperparameters=False,
            calibration_folds=3,
            calibration_band_draws=100,
        )
        cls.encoder = FullCovarianceBootstrapWassersteinFPCAEncoder(
            cls.config
        )
        representation = cls.encoder.fit_transform(runs)
        cls.data = PreparedV2Data(runs, representation)
        cls.model = SelectedJointGaussianProcessDistributionModel(cls.config)
        cls.model.fit(cls.data)
        cls.calibrator = SelectedSettingGroupedResidualBandCalibrator(
            cls.config
        )
        cls.calibrator.fit(cls.data, cls.encoder, cls.model)

    def test_every_fold_reselects_from_fold_training_runs(self):
        audit = self.calibrator.fold_selection_audit

        self.assertEqual(len(audit), 3)
        for record in audit:
            self.assertEqual(record.training_run_count, 8)
            decision = record.decisions["temperature_c"]
            self.assertEqual(decision.run_count, 8)
            self.assertEqual(
                tuple(decision.candidate_evidence),
                ("diagonal", "full"),
            )
        self.assertEqual(
            self.model.selection_decisions["temperature_c"].run_count,
            12,
        )

    def test_inherited_calibration_changes_only_band_width(self):
        point = np.array([1.0, 2.0, 3.0])
        covariance = np.diag([0.04, 0.09, 0.16])
        effect = DistributionEffectEstimate(
            treatment_name="x1",
            quantile_grid=np.array([0.25, 0.50, 0.75]),
            effects={"temperature_c": point},
            covariances={"temperature_c": covariance},
            marginal_variances={"temperature_c": np.diag(covariance)},
            lower_bounds={
                "temperature_c": point - np.array([0.4, 0.6, 0.8])
            },
            upper_bounds={
                "temperature_c": point + np.array([0.4, 0.6, 0.8])
            },
            critical_values={"temperature_c": 2.0},
            interval_kind="joint_gp_uncalibrated_simultaneous_max_t",
        )

        calibrated = self.calibrator.calibrate(effect)

        np.testing.assert_allclose(
            calibrated.effects["temperature_c"],
            point,
        )
        np.testing.assert_allclose(
            calibrated.covariances["temperature_c"],
            covariance,
        )
        multiplier = self.calibrator.result.multipliers["temperature_c"]
        np.testing.assert_allclose(
            calibrated.upper_bounds["temperature_c"] - point,
            multiplier * (effect.upper_bounds["temperature_c"] - point),
        )


class V21BenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.contract,
            cls.dataset,
            cls.config,
            cls.fit,
        ) = _v21_smoke_artifacts()
        with patch(
            "experiments.pu_dcgp_v21.benchmark_runner."
            "fit_aligned_v21_benchmark_methods",
            return_value=cls.fit,
        ):
            cls.result = run_v21_benchmark_dataset(
                cls.contract,
                cls.config,
                "identified_balanced_particles",
                48,
                0,
            )

    def test_dataset_runner_returns_eight_metrics_and_complete_audit(self):
        self.assertIsInstance(self.result, V21BenchmarkDatasetResult)
        self.assertEqual(
            tuple(record.method_name for record in self.result.records),
            self.contract.methods,
        )
        self.assertEqual(len(self.result.structure_selections), 12)
        for record in self.result.records:
            self.assertTrue(np.isfinite(record.median_normalized_irmse))
            self.assertTrue(np.isfinite(record.active_admission_rate))

    def test_selected_gate_uses_explicit_reported_claims(self):
        gated = self.result.records[-1]
        ungated = self.result.records[-2]

        self.assertEqual(
            gated.method_name,
            "support_gated_joint_pu_dcgp_bic_selected",
        )
        self.assertLessEqual(
            gated.active_admission_rate,
            ungated.active_admission_rate,
        )
        self.assertLessEqual(
            gated.null_false_admission_rate,
            ungated.null_false_admission_rate,
        )

    def test_checkpointed_runner_skips_an_existing_dataset(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runner.jsonl"
            with patch(
                "experiments.pu_dcgp_v21.benchmark_runner."
                "run_v21_benchmark_dataset",
                return_value=self.result,
            ) as mocked:
                first = run_checkpointed_v21_benchmark(
                    path,
                    self.contract,
                    self.config,
                    sample_sizes=(48,),
                    replicate_indices=(0,),
                    scenario_ids=("identified_balanced_particles",),
                )
                second = run_checkpointed_v21_benchmark(
                    path,
                    self.contract,
                    self.config,
                    sample_sizes=(48,),
                    replicate_indices=(0,),
                    scenario_ids=("identified_balanced_particles",),
                )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(len(second), 1)


class V21BenchmarkAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.contract,
            cls.dataset,
            cls.config,
            cls.fit,
        ) = _v21_smoke_artifacts()

    def test_development_contract_has_new_seed_and_eight_frozen_methods(self):
        self.assertEqual(self.contract.random_seed, 32026)
        self.assertEqual(len(self.contract.methods), 8)
        self.assertEqual(
            self.contract.methods[-3:],
            (
                "joint_pu_dcgp_bic_selected",
                "joint_pu_dcgp_bic_selected_calibrated",
                "support_gated_joint_pu_dcgp_bic_selected",
            ),
        )

    def test_eight_aligned_methods_return_all_effects_and_finite_metrics(self):
        self.assertEqual(
            tuple(result.method_name for result in self.fit.methods),
            self.contract.methods,
        )
        for result in self.fit.methods:
            self.assertEqual(len(result.effects), 12)
            metrics = evaluate_benchmark_method(
                result,
                self.dataset,
                self.contract,
            )
            self.assertTrue(np.isfinite(metrics.median_normalized_irmse))
            self.assertTrue(np.isfinite(metrics.simultaneous_coverage_rate))

    def test_selected_model_audit_retains_full_and_fold_local_decisions(self):
        audit = self.fit.selected_audit

        self.assertEqual(
            tuple(audit.full_data_selections),
            ("temperature_c", "velocity_m_s", "particle_diameter_um"),
        )
        for decision in audit.full_data_selections.values():
            self.assertEqual(decision.run_count, 48)
            self.assertEqual(
                tuple(decision.candidate_evidence),
                ("diagonal", "full"),
            )
        self.assertEqual(len(audit.calibration_fold_selections), 3)
        for fold in audit.calibration_fold_selections:
            self.assertLess(fold.training_run_count, 48)

    def test_calibration_and_gate_do_not_change_selected_point_predictions(self):
        selected, calibrated, gated = self.fit.methods[-3:]

        for raw_effect, calibrated_effect, gated_effect in zip(
            selected.effects,
            calibrated.effects,
            gated.effects,
        ):
            np.testing.assert_allclose(
                calibrated_effect.point_effect,
                raw_effect.point_effect,
            )
            np.testing.assert_allclose(
                calibrated_effect.effect_covariance,
                raw_effect.effect_covariance,
            )
            np.testing.assert_allclose(
                gated_effect.point_effect,
                calibrated_effect.point_effect,
            )
            np.testing.assert_allclose(
                gated_effect.lower_bound,
                calibrated_effect.lower_bound,
            )


class V21DatasetCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.contract = pu_dcgp_v21_development_contract()
        self.config = PUDCGPV21Config(calibration_folds=3)
        self.signature = v21_benchmark_run_signature(
            self.contract,
            self.config,
        )
        self.records = tuple(
            V2BenchmarkReplicateRecord(
                scenario_id="identified_balanced",
                sample_size=48,
                replicate_index=0,
                method_name=method_name,
                median_normalized_irmse=0.1,
                shape_median_normalized_irmse=0.2,
                simultaneous_coverage_rate=0.95,
                active_coverage_rate=0.95,
                shape_coverage_rate=0.95,
                normalized_mean_band_width=0.3,
                active_admission_rate=0.8,
                null_false_admission_rate=0.05,
                target_unsupported_admitted=None,
                active_whole_curve_direction_rate=0.6,
                null_whole_curve_false_admission_rate=0.02,
                runtime_seconds=1.0,
            )
            for method_name in self.contract.methods
        )
        full_decisions = {
            outcome: self._decision(outcome, 48)
            for outcome in self.contract.outcome_names
        }
        fold_decisions = tuple(
            FoldStructureSelectionAudit(
                fold_index=fold_index,
                training_run_count=32,
                decisions={
                    outcome: self._decision(outcome, 32)
                    for outcome in self.contract.outcome_names
                },
            )
            for fold_index in range(self.config.calibration_folds)
        )
        self.structure_selections = flatten_v21_structure_audit(
            V21SelectedMethodAudit(
                full_data_selections=full_decisions,
                calibration_fold_selections=fold_decisions,
            )
        )
        self.checkpoint = V21DatasetCheckpoint(
            run_signature=self.signature,
            dataset_key=("identified_balanced", 48, 0),
            records=self.records,
            structure_selections=self.structure_selections,
        )

    def _decision(self, outcome, run_count):
        evidence = {
            "diagonal": CoregionalizationCandidateEvidence(
                "diagonal",
                100.0,
                5,
                220.0,
            ),
            "full": CoregionalizationCandidateEvidence(
                "full",
                95.0,
                6,
                230.0,
            ),
        }
        return CoregionalizationSelectionDecision(
            outcome=outcome,
            run_count=run_count,
            component_count=2,
            selected_structure="diagonal",
            candidate_evidence=evidence,
        )

    def test_complete_dataset_round_trip_preserves_metrics_and_audit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v21.jsonl"
            append_v21_dataset_checkpoint(
                path,
                self.checkpoint,
                self.contract,
                self.config,
            )

            loaded = load_v21_dataset_checkpoints(
                path,
                self.contract,
                self.config,
            )

        self.assertEqual(loaded, (self.checkpoint,))
        self.assertEqual(len(loaded[0].records), 8)
        self.assertEqual(
            len(loaded[0].structure_selections),
            3 * (1 + self.config.calibration_folds),
        )

    def test_signature_change_rejects_existing_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v21.jsonl"
            append_v21_dataset_checkpoint(
                path,
                self.checkpoint,
                self.contract,
                self.config,
            )

            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                load_v21_dataset_checkpoints(
                    path,
                    self.contract,
                    replace(
                        self.config,
                        particle_bootstrap_replicates=201,
                    ),
                )

    def test_interrupted_tail_is_removed_before_next_complete_dataset(self):
        second_records = tuple(
            replace(record, replicate_index=1)
            for record in self.records
        )
        second = replace(
            self.checkpoint,
            dataset_key=("identified_balanced", 48, 1),
            records=second_records,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v21.jsonl"
            append_v21_dataset_checkpoint(
                path,
                self.checkpoint,
                self.contract,
                self.config,
            )
            with path.open("ab") as stream:
                stream.write(b'{"interrupted"')
            append_v21_dataset_checkpoint(
                path,
                second,
                self.contract,
                self.config,
            )
            loaded = load_v21_dataset_checkpoints(
                path,
                self.contract,
                self.config,
            )

        self.assertEqual(
            tuple(checkpoint.dataset_key for checkpoint in loaded),
            (
                ("identified_balanced", 48, 0),
                ("identified_balanced", 48, 1),
            ),
        )

    def test_metrics_cannot_be_saved_without_complete_structure_audit(self):
        incomplete = replace(
            self.checkpoint,
            structure_selections=(),
        )

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "structure audit"):
                append_v21_dataset_checkpoint(
                    Path(directory) / "v21.jsonl",
                    incomplete,
                    self.contract,
                    self.config,
                )

    def test_method_order_must_match_the_frozen_contract(self):
        misordered = replace(
            self.checkpoint,
            records=(
                self.records[1],
                self.records[0],
                *self.records[2:],
            ),
        )

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "method order"):
                append_v21_dataset_checkpoint(
                    Path(directory) / "v21.jsonl",
                    misordered,
                    self.contract,
                    self.config,
                )

    def test_fold_audit_cannot_claim_the_full_data_run_count(self):
        first_fold = next(
            index
            for index, record in enumerate(self.structure_selections)
            if record.scope == "calibration_fold"
        )
        selections = list(self.structure_selections)
        selections[first_fold] = replace(
            selections[first_fold],
            training_run_count=48,
        )
        leaked = replace(
            self.checkpoint,
            structure_selections=tuple(selections),
        )

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "full-data run count"):
                append_v21_dataset_checkpoint(
                    Path(directory) / "v21.jsonl",
                    leaked,
                    self.contract,
                    self.config,
                )


if __name__ == "__main__":
    unittest.main()
