
import unittest

import numpy as np

from experiments.pu_dcgp import (
    BootstrapWassersteinFPCAEncoder,
    a_group_doe_estimands,
    JointDistributionPrediction,
    ManifestDataSource,
    RunBatch,
)
from experiments.pu_dcgp.contracts import (
    JointScorePrediction as V1JointScorePrediction,
)
from experiments.pu_dcgp.benchmark_generator import (
    generate_identified_balanced_dataset,
    generate_insufficient_overlap_dataset,
    generate_module_sign_reversal_dataset,
    generate_sequence_aligned_drift_dataset,
)
from experiments.pu_dcgp.benchmark_methods import (
    BenchmarkEffectEstimate,
    BenchmarkMethodResult,
    _matched_contrast_points,
)
from experiments.pu_dcgp.benchmark_metrics import evaluate_benchmark_method
from experiments.pu_dcgp.distribution_encoder import OutcomeFPCAState
from experiments.pu_dcgp_v2 import (
    ClaimAdmissionDecision,
    ClaimEvidence,
    ClaimSpecificGate,
    DistributionBandCalibrator,
    DistributionEffectEstimate,
    GaussianJointDistributionalContrastEstimator,
    FullCovarianceDistributionEncoder,
    FullCovarianceBootstrapWassersteinFPCAEncoder,
    ExactICMGaussianProcessRegressor,
    JointGaussianProcessDistributionModel,
    audit_full_covariance_encoder,
    render_full_covariance_encoder_audit,
    FullScoreUncertaintyRepresentation,
    JointScorePrediction,
    JointScoreResponseModel,
    JointDistributionalContrastEstimator,
    PairedContrastDesign,
    PUDCGPV2Config,
    PUDCGPV2Workflow,
    PreparedV2Data,
    TraceNormalizedICMParameterization,
    a_group_paired_contrast_design,
    SupportEvidence,
    SettingGroupedResidualBandCalibrator,
    finite_sample_curve_multiplier,
    fit_aligned_v2_benchmark_methods,
    pu_dcgp_v2_benchmark_contract,
    synthetic_paired_contrast_design,
    HierarchicalClaimSpecificGate,
    evaluate_v2_benchmark_selection,
    apply_v2_claim_decisions,
    evaluate_v2_synthetic_claim_decisions,
)


def _runs() -> RunBatch:
    return RunBatch(
        run_ids=("A001", "A002"),
        groups=("A", "A"),
        doe_modules=("DOE1", "DOE1"),
        treatment_names=("current_a",),
        treatment_values=np.array([[600.0], [800.0]]),
        controlled_process_names=("hydrogen_setting",),
        controlled_process_values=np.array([[2.5], [2.5]]),
        context_names=("execution_order",),
        context_values=np.array([[1.0], [2.0]]),
        particle_samples={
            "temperature_c": (np.array([100.0]), np.array([110.0]))
        },
    )


def _representation() -> FullScoreUncertaintyRepresentation:
    return FullScoreUncertaintyRepresentation(
        run_ids=("A001", "A002"),
        outcome_names=("temperature_c",),
        quantile_grid=np.array([0.25, 0.50, 0.75]),
        scores={"temperature_c": np.array([[0.1, -0.2], [0.3, 0.4]])},
        score_covariances={
            "temperature_c": np.array(
                [
                    [[0.04, 0.01], [0.01, 0.09]],
                    [[0.02, -0.005], [-0.005, 0.03]],
                ]
            )
        },
    )


def _calibration_runs() -> RunBatch:
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
    return RunBatch(
        run_ids=tuple(f"cal-{index:02d}" for index in range(len(settings))),
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


class _Encoder(FullCovarianceDistributionEncoder):
    def fit_transform(self, runs):
        return _representation()

    def transform(self, runs):
        return _representation()

    def inverse_transform(self, prediction):
        return JointDistributionPrediction(
            quantile_grid=np.array([0.25, 0.50, 0.75]),
            means={},
            covariances={},
        )


class _Model(JointScoreResponseModel):
    def __init__(self):
        self.fit_data = None

    def fit(self, data):
        self.fit_data = data

    def predict(self, treatments, contexts):
        return JointScorePrediction(means={}, covariances={})


class _Calibrator(DistributionBandCalibrator):
    def __init__(self):
        self.fit_data = None

    def fit(self, data, encoder, model):
        self.fit_data = data

    def calibrate(self, effect):
        return effect


class _Gate(ClaimSpecificGate):
    def decide(self, support, claims):
        return ClaimAdmissionDecision(
            status="admit",
            admitted_claims=("existence",),
            failed_support_gates=(),
        )


class _ContrastEstimator(JointDistributionalContrastEstimator):
    def estimate(self, model, encoder, design):
        return DistributionEffectEstimate(
            treatment_name=design.treatment_name,
            quantile_grid=np.array([0.5]),
            effects={},
            covariances={},
            marginal_variances={},
            lower_bounds={},
            upper_bounds={},
            critical_values={},
            interval_kind="test",
        )


class _FixedPredictionModel(JointScoreResponseModel):
    def __init__(self, prediction):
        self.prediction = prediction

    def fit(self, data):
        return None

    def predict(self, treatments, contexts):
        return self.prediction


class _FixedDistributionEncoder(FullCovarianceDistributionEncoder):
    def __init__(self, prediction):
        self.prediction = prediction

    def fit_transform(self, runs):
        return _representation()

    def transform(self, runs):
        return _representation()

    def inverse_transform(self, prediction):
        return self.prediction


class V2ScaffoldTests(unittest.TestCase):
    def test_config_freezes_scope_and_success_thresholds(self):
        config = PUDCGPV2Config()

        self.assertEqual(config.analysis_groups, ("A",))
        self.assertEqual(config.benchmark_random_seed, 22026)
        self.assertEqual(config.formal_replicate_start, 20)
        self.assertEqual(config.formal_replicate_count, 200)
        self.assertEqual(config.heterogeneous_coverage_improvement, 0.02)
        self.assertEqual(config.active_existence_power_min, 0.80)

    def test_full_score_covariance_contract_accepts_psd_blocks(self):
        _representation().validate()

    def test_full_score_covariance_contract_rejects_v1_diagonal_shape(self):
        representation = _representation()
        invalid = FullScoreUncertaintyRepresentation(
            run_ids=representation.run_ids,
            outcome_names=representation.outcome_names,
            quantile_grid=representation.quantile_grid,
            scores=representation.scores,
            score_covariances={
                "temperature_c": np.array([[0.04, 0.09], [0.02, 0.03]])
            },
        )

        with self.assertRaisesRegex(ValueError, "score covariance must have shape"):
            invalid.validate()

    def test_workflow_prepares_and_fits_declared_components(self):
        model = _Model()
        calibrator = _Calibrator()
        workflow = PUDCGPV2Workflow(
            config=PUDCGPV2Config(),
            encoder=_Encoder(),
            model=model,
            calibrator=calibrator,
            gate=_Gate(),
            contrast_estimator=_ContrastEstimator(),
        )

        data = workflow.prepare(_runs())
        workflow.fit(data)
        effect = workflow.estimate_uncalibrated(
            PairedContrastDesign(
                treatment_name="current_a",
                reference_treatments=np.array([[600.0]]),
                intervention_treatments=np.array([[800.0]]),
                contexts=np.array([[1.0]]),
            )
        )

        self.assertIs(model.fit_data, data)
        self.assertIs(calibrator.fit_data, data)
        self.assertEqual(
            data.distributions.score_covariances["temperature_c"].shape,
            (2, 2, 2),
        )
        self.assertEqual(effect.treatment_name, "current_a")

    def test_claim_contract_keeps_three_claims_separate(self):
        support = SupportEvidence(True, True, True, True, True)
        claims = ClaimEvidence(True, False, False)

        decision = _Gate().decide(support, claims)

        self.assertEqual(decision.admitted_claims, ("existence",))
        self.assertNotIn("whole_curve_direction", decision.admitted_claims)


class ClaimSpecificGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = HierarchicalClaimSpecificGate()
        self.supported = SupportEvidence(True, True, True, True, True)

    def test_existence_claim_is_the_reporting_admission(self):
        decision = self.gate.decide(
            self.supported,
            ClaimEvidence(True, False, False),
        )

        self.assertEqual(decision.status, "existence_admit")
        self.assertEqual(decision.admitted_claims, ("existence",))
        self.assertEqual(decision.failed_support_gates, ())

    def test_mean_direction_can_be_retained_without_existence_reporting(self):
        decision = self.gate.decide(
            self.supported,
            ClaimEvidence(False, True, False),
        )

        self.assertEqual(decision.status, "mean_direction_only")
        self.assertEqual(decision.admitted_claims, ("mean_direction",))
        self.assertNotIn("existence", decision.admitted_claims)

    def test_any_support_failure_vetoes_all_statistical_claims(self):
        decision = self.gate.decide(
            SupportEvidence(True, True, False, True, True),
            ClaimEvidence(True, True, True),
        )

        self.assertEqual(decision.status, "support_abstain")
        self.assertEqual(decision.admitted_claims, ())
        self.assertEqual(decision.failed_support_gates, ("sequence_sign",))

    def test_structural_failure_has_distinct_status(self):
        decision = self.gate.decide(
            SupportEvidence(False, True, True, True, True),
            ClaimEvidence(True, True, True),
        )

        self.assertEqual(decision.status, "insufficient_support")
        self.assertEqual(decision.admitted_claims, ())


class FullCovarianceEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = ManifestDataSource(groups=("A",)).load()
        cls.config = PUDCGPV2Config(particle_bootstrap_replicates=40)
        cls.encoder = FullCovarianceBootstrapWassersteinFPCAEncoder(
            cls.config
        )
        cls.representation = cls.encoder.fit_transform(cls.runs)
        cls.v1_encoder = BootstrapWassersteinFPCAEncoder(
            cls.encoder.legacy_config
        )
        cls.v1_representation = cls.v1_encoder.fit_transform(cls.runs)

    def test_scores_and_covariance_diagonal_match_aligned_v1(self):
        for outcome in self.representation.outcome_names:
            np.testing.assert_allclose(
                self.representation.scores[outcome],
                self.v1_representation.scores[outcome],
                atol=0.0,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                np.diagonal(
                    self.representation.score_covariances[outcome],
                    axis1=1,
                    axis2=2,
                ),
                self.v1_representation.score_variances[outcome],
                atol=1e-12,
                rtol=1e-12,
            )

    def test_covariance_blocks_are_symmetric_psd_and_not_diagonal(self):
        for outcome in self.representation.outcome_names:
            covariance = self.representation.score_covariances[outcome]
            np.testing.assert_allclose(
                covariance,
                np.swapaxes(covariance, 1, 2),
                atol=0.0,
                rtol=0.0,
            )
            self.assertGreaterEqual(
                float(np.linalg.eigvalsh(covariance).min()),
                -self.config.covariance_eigenvalue_tolerance,
            )
            off_diagonal = covariance[:, 0, 1]
            self.assertTrue(np.any(np.abs(off_diagonal) > 1e-12))

    def test_full_covariances_are_reproducible(self):
        repeated = self.encoder.transform(self.runs)

        for outcome in self.representation.outcome_names:
            np.testing.assert_allclose(
                repeated.score_covariances[outcome],
                self.representation.score_covariances[outcome],
                atol=0.0,
                rtol=0.0,
            )

    def test_joint_inverse_transform_matches_v1_when_cross_scores_are_zero(self):
        v2_means = {
            outcome: score[:3]
            for outcome, score in self.representation.scores.items()
        }
        v1_covariances = {}
        v2_covariances = {}
        for outcome, means in v2_means.items():
            point_count, component_count = means.shape
            v1_covariance = np.empty(
                (component_count, point_count, point_count), dtype=float
            )
            v2_covariance = np.zeros(
                (
                    point_count,
                    component_count,
                    point_count,
                    component_count,
                ),
                dtype=float,
            )
            for component in range(component_count):
                block = np.eye(point_count) * (component + 1.0)
                v1_covariance[component] = block
                v2_covariance[:, component, :, component] = block
            v1_covariances[outcome] = v1_covariance
            v2_covariances[outcome] = v2_covariance

        v1_distribution = self.v1_encoder.inverse_transform_joint(
            V1JointScorePrediction(
                means=v2_means,
                covariances=v1_covariances,
            )
        )
        v2_distribution = self.encoder.inverse_transform(
            JointScorePrediction(
                means=v2_means,
                covariances=v2_covariances,
            )
        )

        for outcome in self.representation.outcome_names:
            np.testing.assert_allclose(
                v2_distribution.means[outcome],
                v1_distribution.means[outcome],
            )
            np.testing.assert_allclose(
                v2_distribution.covariances[outcome],
                v1_distribution.covariances[outcome],
            )

    def test_a_group_audit_finds_material_off_diagonal_information(self):
        audit = audit_full_covariance_encoder(self.runs, self.config)
        rendered = render_full_covariance_encoder_audit(audit)

        self.assertEqual(len(audit.entries), 3)
        for entry in audit.entries:
            self.assertLess(entry.maximum_v1_diagonal_difference, 1e-12)
            self.assertGreater(entry.median_absolute_score_correlation, 0.25)
            self.assertGreaterEqual(entry.minimum_eigenvalue, -1e-10)
        self.assertIn("representation evidence only", rendered)


class ExactICMGaussianProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictors = np.linspace(0.0, 1.0, 12)[:, None]
        first = np.sin(2 * np.pi * cls.predictors[:, 0])
        second = 0.7 * first + 0.3 * np.cos(
            2 * np.pi * cls.predictors[:, 0]
        )
        cls.targets = np.column_stack([first, second])
        cls.observation_covariances = np.repeat(
            np.array([[[0.002, 0.001], [0.001, 0.003]]]),
            len(cls.predictors),
            axis=0,
        )
        cls.config = PUDCGPV2Config(
            joint_process_lengthscale=1.0,
            joint_context_lengthscale=1.0,
            joint_noise_variance=1e-5,
        )
        cls.model = ExactICMGaussianProcessRegressor(
            cls.config,
            n_process_features=1,
        )
        cls.model.fit(
            cls.predictors,
            cls.targets,
            cls.observation_covariances,
        )

    def test_block_diagonal_observation_covariance_preserves_run_blocks(self):
        blocks = np.array(
            [
                [[1.0, 0.2], [0.2, 2.0]],
                [[3.0, -0.4], [-0.4, 4.0]],
            ]
        )

        matrix = self.model._block_diagonal_observation_covariance(blocks)

        np.testing.assert_allclose(matrix[:2, :2], blocks[0])
        np.testing.assert_allclose(matrix[2:, 2:], blocks[1])
        np.testing.assert_allclose(matrix[:2, 2:], 0.0)
        np.testing.assert_allclose(matrix[2:, :2], 0.0)

    def test_joint_gp_recovers_smooth_training_curves_and_psd_covariance(self):
        means, covariance = self.model.predict(self.predictors)
        flat_covariance = covariance.reshape(24, 24)

        self.assertLess(
            float(np.sqrt(np.mean(np.square(means - self.targets)))),
            0.02,
        )
        self.assertEqual(covariance.shape, (12, 2, 12, 2))
        np.testing.assert_allclose(
            flat_covariance,
            flat_covariance.T,
            atol=1e-12,
        )
        self.assertGreaterEqual(
            float(np.linalg.eigvalsh(flat_covariance).min()),
            -1e-10,
        )
        self.assertGreater(
            float(np.max(np.abs(covariance[:, 0, :, 1]))),
            1e-6,
        )

    def test_trace_normalized_parameterization_is_identifiable_and_spd(self):
        parameterization = TraceNormalizedICMParameterization(self.config)
        original = np.array([[1.2, 0.4], [0.4, 0.8]])
        parameters = parameterization.encode(2.0, 8.0, 1.0, 0.2, original)

        decoded = parameterization.decode(parameters, 2)

        np.testing.assert_allclose(
            decoded.coregionalization,
            original / (np.trace(original) / 2),
        )
        self.assertAlmostEqual(
            float(np.trace(decoded.coregionalization) / 2),
            1.0,
        )
        self.assertGreater(
            float(np.linalg.eigvalsh(decoded.coregionalization).min()),
            0.0,
        )
        self.assertEqual(len(parameters), len(parameterization.bounds(2)))

    def test_training_likelihood_selection_improves_nll_and_recovers_dependence(self):
        result = self.model.optimization_result

        self.assertTrue(result.converged)
        self.assertLess(
            result.selected_negative_log_likelihood,
            result.initial_negative_log_likelihood,
        )
        self.assertAlmostEqual(
            self.model.negative_log_likelihood,
            result.selected_negative_log_likelihood,
            places=7,
        )
        self.assertLessEqual(
            result.function_evaluations,
            2 * result.iterations + 5,
        )
        self.assertGreater(abs(self.model.coregionalization[0, 1]), 0.5)

    def test_training_likelihood_selection_is_reproducible(self):
        repeated = ExactICMGaussianProcessRegressor(
            self.config,
            n_process_features=1,
        )
        repeated.fit(
            self.predictors,
            self.targets,
            self.observation_covariances,
        )

        np.testing.assert_allclose(
            repeated.coregionalization,
            self.model.coregionalization,
        )
        self.assertEqual(repeated.hyperparameters, self.model.hyperparameters)
        self.assertAlmostEqual(
            repeated.negative_log_likelihood,
            self.model.negative_log_likelihood,
        )

    def test_diagonal_latent_and_observation_structure_has_zero_cross_score_posterior(self):
        diagonal_observation = self.observation_covariances.copy()
        diagonal_observation[:, 0, 1] = 0.0
        diagonal_observation[:, 1, 0] = 0.0
        model = ExactICMGaussianProcessRegressor(
            self.config,
            n_process_features=1,
        )
        model.fit(
            self.predictors,
            self.targets,
            diagonal_observation,
            coregionalization=np.eye(2),
            optimize_hyperparameters=False,
        )

        _, covariance = model.predict(self.predictors[:4])

        np.testing.assert_allclose(
            covariance[:, 0, :, 1],
            0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            covariance[:, 1, :, 0],
            0.0,
            atol=1e-12,
        )

    def test_off_diagonal_observation_blocks_change_joint_posterior(self):
        diagonal_observation = self.observation_covariances.copy()
        diagonal_observation[:, 0, 1] = 0.0
        diagonal_observation[:, 1, 0] = 0.0
        fixed_coregionalization = np.array([[1.0, 0.5], [0.5, 1.0]])
        full_model = ExactICMGaussianProcessRegressor(
            self.config,
            n_process_features=1,
        )
        diagonal_model = ExactICMGaussianProcessRegressor(
            self.config,
            n_process_features=1,
        )
        full_model.fit(
            self.predictors,
            self.targets,
            self.observation_covariances,
            coregionalization=fixed_coregionalization,
            optimize_hyperparameters=False,
        )
        diagonal_model.fit(
            self.predictors,
            self.targets,
            diagonal_observation,
            coregionalization=fixed_coregionalization,
            optimize_hyperparameters=False,
        )

        full_means, full_covariance = full_model.predict(self.predictors[:4])
        diagonal_means, diagonal_covariance = diagonal_model.predict(
            self.predictors[:4]
        )

        self.assertGreater(
            float(np.max(np.abs(full_means - diagonal_means))),
            1e-7,
        )
        self.assertGreater(
            float(np.max(np.abs(full_covariance - diagonal_covariance))),
            1e-7,
        )

    def test_distribution_model_returns_frozen_joint_axes(self):
        runs = _runs()
        representation = _representation()
        model = JointGaussianProcessDistributionModel(
            PUDCGPV2Config(joint_noise_variance=0.01)
        )
        model.fit(PreparedV2Data(runs=runs, distributions=representation))

        prediction = model.predict(
            runs.treatment_values,
            runs.context_values,
        )

        self.assertEqual(
            prediction.means["temperature_c"].shape,
            (2, 2),
        )
        self.assertEqual(
            prediction.covariances["temperature_c"].shape,
            (2, 2, 2, 2),
        )


class JointDistributionalContrastTests(unittest.TestCase):
    def test_a_group_design_uses_frozen_strata_and_mean_context_only(self):
        runs = ManifestDataSource(groups=("A",)).load()
        expected_context = runs.context_values.mean(axis=0)

        for estimand in a_group_doe_estimands():
            design = a_group_paired_contrast_design(runs, estimand)
            treatment_index = runs.treatment_names.index(
                estimand.treatment_name
            )

            self.assertEqual(design.reference_treatments.shape, (9, 4))
            self.assertEqual(design.intervention_treatments.shape, (9, 4))
            np.testing.assert_allclose(
                design.reference_treatments[:, treatment_index],
                estimand.reference_value,
            )
            np.testing.assert_allclose(
                design.intervention_treatments[:, treatment_index],
                estimand.intervention_value,
            )
            other_indices = [
                index
                for index in range(4)
                if index != treatment_index
            ]
            np.testing.assert_allclose(
                design.reference_treatments[:, other_indices],
                design.intervention_treatments[:, other_indices],
            )
            np.testing.assert_allclose(
                design.contexts,
                np.tile(expected_context, (9, 1)),
            )

    def test_paired_contrast_matches_analytic_mean_and_covariance(self):
        quantile_grid = np.array([0.25, 0.50, 0.75])
        point_effect = np.array([1.0, 2.0, 3.0])
        means = np.vstack(
            [
                np.zeros(3),
                np.zeros(3),
                point_effect,
                point_effect,
            ]
        )
        quantile_covariance = np.array(
            [
                [0.04, 0.01, 0.00],
                [0.01, 0.09, 0.02],
                [0.00, 0.02, 0.16],
            ]
        )
        covariance = np.kron(
            np.eye(4),
            quantile_covariance,
        ).reshape(4, 3, 4, 3)
        distribution = JointDistributionPrediction(
            quantile_grid=quantile_grid,
            means={"temperature_c": means},
            covariances={"temperature_c": covariance},
        )
        design = PairedContrastDesign(
            treatment_name="current_a",
            reference_treatments=np.array([[0.0], [0.0]]),
            intervention_treatments=np.array([[1.0], [1.0]]),
            contexts=np.array([[0.0], [0.0]]),
        )
        estimator = GaussianJointDistributionalContrastEstimator(
            PUDCGPV2Config(
                quantile_grid=tuple(quantile_grid),
                posterior_band_draws=2000,
            )
        )

        effect = estimator.estimate(
            _FixedPredictionModel(JointScorePrediction({}, {})),
            _FixedDistributionEncoder(distribution),
            design,
        )

        np.testing.assert_allclose(
            effect.effects["temperature_c"], point_effect
        )
        np.testing.assert_allclose(
            effect.covariances["temperature_c"], quantile_covariance
        )
        np.testing.assert_allclose(
            effect.marginal_variances["temperature_c"],
            np.diag(quantile_covariance),
        )
        self.assertGreater(effect.critical_values["temperature_c"], 1.96)
        self.assertEqual(
            effect.interval_kind,
            "joint_gp_uncalibrated_simultaneous_max_t",
        )

    def test_full_cross_score_terms_reach_effect_covariance(self):
        config = PUDCGPV2Config(
            quantile_grid=(0.25, 0.50, 0.75),
            posterior_band_draws=1000,
        )
        encoder = FullCovarianceBootstrapWassersteinFPCAEncoder(config)
        weights = encoder._basis.quadrature_weights
        encoder._basis.states = {
            "temperature_c": OutcomeFPCAState(
                mean_quantile=np.array([0.0, 1.0, 2.0]),
                quadrature_weights=weights.copy(),
                scaled_components=np.array(
                    [[0.4, 0.4, 0.4], [0.3, 0.0, -0.3]]
                ),
                explained_variance_ratio=np.array([0.8, 0.2]),
            )
        }
        score_means = np.array([[0.0, 0.0], [0.2, 0.1]])
        full_block = np.array([[0.04, 0.015], [0.015, 0.03]])
        diagonal_block = np.diag(np.diag(full_block))

        def prediction(block):
            covariance = np.zeros((2, 2, 2, 2))
            covariance[0, :, 0, :] = block
            covariance[1, :, 1, :] = block
            return JointScorePrediction(
                means={"temperature_c": score_means},
                covariances={"temperature_c": covariance},
            )

        design = PairedContrastDesign(
            treatment_name="current_a",
            reference_treatments=np.array([[0.0]]),
            intervention_treatments=np.array([[1.0]]),
            contexts=np.array([[0.0]]),
        )
        estimator = GaussianJointDistributionalContrastEstimator(config)
        full = estimator.estimate(
            _FixedPredictionModel(prediction(full_block)),
            encoder,
            design,
        )
        diagonal = estimator.estimate(
            _FixedPredictionModel(prediction(diagonal_block)),
            encoder,
            design,
        )
        full_covariance = full.covariances["temperature_c"]
        diagonal_covariance = diagonal.covariances["temperature_c"]

        self.assertGreater(
            float(np.max(np.abs(full_covariance - diagonal_covariance))),
            1e-6,
        )
        self.assertGreaterEqual(
            float(np.linalg.eigvalsh(full_covariance).min()),
            -1e-10,
        )

    def test_uncalibrated_band_is_reproducible(self):
        quantile_grid = np.array([0.25, 0.50, 0.75])
        distribution = JointDistributionPrediction(
            quantile_grid=quantile_grid,
            means={
                "temperature_c": np.array(
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                )
            },
            covariances={
                "temperature_c": np.eye(6).reshape(2, 3, 2, 3) * 0.04
            },
        )
        design = PairedContrastDesign(
            treatment_name="current_a",
            reference_treatments=np.array([[0.0]]),
            intervention_treatments=np.array([[1.0]]),
            contexts=np.array([[0.0]]),
        )
        estimator = GaussianJointDistributionalContrastEstimator(
            PUDCGPV2Config(
                quantile_grid=tuple(quantile_grid),
                posterior_band_draws=1000,
            )
        )
        model = _FixedPredictionModel(JointScorePrediction({}, {}))
        encoder = _FixedDistributionEncoder(distribution)

        first = estimator.estimate(model, encoder, design)
        second = estimator.estimate(model, encoder, design)

        np.testing.assert_allclose(
            first.lower_bounds["temperature_c"],
            second.lower_bounds["temperature_c"],
        )
        np.testing.assert_allclose(
            first.upper_bounds["temperature_c"],
            second.upper_bounds["temperature_c"],
        )
        self.assertEqual(
            first.critical_values["temperature_c"],
            second.critical_values["temperature_c"],
        )


class SettingGroupedBandCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = _calibration_runs()
        cls.config = PUDCGPV2Config(
            particle_bootstrap_replicates=20,
            optimize_joint_hyperparameters=False,
            calibration_folds=3,
            calibration_band_draws=300,
        )
        cls.encoder = FullCovarianceBootstrapWassersteinFPCAEncoder(
            cls.config
        )
        cls.representation = cls.encoder.fit_transform(cls.runs)
        cls.model = JointGaussianProcessDistributionModel(cls.config)
        cls.data = PreparedV2Data(
            runs=cls.runs,
            distributions=cls.representation,
        )
        cls.model.fit(cls.data)
        cls.calibrator = SettingGroupedResidualBandCalibrator(cls.config)
        cls.calibrator.fit(cls.data, cls.encoder, cls.model)

    def test_finite_sample_multiplier_uses_frozen_upper_rank(self):
        multiplier, rank = finite_sample_curve_multiplier(
            (0.2, 0.5, 0.1, 0.9, 0.4),
            0.80,
        )

        self.assertEqual(rank, 5)
        self.assertEqual(multiplier, 0.9)

    def test_cross_fit_keeps_setting_groups_and_scores_each_curve(self):
        result = self.calibrator.result

        self.assertEqual(result.setting_count, 6)
        self.assertEqual(result.curve_count, 12)
        self.assertEqual(result.quantile_ranks["temperature_c"], 12)
        self.assertTrue(np.isfinite(result.multipliers["temperature_c"]))
        self.assertGreater(result.multipliers["temperature_c"], 0.0)

    def test_calibration_scales_only_band_half_width_and_critical_value(self):
        point = np.array([1.0, 2.0, 3.0])
        covariance = np.diag([0.04, 0.09, 0.16])
        uncalibrated = DistributionEffectEstimate(
            treatment_name="x1",
            quantile_grid=np.array([0.25, 0.50, 0.75]),
            effects={"temperature_c": point},
            covariances={"temperature_c": covariance},
            marginal_variances={"temperature_c": np.diag(covariance)},
            lower_bounds={"temperature_c": point - np.array([0.4, 0.6, 0.8])},
            upper_bounds={"temperature_c": point + np.array([0.4, 0.6, 0.8])},
            critical_values={"temperature_c": 2.0},
            interval_kind="joint_gp_uncalibrated_simultaneous_max_t",
        )

        calibrated = self.calibrator.calibrate(uncalibrated)
        multiplier = self.calibrator.result.multipliers["temperature_c"]

        np.testing.assert_allclose(
            calibrated.effects["temperature_c"], point
        )
        np.testing.assert_allclose(
            calibrated.covariances["temperature_c"], covariance
        )
        np.testing.assert_allclose(
            calibrated.lower_bounds["temperature_c"],
            point - multiplier * np.array([0.4, 0.6, 0.8]),
        )
        np.testing.assert_allclose(
            calibrated.upper_bounds["temperature_c"],
            point + multiplier * np.array([0.4, 0.6, 0.8]),
        )
        self.assertEqual(
            calibrated.critical_values["temperature_c"],
            2.0 * multiplier,
        )

    def test_cross_fitted_multiplier_is_reproducible(self):
        repeated = SettingGroupedResidualBandCalibrator(self.config)
        repeated.fit(self.data, self.encoder, self.model)

        self.assertEqual(repeated.result, self.calibrator.result)


class V2BenchmarkAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = pu_dcgp_v2_benchmark_contract()
        cls.dataset = generate_identified_balanced_dataset(
            cls.contract,
            sample_size=48,
            replicate_index=0,
        )
        cls.config = PUDCGPV2Config(
            quantile_grid=cls.contract.quantile_grid,
            particle_bootstrap_replicates=20,
            posterior_band_draws=500,
            calibration_folds=3,
            calibration_band_draws=200,
            optimize_joint_hyperparameters=False,
        )
        cls.results = fit_aligned_v2_benchmark_methods(
            cls.dataset,
            cls.contract,
            cls.config,
        )

    def test_v2_contract_has_independent_seed_and_six_methods(self):
        self.assertEqual(self.contract.random_seed, 22026)
        self.assertEqual(self.contract.pilot_replicate_count, 20)
        self.assertEqual(self.contract.replicate_count, 200)
        self.assertEqual(len(self.contract.methods), 6)
        self.assertEqual(
            self.contract.methods[-2:],
            (
                "joint_pu_dcgp_group_calibrated",
                "support_gated_joint_pu_dcgp",
            ),
        )

    def test_synthetic_contrast_design_exactly_matches_v1_points(self):
        for treatment_name in self.contract.treatment_names:
            v1_reference, v1_intervention, v1_contexts = (
                _matched_contrast_points(
                    self.dataset.runs,
                    treatment_name,
                )
            )
            v2 = synthetic_paired_contrast_design(
                self.dataset.runs,
                treatment_name,
            )

            np.testing.assert_allclose(
                v2.reference_treatments, v1_reference
            )
            np.testing.assert_allclose(
                v2.intervention_treatments, v1_intervention
            )
            np.testing.assert_allclose(v2.contexts, v1_contexts)

    def test_six_aligned_methods_return_all_twelve_effects(self):
        self.assertEqual(
            tuple(result.method_name for result in self.results),
            (
                "mean_gp",
                "distribution_gp_no_pu",
                "pu_dcgp_diagonal_v1",
                "joint_pu_dcgp_full",
                "joint_pu_dcgp_group_calibrated",
                "support_gated_joint_pu_dcgp",
            ),
        )
        for result in self.results:
            self.assertEqual(len(result.effects), 12)
            self.assertEqual(
                tuple(
                    (effect.estimand_id, effect.outcome)
                    for effect in result.effects
                ),
                tuple(
                    (truth.estimand_id, truth.outcome)
                    for truth in self.dataset.truths
                ),
            )

    def test_calibration_changes_only_joint_band_bounds(self):
        full = self.results[3]
        calibrated = self.results[4]
        changed_bounds = 0
        for full_effect, calibrated_effect in zip(
            full.effects,
            calibrated.effects,
        ):
            np.testing.assert_allclose(
                calibrated_effect.point_effect,
                full_effect.point_effect,
            )
            np.testing.assert_allclose(
                calibrated_effect.effect_covariance,
                full_effect.effect_covariance,
            )
            if not np.allclose(
                calibrated_effect.lower_bound,
                full_effect.lower_bound,
            ):
                changed_bounds += 1
        self.assertGreater(changed_bounds, 0)

    def test_all_six_records_are_accepted_by_known_truth_metrics(self):
        for result in self.results:
            metrics = evaluate_benchmark_method(
                result,
                self.dataset,
                self.contract,
            )
            self.assertTrue(np.isfinite(metrics.median_normalized_irmse))
            self.assertTrue(np.isfinite(metrics.simultaneous_coverage_rate))
            self.assertTrue(np.isfinite(metrics.normalized_mean_band_width))

    def test_v2_selection_uses_existence_and_gated_reported_flags(self):
        calibrated = evaluate_v2_benchmark_selection(
            self.results[4],
            self.dataset,
            self.contract,
        )
        gated = evaluate_v2_benchmark_selection(
            self.results[5],
            self.dataset,
            self.contract,
        )

        self.assertEqual(
            calibrated.admitted_count,
            sum(effect.reported for effect in self.results[4].effects),
        )
        self.assertEqual(
            gated.admitted_count,
            sum(effect.reported for effect in self.results[5].effects),
        )
        self.assertLessEqual(gated.admitted_count, calibrated.admitted_count)
        for effect in self.results[3:5]:
            for estimate in effect.effects:
                self.assertEqual(
                    estimate.reported,
                    bool(
                        np.any(estimate.lower_bound > 0.0)
                        or np.any(estimate.upper_bound < 0.0)
                    ),
                )
        for calibrated_effect, gated_effect in zip(
            self.results[4].effects,
            self.results[5].effects,
        ):
            np.testing.assert_allclose(
                gated_effect.point_effect,
                calibrated_effect.point_effect,
            )
            np.testing.assert_allclose(
                gated_effect.lower_bound,
                calibrated_effect.lower_bound,
            )
            np.testing.assert_allclose(
                gated_effect.upper_bound,
                calibrated_effect.upper_bound,
            )

    def test_three_failure_generators_trigger_their_support_vetoes(self):
        cases = (
            (
                generate_sequence_aligned_drift_dataset,
                "sequence_sign",
            ),
            (
                generate_module_sign_reversal_dataset,
                "module_mean_direction",
            ),
            (
                generate_insufficient_overlap_dataset,
                "structural_support",
            ),
        )
        for generator, expected_gate in cases:
            dataset = generator(self.contract, 48, 0)
            calibrated = self._truth_backed_result(dataset)
            decisions = evaluate_v2_synthetic_claim_decisions(
                dataset,
                self.contract,
                self.config,
                calibrated,
            )
            scenario = next(
                scenario
                for scenario in self.contract.scenarios
                if scenario.scenario_id == dataset.scenario_id
            )
            target = next(
                decision
                for decision in decisions
                if decision.treatment_name == scenario.target_treatment
                and decision.outcome == scenario.target_outcome
            )
            gated = apply_v2_claim_decisions(calibrated, decisions)
            gated_target = next(
                effect
                for effect in gated.effects
                if effect.treatment_name == scenario.target_treatment
                and effect.outcome == scenario.target_outcome
            )

            self.assertIn(
                expected_gate,
                target.decision.failed_support_gates,
            )
            self.assertFalse(gated_target.reported)

    def _truth_backed_result(self, dataset):
        effects = []
        for truth in dataset.truths:
            covariance = np.eye(len(truth.effect)) * 1e-6
            half_width = np.full(len(truth.effect), 1e-3)
            effects.append(
                BenchmarkEffectEstimate(
                    estimand_id=truth.estimand_id,
                    treatment_name=truth.treatment_name,
                    outcome=truth.outcome,
                    quantile_grid=truth.quantile_grid,
                    point_effect=truth.effect,
                    marginal_variance=np.diag(covariance),
                    effect_covariance=covariance,
                    lower_bound=truth.effect - half_width,
                    upper_bound=truth.effect + half_width,
                    interval_kind="test_calibrated",
                    admission_status="ungated_existence",
                    reported=truth.is_active,
                    failed_gates=(),
                )
            )
        return BenchmarkMethodResult(
            method_name="joint_pu_dcgp_group_calibrated",
            scenario_id=dataset.scenario_id,
            sample_size=dataset.sample_size,
            replicate_index=dataset.replicate_index,
            effects=tuple(effects),
            preparation_seconds=0.0,
            fit_seconds=0.0,
            prediction_seconds=0.0,
        )


if __name__ == "__main__":
    unittest.main()
