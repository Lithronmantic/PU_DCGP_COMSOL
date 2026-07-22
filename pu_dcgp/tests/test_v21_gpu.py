"""Numerical checks for the optional PU-DCGP v2.1 GPU skeleton."""

import unittest

import numpy as np
from scipy.linalg import cho_solve

from experiments.pu_dcgp.benchmark_generator import (
    generate_identified_balanced_dataset,
)
from experiments.pu_dcgp_v21.gpu_linear_algebra import (
    CuPyDenseLinearAlgebra,
    inspect_cupy_backend,
)
from experiments.pu_dcgp_v21.gpu_likelihood import (
    CuPyICMLikelihoodWorkspace,
)
from experiments.pu_dcgp_v21.gpu_model import (
    GPUAcceleratedStructuredExactICMGaussianProcessRegressor,
)
from experiments.pu_dcgp_v21 import (
    CoregionalizationCandidateFitter,
    PUDCGPV21Config,
    StructuredExactICMGaussianProcessRegressor,
    fit_v21_selected_effect_methods,
    pu_dcgp_v21_development_contract,
)


class CuPyDenseLinearAlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status = inspect_cupy_backend()
        if not cls.status.available:
            raise unittest.SkipTest(cls.status.detail)
        cls.backend = CuPyDenseLinearAlgebra()

    def test_probe_reports_a_cuda_device(self):
        self.assertTrue(self.status.available)
        self.assertTrue(self.status.device_name)

    def test_float64_cholesky_solve_matches_cpu(self):
        rng = np.random.default_rng(32026)
        factor = rng.normal(size=(64, 64))
        covariance = factor @ factor.T + np.eye(64) * 0.5
        right_hand_side = rng.normal(size=(64, 5))
        cpu_cholesky = np.linalg.cholesky(covariance)
        expected = cho_solve(
            (cpu_cholesky, True),
            right_hand_side,
            check_finite=False,
        )

        device_covariance = self.backend.to_device(covariance)
        device_rhs = self.backend.to_device(right_hand_side)
        device_cholesky = self.backend.cholesky(device_covariance)
        actual = self.backend.to_host(
            self.backend.cholesky_solve(device_cholesky, device_rhs)
        )

        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_identity_inverse_solve_matches_cpu(self):
        rng = np.random.default_rng(22026)
        factor = rng.normal(size=(48, 48))
        covariance = factor @ factor.T + np.eye(48)
        expected = np.linalg.inv(covariance)

        device_covariance = self.backend.to_device(covariance)
        device_cholesky = self.backend.cholesky(device_covariance)
        actual = self.backend.to_host(
            self.backend.cholesky_solve(
                device_cholesky,
                self.backend.identity(len(covariance)),
            )
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-11,
            atol=1e-11,
        )


class CuPyICMLikelihoodWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = inspect_cupy_backend()
        if not status.available:
            raise unittest.SkipTest(status.detail)
        cls.config = PUDCGPV21Config()
        process = np.linspace(-1.0, 1.0, 18)
        context = np.tile([-1.0, 0.0, 1.0], 6)
        cls.predictors = np.column_stack([process, context])
        cls.targets = np.column_stack(
            [
                np.sin(process) + 0.1 * context,
                0.7 * np.sin(process) - 0.2 * context,
                np.cos(process) + 0.05 * context,
            ]
        ).ravel()
        observation_blocks = np.tile(
            np.array(
                [
                    [0.05, 0.01, -0.005],
                    [0.01, 0.07, 0.012],
                    [-0.005, 0.012, 0.06],
                ]
            ),
            (len(process), 1, 1),
        )
        cls.observation_covariance = (
            StructuredExactICMGaussianProcessRegressor
            ._block_diagonal_observation_covariance(observation_blocks)
        )

    def test_nll_and_gradient_match_cpu_for_both_structures(self):
        initial_coregionalization = np.array(
            [
                [1.0, 0.2, -0.1],
                [0.2, 1.1, 0.15],
                [-0.1, 0.15, 0.9],
            ]
        )
        for structure in ("diagonal", "full"):
            with self.subTest(structure=structure):
                model = StructuredExactICMGaussianProcessRegressor(
                    self.config,
                    structure,
                    n_process_features=1,
                )
                parameterization = model._parameterization()
                parameters = parameterization.encode(
                    1.4,
                    2.2,
                    1.1,
                    0.08,
                    initial_coregionalization,
                )
                expected_nll, expected_gradient = (
                    model._training_nll_and_gradient(
                        parameters,
                        3,
                        self.predictors,
                        self.targets,
                        self.observation_covariance,
                        parameterization,
                    )
                )
                workspace = CuPyICMLikelihoodWorkspace(
                    self.config,
                    self.predictors,
                    self.targets,
                    self.observation_covariance,
                    component_count=3,
                    n_process_features=1,
                )
                static_arrays = (
                    workspace.targets,
                    workspace.observation_covariance,
                    workspace.process_distance,
                    workspace.context_distance,
                )

                actual_nll, actual_gradient = workspace.evaluate(
                    parameters,
                    parameterization,
                )

                self.assertAlmostEqual(actual_nll, expected_nll, places=9)
                np.testing.assert_allclose(
                    actual_gradient,
                    expected_gradient,
                    rtol=1e-9,
                    atol=1e-9,
                )
                self.assertEqual(
                    static_arrays,
                    (
                        workspace.targets,
                        workspace.observation_covariance,
                        workspace.process_distance,
                        workspace.context_distance,
                    ),
                )


class GPUAcceleratedStructuredModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = inspect_cupy_backend()
        if not status.available:
            raise unittest.SkipTest(status.detail)
        process = np.linspace(-1.0, 1.0, 30)
        context = np.tile([-1.0, 0.0, 1.0], 10)
        cls.predictors = np.column_stack([process, context])
        cls.targets = np.column_stack(
            [
                np.sin(1.2 * process) + 0.1 * context,
                0.7 * np.sin(1.2 * process) - 0.15 * context,
                np.cos(process) + 0.05 * context,
            ]
        )
        cls.observation_covariances = np.tile(
            np.array(
                [
                    [0.04, 0.008, -0.004],
                    [0.008, 0.06, 0.01],
                    [-0.004, 0.01, 0.05],
                ]
            ),
            (len(process), 1, 1),
        )
        cls.config = PUDCGPV21Config(
            optimizer_max_iterations=30,
            optimizer_ftol=1e-9,
        )

    def test_optimized_model_and_prediction_match_cpu(self):
        prediction_points = np.array(
            [
                [-0.75, -0.5],
                [0.0, 0.0],
                [0.8, 0.5],
            ]
        )
        for structure in ("diagonal", "full"):
            with self.subTest(structure=structure):
                cpu_model = StructuredExactICMGaussianProcessRegressor(
                    self.config,
                    structure,
                    n_process_features=1,
                )
                gpu_model = (
                    GPUAcceleratedStructuredExactICMGaussianProcessRegressor(
                        self.config,
                        structure,
                        n_process_features=1,
                    )
                )

                cpu_model.fit(
                    self.predictors,
                    self.targets,
                    self.observation_covariances,
                )
                gpu_model.fit(
                    self.predictors,
                    self.targets,
                    self.observation_covariances,
                )
                cpu_mean, cpu_covariance = cpu_model.predict(
                    prediction_points
                )
                gpu_mean, gpu_covariance = gpu_model.predict(
                    prediction_points
                )

                self.assertIsNotNone(
                    gpu_model.gpu_likelihood_workspace
                )
                self.assertEqual(
                    gpu_model.optimization_result.converged,
                    cpu_model.optimization_result.converged,
                )
                self.assertEqual(
                    gpu_model.optimization_result.iterations,
                    cpu_model.optimization_result.iterations,
                )
                self.assertAlmostEqual(
                    gpu_model.negative_log_likelihood,
                    cpu_model.negative_log_likelihood,
                    places=6,
                )
                np.testing.assert_allclose(
                    gpu_mean,
                    cpu_mean,
                    rtol=1e-7,
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    gpu_covariance,
                    cpu_covariance,
                    rtol=1e-7,
                    atol=1e-8,
                )

    def test_candidate_selection_matches_cpu(self):
        cpu_result = CoregionalizationCandidateFitter(
            self.config,
            n_process_features=1,
        ).fit(
            "temperature_c",
            self.predictors,
            self.targets,
            self.observation_covariances,
        )
        gpu_result = CoregionalizationCandidateFitter(
            self.config,
            n_process_features=1,
            model_class=(
                GPUAcceleratedStructuredExactICMGaussianProcessRegressor
            ),
        ).fit(
            "temperature_c",
            self.predictors,
            self.targets,
            self.observation_covariances,
        )

        self.assertEqual(
            gpu_result.decision.selected_structure,
            cpu_result.decision.selected_structure,
        )
        for structure in ("diagonal", "full"):
            self.assertAlmostEqual(
                gpu_result.decision.candidate_evidence[
                    structure
                ].negative_log_likelihood,
                cpu_result.decision.candidate_evidence[
                    structure
                ].negative_log_likelihood,
                places=6,
            )
            self.assertAlmostEqual(
                gpu_result.decision.candidate_evidence[structure].bic,
                cpu_result.decision.candidate_evidence[structure].bic,
                places=6,
            )


class GPUEndToEndSelectedMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = inspect_cupy_backend()
        if not status.available:
            raise unittest.SkipTest(status.detail)
        cls.contract = pu_dcgp_v21_development_contract()
        cls.dataset = generate_identified_balanced_dataset(
            cls.contract,
            sample_size=48,
            replicate_index=0,
        )
        cls.config = PUDCGPV21Config(
            quantile_grid=cls.contract.quantile_grid,
            particle_bootstrap_replicates=20,
            posterior_band_draws=500,
            calibration_folds=3,
            calibration_band_draws=100,
            optimizer_max_iterations=5,
            optimizer_ftol=1e-8,
        )

    def test_calibrated_effects_and_fold_selections_match_cpu(self):
        cpu_fit = fit_v21_selected_effect_methods(
            self.dataset,
            self.contract,
            self.config,
        )
        gpu_fit = fit_v21_selected_effect_methods(
            self.dataset,
            self.contract,
            self.config,
            model_class=(
                GPUAcceleratedStructuredExactICMGaussianProcessRegressor
            ),
        )

        self.assertEqual(
            {
                outcome: decision.selected_structure
                for outcome, decision
                in gpu_fit.audit.full_data_selections.items()
            },
            {
                outcome: decision.selected_structure
                for outcome, decision
                in cpu_fit.audit.full_data_selections.items()
            },
        )
        self.assertEqual(
            [
                {
                    outcome: decision.selected_structure
                    for outcome, decision in fold.decisions.items()
                }
                for fold in gpu_fit.audit.calibration_fold_selections
            ],
            [
                {
                    outcome: decision.selected_structure
                    for outcome, decision in fold.decisions.items()
                }
                for fold in cpu_fit.audit.calibration_fold_selections
            ],
        )
        for cpu_method, gpu_method in zip(
            cpu_fit.methods,
            gpu_fit.methods,
            strict=True,
        ):
            self.assertEqual(cpu_method.method_name, gpu_method.method_name)
            for cpu_effect, gpu_effect in zip(
                cpu_method.effects,
                gpu_method.effects,
                strict=True,
            ):
                np.testing.assert_allclose(
                    gpu_effect.point_effect,
                    cpu_effect.point_effect,
                    rtol=1e-7,
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    gpu_effect.effect_covariance,
                    cpu_effect.effect_covariance,
                    rtol=1e-7,
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    gpu_effect.lower_bound,
                    cpu_effect.lower_bound,
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    gpu_effect.upper_bound,
                    cpu_effect.upper_bound,
                    rtol=1e-6,
                    atol=1e-6,
                )
                self.assertEqual(
                    gpu_effect.admission_status,
                    cpu_effect.admission_status,
                )
                self.assertEqual(
                    gpu_effect.reported,
                    cpu_effect.reported,
                )
                self.assertEqual(
                    gpu_effect.failed_gates,
                    cpu_effect.failed_gates,
                )
