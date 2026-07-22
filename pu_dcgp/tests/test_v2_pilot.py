
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pu_dcgp.benchmark_generator import (
    generate_identified_balanced_dataset,
)
from experiments.pu_dcgp.benchmark_methods import (
    BenchmarkEffectEstimate,
    BenchmarkMethodResult,
)
from experiments.pu_dcgp_v2.benchmark_checkpoint import (
    V2DatasetCheckpoint,
    append_v2_dataset_checkpoint,
    load_v2_dataset_checkpoints,
    merge_v2_checkpoint_shards,
    run_checkpointed_v2_benchmark,
    v2_benchmark_run_signature,
)
from experiments.pu_dcgp_v2.benchmark_contract import (
    pu_dcgp_v2_benchmark_contract,
)
from experiments.pu_dcgp_v2.benchmark_completion_audit import (
    audit_v2_checkpoints,
)
from experiments.pu_dcgp_v2.benchmark_settings import (
    v2_formal_plan,
    v2_pilot_plan,
)
from experiments.pu_dcgp_v2.benchmark_runner import (
    V2BenchmarkAggregateRecord,
    V2BenchmarkReplicateRecord,
    aggregate_v2_benchmark_records,
    run_v2_benchmark_dataset,
)
from experiments.pu_dcgp_v2.benchmark_hypotheses import (
    evaluate_v2_benchmark_hypotheses,
)
from experiments.pu_dcgp_v2.config import PUDCGPV2Config
from experiments.pu_dcgp_v2.likelihood_selection import (
    TraceNormalizedICMParameterization,
)
from experiments.pu_dcgp_v2.gaussian_process import (
    ExactICMGaussianProcessRegressor,
)


class V2BenchmarkPlanTests(unittest.TestCase):
    def setUp(self):
        self.contract = pu_dcgp_v2_benchmark_contract()
        self.config = PUDCGPV2Config()

    def test_pilot_and_formal_axes_are_disjoint_and_complete(self):
        pilot = v2_pilot_plan(self.contract, self.config)
        formal = v2_formal_plan(self.contract, self.config)

        self.assertEqual(pilot.replicate_indices, tuple(range(20)))
        self.assertEqual(formal.replicate_indices, tuple(range(20, 220)))
        self.assertTrue(
            set(pilot.replicate_indices).isdisjoint(formal.replicate_indices)
        )
        self.assertEqual(pilot.dataset_count, 5 * 3 * 20)
        self.assertEqual(pilot.method_record_count, 5 * 3 * 20 * 6)

    def test_signature_changes_with_frozen_scientific_configuration(self):
        original = v2_benchmark_run_signature(self.contract, self.config)
        changed = v2_benchmark_run_signature(
            self.contract,
            replace(self.config, calibration_folds=4),
        )

        self.assertEqual(len(original), 64)
        self.assertNotEqual(original, changed)


class TraceNormalizedDerivativeTests(unittest.TestCase):
    def test_coregionalization_derivatives_match_central_difference(self):
        parameterization = TraceNormalizedICMParameterization(PUDCGPV2Config())
        coregionalization = np.array(
            [
                [1.2, 0.2, -0.1],
                [0.2, 0.9, 0.15],
                [-0.1, 0.15, 1.1],
            ]
        )
        parameters = parameterization.encode(2.0, 8.0, 1.0, 0.2, coregionalization)
        analytic = parameterization.coregionalization_derivatives(
            parameters,
            3,
        )
        step = 1e-6

        for offset, derivative in enumerate(analytic, start=4):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[offset] += step
            minus[offset] -= step
            numeric = (
                parameterization.decode(plus, 3).coregionalization
                - parameterization.decode(minus, 3).coregionalization
            ) / (2 * step)

            np.testing.assert_allclose(
                derivative,
                numeric,
                rtol=2e-6,
                atol=2e-8,
            )
            self.assertAlmostEqual(float(np.trace(derivative)), 0.0, places=10)

    def test_joint_gp_marginal_likelihood_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(19)
        config = PUDCGPV2Config()
        model = ExactICMGaussianProcessRegressor(
            config,
            n_process_features=2,
        )
        predictors = rng.normal(size=(7, 3))
        targets = rng.normal(size=(7, 3)).ravel()
        blocks = np.repeat(
            (
                np.eye(3) * 0.04
                + np.full((3, 3), 0.005)
            )[None, :, :],
            7,
            axis=0,
        )
        observation_covariance = (
            model._block_diagonal_observation_covariance(blocks)
        )
        parameterization = TraceNormalizedICMParameterization(config)
        parameters = parameterization.encode(
            1.7,
            3.2,
            0.8,
            0.15,
            np.array(
                [
                    [1.0, 0.2, -0.1],
                    [0.2, 0.9, 0.15],
                    [-0.1, 0.15, 1.1],
                ]
            ),
        )

        value, analytic = model._training_nll_and_gradient(
            parameters,
            3,
            predictors,
            targets,
            observation_covariance,
            parameterization,
        )
        numeric = np.empty_like(parameters)
        step = 1e-6
        for index in range(len(parameters)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            numeric[index] = (
                model._training_negative_log_likelihood(
                    plus,
                    3,
                    predictors,
                    targets,
                    observation_covariance,
                    parameterization,
                )
                - model._training_negative_log_likelihood(
                    minus,
                    3,
                    predictors,
                    targets,
                    observation_covariance,
                    parameterization,
                )
            ) / (2 * step)

        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(
            analytic,
            numeric,
            rtol=2e-5,
            atol=2e-6,
        )


class V2DatasetRunnerTests(unittest.TestCase):
    def setUp(self):
        self.contract = pu_dcgp_v2_benchmark_contract()
        self.dataset = generate_identified_balanced_dataset(
            self.contract,
            sample_size=48,
            replicate_index=0,
        )

    def test_one_dataset_produces_the_frozen_six_method_set(self):
        results = self._ideal_results()
        with patch(
            "experiments.pu_dcgp_v2.benchmark_runner."
            "fit_aligned_v2_benchmark_methods",
            return_value=results,
        ):
            records = run_v2_benchmark_dataset(
                self.contract,
                PUDCGPV2Config(),
                self.dataset.scenario_id,
                self.dataset.sample_size,
                self.dataset.replicate_index,
            )

        self.assertEqual(
            tuple(record.method_name for record in records),
            self.contract.methods,
        )
        self.assertEqual(len(records), 6)
        self.assertTrue(
            all(record.simultaneous_coverage_rate == 1.0 for record in records)
        )
        self.assertTrue(
            all(
                record.active_whole_curve_direction_rate <= 1.0
                for record in records
            )
        )

        aggregates = aggregate_v2_benchmark_records(records)
        self.assertEqual(len(aggregates), 6)
        self.assertTrue(all(record.replicate_count == 1 for record in aggregates))

    def _ideal_results(self):
        effects = []
        for truth in self.dataset.truths:
            half_width = np.full_like(truth.effect, 0.01)
            effects.append(
                BenchmarkEffectEstimate(
                    estimand_id=truth.estimand_id,
                    treatment_name=truth.treatment_name,
                    outcome=truth.outcome,
                    quantile_grid=truth.quantile_grid,
                    point_effect=truth.effect,
                    marginal_variance=np.full_like(truth.effect, 1e-6),
                    effect_covariance=np.eye(len(truth.effect)) * 1e-6,
                    lower_bound=truth.effect - half_width,
                    upper_bound=truth.effect + half_width,
                    interval_kind="test",
                    admission_status="test",
                    reported=truth.is_active,
                    failed_gates=(),
                )
            )
        return tuple(
            BenchmarkMethodResult(
                method_name=method_name,
                scenario_id=self.dataset.scenario_id,
                sample_size=self.dataset.sample_size,
                replicate_index=self.dataset.replicate_index,
                effects=tuple(effects),
                preparation_seconds=0.1,
                fit_seconds=0.2,
                prediction_seconds=0.3,
            )
            for method_name in self.contract.methods
        )


class V2DatasetCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.contract = pu_dcgp_v2_benchmark_contract()
        self.config = PUDCGPV2Config()
        self.signature = v2_benchmark_run_signature(
            self.contract,
            self.config,
        )

    def test_one_json_line_round_trips_one_complete_dataset(self):
        checkpoint = self._checkpoint(0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shard.jsonl"
            append_v2_dataset_checkpoint(
                path,
                checkpoint,
                self.contract.methods,
            )
            loaded = load_v2_dataset_checkpoints(
                path,
                self.signature,
                self.contract.methods,
            )

            self.assertEqual(loaded, (checkpoint,))
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_resume_skips_complete_dataset_and_runs_only_missing_key(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume.jsonl"
            append_v2_dataset_checkpoint(
                path,
                self._checkpoint(0),
                self.contract.methods,
            )
            calls = []

            def fake_run(contract, config, scenario_id, sample_size, index):
                calls.append(index)
                return self._checkpoint(index).records

            with patch(
                "experiments.pu_dcgp_v2.benchmark_runner."
                "run_v2_benchmark_dataset",
                side_effect=fake_run,
            ):
                loaded = run_checkpointed_v2_benchmark(
                    path,
                    self.contract,
                    self.config,
                    (48,),
                    (0, 1),
                    ("identified_balanced_particles",),
                )

            self.assertEqual(calls, [1])
            self.assertEqual(len(loaded), 2)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_resume_discards_only_an_interrupted_final_json_fragment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.jsonl"
            append_v2_dataset_checkpoint(
                path,
                self._checkpoint(0),
                self.contract.methods,
            )
            with path.open("ab") as stream:
                stream.write(b'{"schema":"pu_dcgp_v2_data')

            visible = load_v2_dataset_checkpoints(
                path,
                self.signature,
                self.contract.methods,
            )
            append_v2_dataset_checkpoint(
                path,
                self._checkpoint(1),
                self.contract.methods,
            )
            recovered = load_v2_dataset_checkpoints(
                path,
                self.signature,
                self.contract.methods,
            )

            self.assertEqual(visible, (self._checkpoint(0),))
            self.assertEqual(
                tuple(checkpoint.dataset_key[2] for checkpoint in recovered),
                (0, 1),
            )
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_disjoint_shards_merge_and_rerun_idempotently(self):
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            merged = Path(directory) / "merged.jsonl"
            append_v2_dataset_checkpoint(
                first,
                self._checkpoint(0),
                self.contract.methods,
            )
            append_v2_dataset_checkpoint(
                second,
                self._checkpoint(1),
                self.contract.methods,
            )

            initial = merge_v2_checkpoint_shards(
                (first, second),
                merged,
                self.signature,
                self.contract.methods,
            )
            repeated = merge_v2_checkpoint_shards(
                (first, second),
                merged,
                self.signature,
                self.contract.methods,
            )

            self.assertEqual(initial, repeated)
            self.assertEqual(len(repeated), 2)
            self.assertEqual(len(merged.read_text().splitlines()), 2)

    def test_integrity_audit_distinguishes_partial_from_invalid(self):
        plan = replace(
            v2_pilot_plan(self.contract, self.config),
            scenario_ids=("identified_balanced_particles",),
            sample_sizes=(48,),
            replicate_indices=(0, 1),
            dataset_count=2,
            method_record_count=12,
        )

        partial = audit_v2_checkpoints(
            (self._checkpoint(0),),
            self.contract,
            plan,
        )
        completed = audit_v2_checkpoints(
            (self._checkpoint(0), self._checkpoint(1)),
            self.contract,
            plan,
        )

        self.assertTrue(partial.integrity_passed)
        self.assertFalse(partial.phase_complete)
        self.assertEqual(partial.missing_dataset_count, 1)
        self.assertTrue(completed.integrity_passed)
        self.assertTrue(completed.phase_complete)

    def _checkpoint(self, replicate_index):
        key = ("identified_balanced_particles", 48, replicate_index)
        records = tuple(
            V2BenchmarkReplicateRecord(
                scenario_id=key[0],
                sample_size=key[1],
                replicate_index=key[2],
                method_name=method_name,
                median_normalized_irmse=0.1,
                shape_median_normalized_irmse=0.1,
                simultaneous_coverage_rate=0.95,
                active_coverage_rate=0.95,
                shape_coverage_rate=0.95,
                normalized_mean_band_width=0.2,
                active_admission_rate=0.8,
                null_false_admission_rate=0.05,
                target_unsupported_admitted=None,
                active_whole_curve_direction_rate=0.6,
                null_whole_curve_false_admission_rate=0.01,
                runtime_seconds=1.0,
            )
            for method_name in self.contract.methods
        )
        return V2DatasetCheckpoint(self.signature, key, records)


class V2PilotThresholdTests(unittest.TestCase):
    def test_complete_pilot_is_labelled_diagnostic_not_formal(self):
        contract = pu_dcgp_v2_benchmark_contract()
        config = PUDCGPV2Config()
        aggregates = []
        supported = {
            "identified_balanced_particles",
            "identified_heterogeneous_particles",
        }
        failures = {
            "sequence_aligned_drift",
            "module_sign_reversal",
            "insufficient_overlap",
        }
        for scenario in (
            scenario.scenario_id for scenario in contract.scenarios
        ):
            for sample_size in (48, 96, 144):
                for method in contract.methods:
                    shape_error = {
                        "mean_gp": 0.20,
                        "pu_dcgp_diagonal_v1": 0.15,
                        "joint_pu_dcgp_group_calibrated": 0.14,
                    }.get(method, 0.16)
                    coverage = 0.90
                    if method == "distribution_gp_no_pu":
                        coverage = 0.80 if "heterogeneous" in scenario else 0.93
                    if method in {
                        "joint_pu_dcgp_group_calibrated",
                        "support_gated_joint_pu_dcgp",
                    }:
                        coverage = 0.94 if "balanced" in scenario else 0.93
                    target = None
                    if scenario in failures:
                        target = (
                            0.02
                            if method == "support_gated_joint_pu_dcgp"
                            else 0.40
                        )
                    aggregates.append(
                        V2BenchmarkAggregateRecord(
                            scenario_id=scenario,
                            sample_size=sample_size,
                            method_name=method,
                            replicate_count=20,
                            median_normalized_irmse=shape_error,
                            median_shape_normalized_irmse=shape_error,
                            simultaneous_coverage_rate=coverage,
                            active_coverage_rate=coverage,
                            shape_coverage_rate=coverage,
                            normalized_mean_band_width=0.2,
                            active_admission_rate=(
                                0.85
                                if scenario in supported
                                and method
                                == "support_gated_joint_pu_dcgp"
                                else 0.70
                            ),
                            null_false_admission_rate=(
                                0.03
                                if scenario in supported
                                and method
                                == "support_gated_joint_pu_dcgp"
                                else 0.10
                            ),
                            target_unsupported_admission_rate=target,
                            active_whole_curve_direction_rate=0.60,
                            null_whole_curve_false_admission_rate=0.01,
                            median_runtime_seconds=1.0,
                        )
                    )

        decisions = evaluate_v2_benchmark_hypotheses(
            tuple(aggregates),
            contract,
            config,
            "pilot",
        )

        self.assertEqual(
            tuple(decision.hypothesis_id for decision in decisions),
            ("S1", "H2-v2", "H3-v2", "H4-v2"),
        )
        self.assertTrue(all(decision.threshold_met for decision in decisions))
        self.assertTrue(
            all(decision.status == "pilot_threshold_met" for decision in decisions)
        )
        self.assertAlmostEqual(
            decisions[3].evidence[
                "supported_active_whole_curve_direction_power"
            ],
            0.60,
        )


if __name__ == "__main__":
    unittest.main()
