
import unittest
from dataclasses import replace
import json
from statistics import NormalDist
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.pu_dcgp import (
    CausalContrastEstimator,
    BootstrapWassersteinFPCAEncoder,
    ContrastSpec,
    DistributionEffect,
    DistributionEncoder,
    DistributionPrediction,
    DistributionRepresentation,
    JointDistributionPrediction,
    JointScorePrediction,
    DistributionResponseModel,
    EvaluationResult,
    ExactGaussianProcessRegressor,
    GaussianProcessDistributionModel,
    GaussianProcessMeanModel,
    ModelEvaluator,
    ManifestDataSource,
    PUDCGPConfig,
    PUDCGPWorkflow,
    PolynomialRidgeMeanModel,
    PreparedData,
    RunBatch,
    RunDataSource,
    ScorePrediction,
    cross_validate_mean_baselines,
    cross_validate_gp_models,
    grouped_setting_folds,
    subset_run_batch,
    audit_repeated_settings,
    audit_module_predictions,
    audit_matched_factor_support,
    audit_shared_linear_drift,
    find_shared_module_anchors,
    audit_cross_module_offset,
    aps_ysz_a_causal_graph,
    hydrogen_to_argon_setting_ratio,
    response_predictors,
    a_group_doe_estimands,
    audit_contrast_support,
    estimate_matched_distribution_effects,
    estimate_sequence_adjusted_effects,
    estimate_module_consistency,
    bootstrap_matched_mean_effects,
    bootstrap_matched_distribution_effect,
    GATE_NAMES,
    EffectAdmissionEvidence,
    decide_effect_admission,
    evaluate_all_effect_admissions,
    PhysicsDirectionEvidence,
    annotate_physics_consistency,
    frozen_existing_physics_evidence,
    pu_dcgp_benchmark_contract,
    analytic_expected_run_quantiles,
    benchmark_quantile_truths,
    module_reversal_quantile_truths,
    generate_identified_balanced_dataset,
    generate_identified_design,
    generate_identified_heterogeneous_dataset,
    generate_sequence_aligned_design,
    generate_sequence_aligned_drift_dataset,
    generate_module_sign_reversal_dataset,
    generate_insufficient_overlap_dataset,
    generate_insufficient_overlap_design,
    DOEEstimand,
    audit_identified_generator_oracle,
    audit_identified_heterogeneous_generator_oracle,
    audit_particle_count_isolation,
    audit_sequence_drift_generator,
    audit_module_reversal_generator,
    audit_overlap_failure_generator,
    fit_benchmark_point_effect_methods,
    apply_admission_decisions,
    audit_point_method_adapters,
    average_paired_quantile_contrast,
    gaussian_simultaneous_band,
    evaluate_benchmark_method,
    audit_method_band_pilot,
    evaluate_synthetic_admission_decisions,
    benchmark_pilot_config,
    benchmark_formal_config,
    formal_benchmark_plan,
    aggregate_benchmark_records,
    run_benchmark_replicates,
    aggregate_gate_power_diagnostics,
    synthetic_admission_observations,
    append_checkpoint_records,
    benchmark_run_signature,
    completed_dataset_keys,
    load_checkpoint_records,
    merge_checkpoint_shards,
    run_checkpointed_benchmark,
    evaluate_benchmark_predictions,
    BenchmarkAggregateRecord,
    BenchmarkHypothesisDecision,
    evaluate_benchmark_hypotheses,
    write_benchmark_summary,
    audit_formal_checkpoint_records,
    render_formal_benchmark_report,
)
from experiments.pu_dcgp.benchmark_postformal import (
    build_postformal_diagnostics,
    diagnose_coverage_calibration,
    diagnose_prediction,
    diagnose_retained_power,
    diagnose_shape_recovery,
    diagnose_unsupported_admission,
)
from experiments.pu_dcgp.benchmark_postformal_report import (
    build_postformal_paper_tables,
    render_postformal_paper_tables,
    render_coverage_calibration_table,
    render_prediction_table,
    render_retained_power_table,
    render_shape_recovery_table,
    render_unsupported_admission_table,
    write_postformal_paper_tables,
)
from experiments.pu_dcgp.write_postformal_tables import (
    main as write_postformal_tables_main,
)


class StubDataSource(RunDataSource):
    def load(self) -> RunBatch:
        return RunBatch(
            run_ids=("run-1",),
            groups=("A",),
            doe_modules=("stub",),
            treatment_names=("current_a",),
            treatment_values=np.array([[500.0]]),
            controlled_process_names=("hydrogen_setting",),
            controlled_process_values=np.array([[2.5]]),
            context_names=("execution_order",),
            context_values=np.array([[1.0]]),
            particle_samples={"temperature_c": (np.array([100.0, 110.0]),)},
        )


class StubEncoder(DistributionEncoder):
    def fit_transform(self, runs: RunBatch) -> DistributionRepresentation:
        return DistributionRepresentation(
            run_ids=runs.run_ids,
            outcome_names=("temperature_c",),
            quantile_grid=np.array([0.25, 0.5, 0.75]),
            scores={"temperature_c": np.zeros((1, 1))},
            score_variances={"temperature_c": np.ones((1, 1))},
        )

    def transform(self, runs: RunBatch) -> DistributionRepresentation:
        return self.fit_transform(runs)

    def inverse_transform(self, prediction: ScorePrediction) -> DistributionPrediction:
        return DistributionPrediction(
            quantile_grid=np.array([0.25, 0.5, 0.75]),
            means={"temperature_c": prediction.means["temperature_c"]},
            variances={"temperature_c": prediction.variances["temperature_c"]},
        )


class StubModel(DistributionResponseModel):
    def __init__(self) -> None:
        self.fitted_data = None

    def fit(self, data) -> None:
        self.fitted_data = data

    def predict(self, treatments, contexts) -> ScorePrediction:
        return ScorePrediction(
            means={"temperature_c": np.zeros((1, 1))},
            variances={"temperature_c": np.ones((1, 1))},
        )


class StubContrastEstimator(CausalContrastEstimator):
    def estimate(self, model, encoder, contrast) -> DistributionEffect:
        grid = np.array([0.25, 0.5, 0.75])
        zero = np.zeros(3)
        return DistributionEffect(
            treatment_name=contrast.treatment_name,
            quantile_grid=grid,
            effects={"temperature_c": zero},
            lower_bounds={"temperature_c": zero},
            upper_bounds={"temperature_c": zero},
        )


class StubEvaluator(ModelEvaluator):
    def evaluate(self, model, encoder, data) -> EvaluationResult:
        return EvaluationResult(metrics={"stub_metric": 0.0})


class ScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = StubModel()
        self.workflow = PUDCGPWorkflow(
            config=PUDCGPConfig(),
            data_source=StubDataSource(),
            encoder=StubEncoder(),
            model=self.model,
            contrast_estimator=StubContrastEstimator(),
            evaluator=StubEvaluator(),
        )

    def test_v1_scope_is_frozen_in_config(self) -> None:
        config = self.workflow.config
        self.assertEqual(config.primary_group, "A")
        self.assertEqual(config.analysis_groups, ("A",))
        self.assertEqual(config.exploratory_treatments, ("spray_distance_mm",))
        self.assertNotIn("spray_distance_mm", config.confirmatory_treatments)
        self.assertEqual(
            config.controlled_process_columns,
            ("hydrogen_setting", "powder_carrier_gas_setting"),
        )
        self.assertEqual(config.controlled_process_values, (2.5, 10.0))

    def test_joint_prediction_contracts_freeze_covariance_axes(self) -> None:
        score = JointScorePrediction(
            means={"temperature_c": np.zeros((3, 2))},
            covariances={"temperature_c": np.zeros((2, 3, 3))},
        )
        distribution = JointDistributionPrediction(
            quantile_grid=np.array([0.25, 0.5, 0.75]),
            means={"temperature_c": np.zeros((3, 3))},
            covariances={"temperature_c": np.zeros((3, 3, 3, 3))},
        )

        self.assertEqual(score.covariances["temperature_c"].shape, (2, 3, 3))
        self.assertEqual(
            distribution.covariances["temperature_c"].shape,
            (3, 3, 3, 3),
        )

    def test_component_contracts_connect(self) -> None:
        prepared = self.workflow.prepare()
        self.workflow.fit(prepared)
        contrast = ContrastSpec(
            treatment_name="current_a",
            reference_value=450.0,
            intervention_value=550.0,
            fixed_treatments={},
            fixed_context={},
        )

        effect = self.workflow.estimate_effect(contrast)
        evaluation = self.workflow.evaluate(prepared)

        self.assertIs(self.model.fitted_data, prepared)
        self.assertEqual(effect.treatment_name, "current_a")
        self.assertEqual(evaluation.metrics, {"stub_metric": 0.0})


class CausalGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = aps_ysz_a_causal_graph()
        cls.runs = ManifestDataSource(groups=("A",)).load()

    def test_graph_is_acyclic_and_contains_fixed_gas_nodes(self) -> None:
        self.assertTrue(self.graph.is_acyclic())
        nodes = {node.name: node for node in self.graph.nodes}
        self.assertEqual(nodes["hydrogen_setting"].fixed_value, 2.5)
        self.assertEqual(nodes["powder_carrier_gas_setting"].fixed_value, 10.0)
        self.assertFalse(nodes["hydrogen_setting"].separately_estimable)
        self.assertFalse(nodes["powder_carrier_gas_setting"].varies_in_a)

    def test_ratio_is_derived_from_hydrogen_and_argon(self) -> None:
        ratio_parents = {
            edge.source
            for edge in self.graph.edges
            if edge.target == "hydrogen_to_argon_ratio"
        }
        self.assertEqual(
            ratio_parents,
            {"hydrogen_setting", "argon_flow_scfh"},
        )
        ratios = hydrogen_to_argon_setting_ratio(self.runs)
        np.testing.assert_allclose(
            np.unique(ratios),
            np.asarray([2.5 / level for level in (120, 110, 100, 90, 80)]),
        )
        self.assertNotIn(
            "hydrogen_to_argon_ratio",
            self.runs.treatment_names,
        )
        predictors = response_predictors(
            self.runs.treatment_values,
            self.runs.context_values,
        )
        self.assertEqual(
            predictors.shape[1],
            len(self.runs.treatment_names) + 1,
        )


class DOEEstimandTest(unittest.TestCase):
    def test_extreme_level_estimands_match_a_group_support(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        estimands = a_group_doe_estimands()

        self.assertEqual(len(estimands), 4)
        self.assertEqual(
            [estimand.claim_role for estimand in estimands],
            ["confirmatory", "confirmatory", "confirmatory", "exploratory"],
        )
        for estimand in estimands:
            column = runs.treatment_names.index(estimand.treatment_name)
            observed_levels = set(runs.treatment_values[:, column])
            self.assertIn(estimand.reference_value, observed_levels)
            self.assertIn(estimand.intervention_value, observed_levels)
        self.assertIn("H2/Ar", estimands[1].derived_reexpression)

    def test_all_four_contrasts_have_conditional_matched_support(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        audits = [
            audit_contrast_support(runs, estimand)
            for estimand in a_group_doe_estimands()
        ]

        self.assertEqual([len(audit.strata) for audit in audits], [9, 9, 9, 9])
        self.assertEqual(
            [audit.median_absolute_sequence_gap for audit in audits],
            [8.0, 4.0, 1.0, 2.0],
        )
        self.assertEqual(
            [(audit.reference_runs, audit.intervention_runs) for audit in audits],
            [(21, 21), (21, 21), (21, 21), (24, 24)],
        )
        self.assertTrue(all(audit.support_level == "conditional" for audit in audits))
        self.assertTrue(all(audit.positive_sequence_gaps == 9 for audit in audits))
        self.assertEqual(
            [audit.modules_within_comparison for audit in audits],
            [
                ("DOE-1", "DOE-2"),
                ("DOE-1", "DOE-2"),
                ("DOE-1", "DOE-2"),
                ("DOE-1", "DOE-2", "DOE-4"),
            ],
        )


class MatchedDistributionEffectTest(unittest.TestCase):
    def test_known_distribution_shift_is_recovered(self) -> None:
        treatment_rows = []
        particle_samples = []
        for stratum in range(5):
            for current, shift in ((600.0, 0.0), (800.0, 2.0)):
                treatment_rows.append([current, 80.0 + stratum, 20.0, 100.0])
                particle_samples.append(np.array([1.0, 2.0, 3.0]) + shift)
        runs = RunBatch(
            run_ids=tuple(f"shift-{index}" for index in range(10)),
            groups=("A",) * 10,
            doe_modules=("synthetic",) * 10,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.asarray(treatment_rows),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (10, 1)),
            context_names=("execution_order", "measurement_position_mm"),
            context_values=np.column_stack(
                [np.arange(1, 11), np.full(10, 100.0)]
            ),
            particle_samples={"temperature_c": tuple(particle_samples)},
        )
        result = estimate_matched_distribution_effects(
            runs,
            PUDCGPConfig(),
            a_group_doe_estimands()[0],
        )
        effect = result.aggregate_effects["temperature_c"]

        self.assertEqual(len(result.strata), 5)
        self.assertAlmostEqual(effect.mean_difference, 2.0)
        self.assertAlmostEqual(effect.wasserstein_norm, 2.0)
        np.testing.assert_allclose(effect.quantile_difference, 2.0)
        self.assertTrue(effect.leave_one_out_sign_stable)

    def test_real_a_only_distance_velocity_lacks_leave_one_out_stability(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        results = [
            estimate_matched_distribution_effects(
                runs,
                PUDCGPConfig(),
                estimand,
            )
            for estimand in a_group_doe_estimands()
        ]

        unstable = {
            (result.estimand.estimand_id, outcome)
            for result in results
            for outcome, effect in result.aggregate_effects.items()
            if not effect.leave_one_out_sign_stable
        }
        self.assertEqual(
            unstable,
            {("distance_80_to_120", "velocity_m_s")},
        )


class SequenceSensitivityTest(unittest.TestCase):
    def test_known_treatment_and_sequence_effects_are_separated(self) -> None:
        treatment_rows = []
        context_rows = []
        particle_samples = []
        run_index = 0
        for stratum in range(5):
            for current in (600.0, 800.0, 600.0, 800.0):
                run_index += 1
                sequence_per_10 = run_index / 10.0
                treatment_shift = 2.0 if current == 800.0 else 0.0
                sequence_shift = 0.5 * sequence_per_10
                treatment_rows.append(
                    [current, 80.0 + stratum, 20.0, 100.0]
                )
                context_rows.append([run_index, 100.0])
                particle_samples.append(
                    np.array([1.0, 2.0, 3.0])
                    + stratum
                    + treatment_shift
                    + sequence_shift
                )
        runs = RunBatch(
            run_ids=tuple(f"sequence-{index}" for index in range(run_index)),
            groups=("A",) * run_index,
            doe_modules=("synthetic",) * run_index,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.asarray(treatment_rows),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (run_index, 1)),
            context_names=("execution_order", "measurement_position_mm"),
            context_values=np.asarray(context_rows),
            particle_samples={"temperature_c": tuple(particle_samples)},
        )

        result = estimate_sequence_adjusted_effects(
            runs,
            PUDCGPConfig(),
            a_group_doe_estimands()[0],
        )
        effect = result.outcome_effects["temperature_c"]

        self.assertEqual(result.matched_strata, 5)
        self.assertEqual(result.matched_runs, 20)
        self.assertAlmostEqual(effect.unadjusted_mean_effect, 2.05)
        self.assertAlmostEqual(effect.adjusted_mean_effect, 2.0)
        self.assertAlmostEqual(effect.sequence_slope_per_10_runs, 0.5)
        np.testing.assert_allclose(effect.adjusted_quantile_effect, 2.0)
        self.assertAlmostEqual(effect.adjusted_wasserstein_norm, 2.0)

    def test_real_a_sequence_adjustment_retains_all_effect_directions(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        results = [
            estimate_sequence_adjusted_effects(
                runs,
                PUDCGPConfig(),
                estimand,
            )
            for estimand in a_group_doe_estimands()
        ]

        retained = [
            effect.mean_sign_retained
            for result in results
            for effect in result.outcome_effects.values()
        ]
        self.assertTrue(all(retained))
        self.assertTrue(
            all(result.design_condition_number < 3.2 for result in results)
        )


class ModuleConsistencyTest(unittest.TestCase):
    def test_known_opposite_module_effects_are_recovered(self) -> None:
        treatment_rows = []
        modules = []
        particle_samples = []
        for module_index, module_effect in enumerate((2.0, -1.0), start=1):
            for stratum in range(3):
                for current in (600.0, 800.0):
                    treatment_rows.append(
                        [
                            current,
                            80.0 + 10 * module_index + stratum,
                            20.0,
                            100.0,
                        ]
                    )
                    modules.append(f"DOE-{module_index} synthetic")
                    shift = module_effect if current == 800.0 else 0.0
                    particle_samples.append(
                        np.array([1.0, 2.0, 3.0]) + stratum + shift
                    )
        run_count = len(treatment_rows)
        runs = RunBatch(
            run_ids=tuple(f"module-{index}" for index in range(run_count)),
            groups=("A",) * run_count,
            doe_modules=tuple(modules),
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.asarray(treatment_rows),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (run_count, 1)),
            context_names=("execution_order", "measurement_position_mm"),
            context_values=np.column_stack(
                [np.arange(1, run_count + 1), np.full(run_count, 100.0)]
            ),
            particle_samples={"temperature_c": tuple(particle_samples)},
        )

        result = estimate_module_consistency(
            runs,
            PUDCGPConfig(),
            a_group_doe_estimands()[0],
        )
        consistency = result.outcome_consistency["temperature_c"]

        self.assertTrue(result.all_modules_have_multiple_strata)
        self.assertEqual(consistency.module_mean_effects, {"DOE-1": 2.0, "DOE-2": -1.0})
        self.assertAlmostEqual(consistency.pooled_mean_effect, 0.5)
        self.assertAlmostEqual(consistency.module_balanced_mean_effect, 0.5)
        self.assertFalse(consistency.direction_consistent)
        self.assertFalse(consistency.quantile_direction_consistent)
        self.assertAlmostEqual(consistency.absolute_magnitude_ratio, 2.0)
        np.testing.assert_allclose(
            result.modules[0]
            .outcome_effects["temperature_c"]
            .quantile_difference,
            2.0,
        )

    def test_real_a_module_direction_failures_are_frozen(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        results = [
            estimate_module_consistency(
                runs,
                PUDCGPConfig(),
                estimand,
            )
            for estimand in a_group_doe_estimands()
        ]
        failures = {
            (
                result.estimand.estimand_id,
                outcome,
            )
            for result in results
            for outcome, consistency in result.outcome_consistency.items()
            if not consistency.direction_consistent
        }

        self.assertEqual(
            failures,
            {
                ("distance_80_to_120", "velocity_m_s"),
                ("distance_80_to_120", "particle_diameter_um"),
            },
        )
        self.assertTrue(
            all(not result.all_modules_have_multiple_strata for result in results)
        )
        quantile_failures = {
            (result.estimand.estimand_id, outcome)
            for result in results
            for outcome, consistency in result.outcome_consistency.items()
            if not consistency.quantile_direction_consistent
        }
        self.assertEqual(quantile_failures, failures)


class MatchedMeanUncertaintyTest(unittest.TestCase):
    def test_known_effect_interval_is_recovered_and_reproducible(self) -> None:
        treatment_rows = []
        particle_samples = []
        for stratum in range(5):
            for current, shift in ((600.0, 0.0), (800.0, 2.0)):
                for run_offset in (-1.0, 0.0, 1.0):
                    treatment_rows.append(
                        [current, 80.0 + stratum, 20.0, 100.0]
                    )
                    particle_samples.append(
                        np.array([1.0, 2.0, 3.0])
                        + stratum
                        + run_offset
                        + shift
                    )
        run_count = len(treatment_rows)
        runs = RunBatch(
            run_ids=tuple(f"uncertainty-{index}" for index in range(run_count)),
            groups=("A",) * run_count,
            doe_modules=("DOE-1 synthetic",) * run_count,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.asarray(treatment_rows),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (run_count, 1)),
            context_names=("execution_order", "measurement_position_mm"),
            context_values=np.column_stack(
                [np.arange(1, run_count + 1), np.full(run_count, 100.0)]
            ),
            particle_samples={"temperature_c": tuple(particle_samples)},
        )
        config = PUDCGPConfig(effect_bootstrap_replicates=500)

        first = bootstrap_matched_mean_effects(
            runs, config, a_group_doe_estimands()[0]
        )
        second = bootstrap_matched_mean_effects(
            runs, config, a_group_doe_estimands()[0]
        )
        interval = first.outcome_intervals["temperature_c"]

        self.assertAlmostEqual(interval.point_estimate, 2.0)
        self.assertLess(interval.lower_bound, 2.0)
        self.assertGreater(interval.upper_bound, 2.0)
        self.assertTrue(interval.interval_excludes_zero)
        self.assertGreater(interval.same_sign_probability, 0.99)
        np.testing.assert_array_equal(
            interval.bootstrap_effects,
            second.outcome_intervals["temperature_c"].bootstrap_effects,
        )

    def test_real_a_only_distance_velocity_mean_interval_includes_zero(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        config = PUDCGPConfig(effect_bootstrap_replicates=2000)
        results = [
            bootstrap_matched_mean_effects(runs, config, estimand)
            for estimand in a_group_doe_estimands()
        ]
        excluding_zero = {
            (result.estimand.estimand_id, outcome)
            for result in results
            for outcome, interval in result.outcome_intervals.items()
            if interval.interval_excludes_zero
        }

        all_pairs = {
            (result.estimand.estimand_id, outcome)
            for result in results
            for outcome in result.outcome_intervals
        }
        self.assertEqual(
            all_pairs - excluding_zero,
            {("distance_80_to_120", "velocity_m_s")},
        )
        powder_diameter = results[2].outcome_intervals[
            "particle_diameter_um"
        ]
        self.assertGreater(powder_diameter.lower_bound, 0.0)
        self.assertGreater(powder_diameter.same_sign_probability, 0.99)


class MatchedDistributionUncertaintyTest(unittest.TestCase):
    def test_known_distribution_shift_has_a_simultaneous_band(self) -> None:
        treatment_rows = []
        particle_samples = []
        base_particles = np.linspace(10.0, 20.0, 51)
        for stratum in range(6):
            for powder_feed, shift in ((10.0, 0.0), (30.0, 2.0)):
                for run_offset in (-0.15, -0.05, 0.05, 0.15):
                    treatment_rows.append(
                        [600.0 + stratum, 100.0, powder_feed, 100.0]
                    )
                    particle_samples.append(
                        base_particles
                        + 0.2 * stratum
                        + run_offset
                        + shift
                    )
        run_count = len(treatment_rows)
        runs = RunBatch(
            run_ids=tuple(f"distribution-{index}" for index in range(run_count)),
            groups=("A",) * run_count,
            doe_modules=("DOE-1 synthetic",) * run_count,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.asarray(treatment_rows),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (run_count, 1)),
            context_names=("execution_order", "measurement_position_mm"),
            context_values=np.column_stack(
                [np.arange(1, run_count + 1), np.full(run_count, 100.0)]
            ),
            particle_samples={
                "particle_diameter_um": tuple(particle_samples)
            },
        )
        result = bootstrap_matched_distribution_effect(
            runs,
            PUDCGPConfig(effect_bootstrap_replicates=300),
            a_group_doe_estimands()[2],
            "particle_diameter_um",
        )
        band = result.quantile_band

        self.assertAlmostEqual(result.mean_interval.point_estimate, 2.0)
        self.assertTrue(result.mean_interval.interval_excludes_zero)
        np.testing.assert_allclose(band.point_effect, 2.0)
        self.assertEqual(band.bootstrap_effects.shape, (300, 19))
        self.assertGreater(band.simultaneous_critical_value, 0.0)
        self.assertTrue(all(band.pointwise_excludes_zero))
        self.assertTrue(band.simultaneous_excludes_zero_anywhere)
        self.assertTrue(band.simultaneous_excludes_zero_everywhere)
        self.assertTrue(np.all(band.simultaneous_lower_bound > 0.0))

    def test_real_a_corrected_mapping_admits_nine_confirmatory_edges(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        config = PUDCGPConfig(effect_bootstrap_replicates=2000)
        decisions = evaluate_all_effect_admissions(runs, config)
        retained = {
            (decision.estimand.estimand_id, decision.outcome)
            for decision in decisions
            if decision.status != "abstain"
        }

        self.assertEqual(
            retained,
            {
                ("current_600_to_800", "temperature_c"),
                ("current_600_to_800", "velocity_m_s"),
                ("current_600_to_800", "particle_diameter_um"),
                ("argon_80_to_120", "temperature_c"),
                ("argon_80_to_120", "velocity_m_s"),
                ("argon_80_to_120", "particle_diameter_um"),
                ("powder_10_to_30", "temperature_c"),
                ("powder_10_to_30", "velocity_m_s"),
                ("powder_10_to_30", "particle_diameter_um"),
                ("distance_80_to_120", "temperature_c"),
            },
        )
        admitted = next(
            decision
            for decision in decisions
            if decision.status == "conditional_admit"
        )
        annotations = tuple(
            annotate_physics_consistency(
                decision,
                frozen_existing_physics_evidence(),
            )
            for decision in decisions
        )
        represented = tuple(
            annotation
            for annotation in annotations
            if annotation.consistency_status != "not_represented"
        )
        self.assertEqual(admitted.passed_gates, GATE_NAMES)
        self.assertEqual(admitted.failed_gates, ())
        self.assertTrue(admitted.conditional_reasons)
        self.assertEqual(len(represented), 1)
        self.assertEqual(
            represented[0].decision.estimand.estimand_id,
            "argon_80_to_120",
        )
        self.assertEqual(represented[0].decision.outcome, "velocity_m_s")
        self.assertEqual(
            represented[0].consistency_status,
            "direction_consistent",
        )
        self.assertEqual(
            tuple(annotation.decision.status for annotation in annotations),
            tuple(decision.status for decision in decisions),
        )
        abstentions = {
            (decision.estimand.estimand_id, decision.outcome): set(
                decision.failed_gates
            )
            for decision in decisions
            if decision.status == "abstain"
        }
        self.assertIn(
            "mean_interval",
            abstentions[("distance_80_to_120", "velocity_m_s")],
        )
        self.assertIn(
            "simultaneous_quantile_band",
            abstentions[("distance_80_to_120", "velocity_m_s")],
        )
        self.assertTrue(
            {
                "module_mean_direction",
                "module_quantile_direction",
            }.issubset(
                abstentions[("distance_80_to_120", "particle_diameter_um")]
            )
        )


class EffectAdmissionGateTest(unittest.TestCase):
    def test_pure_decision_distinguishes_exploratory_and_insufficient_support(self) -> None:
        all_pass = EffectAdmissionEvidence(*([True] * len(GATE_NAMES)))
        exploratory = decide_effect_admission(
            a_group_doe_estimands()[3],
            "velocity_m_s",
            "conditional",
            ("estimand was frozen as exploratory",),
            all_pass,
            -3.0,
            -5.0,
            -1.0,
            -6.0,
            -0.5,
        )
        insufficient = decide_effect_admission(
            a_group_doe_estimands()[0],
            "velocity_m_s",
            "insufficient",
            ("fewer than five exact-matching strata",),
            EffectAdmissionEvidence(False, True, True, True, True, True, True),
            2.0,
            1.0,
            3.0,
            0.5,
            4.0,
        )

        self.assertEqual(exploratory.status, "exploratory_admit")
        self.assertEqual(insufficient.status, "insufficient_support")
        self.assertEqual(insufficient.failed_gates, ("structural_support",))

    def test_physics_annotation_is_read_only_for_consistent_and_conflicting_evidence(self) -> None:
        all_pass = EffectAdmissionEvidence(*([True] * len(GATE_NAMES)))
        decision = decide_effect_admission(
            a_group_doe_estimands()[1],
            "velocity_m_s",
            "conditional",
            ("systematic table order",),
            all_pass,
            4.397,
            -2.0,
            10.0,
            -3.0,
            12.0,
        )
        consistent = annotate_physics_consistency(
            decision,
            frozen_existing_physics_evidence(),
        )
        conflicting_evidence = PhysicsDirectionEvidence(
            estimand_id="argon_80_to_120",
            outcome="velocity_m_s",
            direction=-1,
            source_models=("synthetic-conflict",),
            directional_slopes=(-0.2,),
            fidelity="test",
            calibrated_to_current_a=False,
            mechanism_scope="test",
        )
        conflicting = annotate_physics_consistency(
            decision,
            (conflicting_evidence,),
        )

        self.assertEqual(consistent.consistency_status, "direction_consistent")
        self.assertEqual(conflicting.consistency_status, "direction_conflicting")
        self.assertIs(consistent.decision, decision)
        self.assertIs(conflicting.decision, decision)
        self.assertEqual(consistent.decision.status, decision.status)
        self.assertEqual(conflicting.decision.status, decision.status)
        self.assertTrue(consistent.admission_unchanged)
        self.assertTrue(conflicting.admission_unchanged)

    def test_physics_annotation_marks_unmodelled_effect_as_not_represented(self) -> None:
        all_pass = EffectAdmissionEvidence(*([True] * len(GATE_NAMES)))
        decision = decide_effect_admission(
            a_group_doe_estimands()[2],
            "particle_diameter_um",
            "conditional",
            ("systematic table order",),
            all_pass,
            -3.721,
            -6.7,
            -0.7,
            -8.0,
            -0.2,
        )
        annotation = annotate_physics_consistency(
            decision,
            frozen_existing_physics_evidence(),
        )

        self.assertEqual(annotation.consistency_status, "not_represented")
        self.assertIsNone(annotation.physical_direction)
        self.assertEqual(annotation.decision.status, "conditional_admit")


class SyntheticBenchmarkContractTest(unittest.TestCase):
    def test_contract_freezes_methods_scenarios_and_replication(self) -> None:
        contract = pu_dcgp_benchmark_contract()

        self.assertEqual(
            contract.methods,
            (
                "mean_gp",
                "distribution_gp_no_pu",
                "pu_dcgp",
                "support_gated_pu_dcgp",
            ),
        )
        self.assertEqual(contract.pilot_replicate_count, 20)
        self.assertEqual(contract.replicate_count, 200)
        self.assertEqual(contract.interval_level, 0.95)
        self.assertEqual(
            tuple(scenario.scenario_id for scenario in contract.scenarios),
            (
                "identified_balanced_particles",
                "identified_heterogeneous_particles",
                "sequence_aligned_drift",
                "module_sign_reversal",
                "insufficient_overlap",
            ),
        )
        self.assertTrue(
            all(
                scenario.sample_sizes == (48, 96, 144)
                and all(size % 16 == 0 for size in scenario.sample_sizes)
                for scenario in contract.scenarios
            )
        )

    def test_contract_contains_active_null_and_shape_effects(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        active_edges = 0
        null_edges = 0
        shape_edges = 0
        for mechanism in contract.mechanisms:
            self.assertEqual(len(mechanism.location_linear), 4)
            self.assertEqual(len(mechanism.location_cubic), 4)
            self.assertEqual(len(mechanism.log_scale_linear), 4)
            self.assertGreater(mechanism.baseline_scale, 0.0)
            self.assertGreater(mechanism.run_location_sd, 0.0)
            for location, cubic, log_scale in zip(
                mechanism.location_linear,
                mechanism.location_cubic,
                mechanism.log_scale_linear,
            ):
                if location != 0.0 or cubic != 0.0 or log_scale != 0.0:
                    active_edges += 1
                else:
                    null_edges += 1
                if log_scale != 0.0:
                    shape_edges += 1

        self.assertEqual(active_edges, 7)
        self.assertEqual(null_edges, 5)
        self.assertEqual(shape_edges, 4)

        extreme_location_changes = {
            mechanism.outcome: tuple(
                2.0 * (linear + cubic)
                for linear, cubic in zip(
                    mechanism.location_linear,
                    mechanism.location_cubic,
                )
            )
            for mechanism in contract.mechanisms
        }
        self.assertEqual(
            extreme_location_changes,
            {
                "temperature_c": (44.0, 10.0, -18.0, 0.0),
                "velocity_m_s": (4.0, 15.0, 0.0, -6.0),
                "particle_diameter_um": (0.0, 0.0, -5.0, 0.0),
            },
        )

        failure_targets = {
            scenario.scenario_id: (
                scenario.target_treatment,
                scenario.target_outcome,
            )
            for scenario in contract.scenarios[2:]
        }
        self.assertEqual(
            failure_targets,
            {
                "sequence_aligned_drift": (
                    "current_norm",
                    "temperature_c",
                ),
                "module_sign_reversal": (
                    "powder_norm",
                    "particle_diameter_um",
                ),
                "insufficient_overlap": (
                    "argon_norm",
                    "velocity_m_s",
                ),
            },
        )
        sequence, module, overlap = contract.scenarios[2:]
        self.assertEqual(sequence.sequence_confounding_ratio, -1.5)
        self.assertEqual(module.module_effect_multipliers, (1.0, -1.0))
        self.assertEqual(overlap.matched_strata_retained, 4)

    def test_contract_prespecifies_prediction_effect_calibration_and_selection_metrics(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        metric_ids = {metric.metric_id for metric in contract.metrics}

        self.assertEqual(len(contract.quantile_grid), 19)
        self.assertEqual(
            metric_ids,
            {
                "mean_prediction_rmse",
                "wasserstein_prediction_rmse",
                "quantile_effect_irmse",
                "simultaneous_band_coverage",
                "active_admission_rate",
                "null_false_admission_rate",
                "unsupported_false_admission_rate",
            },
        )
        self.assertEqual(len(contract.hypotheses), 4)


class SyntheticBenchmarkGeneratorTest(unittest.TestCase):
    def test_identified_design_has_balanced_anchors_and_interior_points(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        design = generate_identified_design(contract, 48, replicate_index=3)
        anchor_values = design.treatment_values[design.is_factorial_anchor]
        interior_values = design.treatment_values[~design.is_factorial_anchor]
        unique_anchors, anchor_counts = np.unique(
            anchor_values,
            axis=0,
            return_counts=True,
        )

        self.assertEqual(design.treatment_values.shape, (48, 4))
        self.assertEqual(len(anchor_values), 32)
        self.assertEqual(len(interior_values), 16)
        self.assertEqual(len(unique_anchors), 16)
        np.testing.assert_array_equal(anchor_counts, 2)
        self.assertTrue(np.all(np.abs(interior_values) < 0.8))
        np.testing.assert_array_equal(
            design.execution_order,
            np.arange(1.0, 49.0),
        )
        for anchor in unique_anchors:
            rows = np.all(design.treatment_values == anchor, axis=1)
            self.assertEqual(
                {design.doe_modules[index] for index in np.flatnonzero(rows)},
                {"DOE-1 synthetic", "DOE-2 synthetic"},
            )

    def test_clean_design_passes_support_for_all_four_treatments(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        dataset = generate_identified_balanced_dataset(contract, 48, 3)

        for treatment_name in contract.treatment_names:
            support = audit_contrast_support(
                dataset.runs,
                DOEEstimand(
                    estimand_id=f"{treatment_name}_minus1_to_plus1",
                    treatment_name=treatment_name,
                    reference_value=-1.0,
                    intervention_value=1.0,
                    claim_role="confirmatory",
                    effect_direction="plus1_minus_minus1",
                ),
            )
            self.assertEqual(support.support_level, "eligible")
            self.assertEqual(len(support.strata), 8)
            self.assertEqual(
                support.modules_within_comparison,
                ("DOE-1 synthetic", "DOE-2 synthetic"),
            )

    def test_sequence_design_is_conditional_only_for_current(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        for sample_size in (48, 96, 144):
            design = generate_sequence_aligned_design(
                contract,
                sample_size,
                replicate_index=3,
            )
            runs = RunBatch(
                run_ids=tuple(
                    f"sequence-design-{index}" for index in range(sample_size)
                ),
                groups=("synthetic",) * sample_size,
                doe_modules=design.doe_modules,
                treatment_names=contract.treatment_names,
                treatment_values=design.treatment_values,
                controlled_process_names=(),
                controlled_process_values=np.empty((sample_size, 0)),
                context_names=("execution_order",),
                context_values=design.execution_order[:, None],
                particle_samples={
                    "temperature_c": tuple(
                        np.ones(2) for _ in range(sample_size)
                    )
                },
            )
            support_levels = {}
            for treatment_name in contract.treatment_names:
                support = audit_contrast_support(
                    runs,
                    DOEEstimand(
                        estimand_id=treatment_name,
                        treatment_name=treatment_name,
                        reference_value=-1.0,
                        intervention_value=1.0,
                        claim_role="confirmatory",
                        effect_direction="plus1_minus_minus1",
                    ),
                )
                support_levels[treatment_name] = support.support_level
            self.assertEqual(support_levels["current_norm"], "conditional")
            self.assertTrue(
                all(
                    support_levels[treatment] == "eligible"
                    for treatment in contract.treatment_names[1:]
                )
            )

    def test_sequence_design_covers_formal_fallback_indices(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        for replicate_index in (21, 97, 140, 199):
            design = generate_sequence_aligned_design(
                contract,
                sample_size=48,
                replicate_index=replicate_index,
            )
            self.assertEqual(design.treatment_values.shape, (48, 4))
            np.testing.assert_array_equal(
                design.execution_order,
                np.arange(1, 49, dtype=float),
            )

    def test_sequence_drift_reverses_raw_current_temperature_direction(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        dataset = generate_sequence_aligned_drift_dataset(contract, 144, 0)
        config = PUDCGPConfig(
            treatment_columns=contract.treatment_names,
            outcome_columns=contract.outcome_names,
            quantile_grid=contract.quantile_grid,
        )
        estimand = DOEEstimand(
            estimand_id="current_norm_minus1_to_plus1",
            treatment_name="current_norm",
            reference_value=-1.0,
            intervention_value=1.0,
            claim_role="confirmatory",
            effect_direction="plus1_minus_minus1",
        )
        matched = estimate_matched_distribution_effects(
            dataset.runs,
            config,
            estimand,
        ).aggregate_effects["temperature_c"]
        adjusted = estimate_sequence_adjusted_effects(
            dataset.runs,
            config,
            estimand,
        ).outcome_effects["temperature_c"]
        truth = next(
            truth
            for truth in dataset.truths
            if truth.treatment_name == "current_norm"
            and truth.outcome == "temperature_c"
        )

        self.assertAlmostEqual(truth.effect[9], 44.0)
        self.assertAlmostEqual(
            dataset.scenario_parameters[
                "drift_contribution_at_matched_gap"
            ],
            -66.0,
        )
        self.assertLess(matched.mean_difference, 0.0)
        self.assertTrue(matched.leave_one_out_sign_stable)
        self.assertGreater(adjusted.adjusted_mean_effect, 0.0)
        self.assertFalse(adjusted.mean_sign_retained)
        self.assertTrue(
            all(
                np.all(samples > 0.0)
                for samples in dataset.runs.particle_samples["temperature_c"]
            )
        )

    def test_sequence_drift_pilot_meets_prespecified_direction_rates(self) -> None:
        audit = audit_sequence_drift_generator(
            pu_dcgp_benchmark_contract()
        )

        self.assertGreaterEqual(
            audit.raw_reversal_rate,
            audit.acceptance_rate,
        )
        self.assertGreaterEqual(
            audit.adjusted_recovery_rate,
            audit.acceptance_rate,
        )
        self.assertGreaterEqual(
            audit.sequence_gate_failure_rate,
            audit.acceptance_rate,
        )
        self.assertEqual(audit.target_conditional_support_rate, 1.0)
        self.assertEqual(audit.non_target_eligible_rate, 1.0)
        self.assertTrue(audit.passed)


    def test_balanced_dataset_is_reproducible_and_has_joint_particle_counts(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        first = generate_identified_balanced_dataset(contract, 48, 2)
        repeated = generate_identified_balanced_dataset(contract, 48, 2)
        different = generate_identified_balanced_dataset(contract, 48, 3)

        np.testing.assert_array_equal(
            first.runs.treatment_values,
            repeated.runs.treatment_values,
        )
        self.assertFalse(
            np.array_equal(
                first.runs.treatment_values,
                different.runs.treatment_values,
            )
        )
        for outcome in contract.outcome_names:
            for run_index in range(48):
                np.testing.assert_array_equal(
                    first.runs.particle_samples[outcome][run_index],
                    repeated.runs.particle_samples[outcome][run_index],
                )
                self.assertEqual(
                    len(first.runs.particle_samples[outcome][run_index]),
                    80,
                )
                self.assertTrue(
                    np.all(
                        first.runs.particle_samples[outcome][run_index] > 0.0
                    )
                )

    def test_heterogeneous_counts_change_precision_but_not_sampled_values(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        balanced = generate_identified_balanced_dataset(contract, 144, 4)
        heterogeneous = generate_identified_heterogeneous_dataset(
            contract,
            144,
            4,
        )
        values = heterogeneous.runs.treatment_values
        anchor_rows = np.all(np.abs(values) == 1.0, axis=1)
        powder = values[:, contract.treatment_names.index("powder_norm")]
        distance = values[:, contract.treatment_names.index("distance_norm")]

        np.testing.assert_array_equal(
            balanced.runs.treatment_values,
            heterogeneous.runs.treatment_values,
        )
        self.assertGreaterEqual(heterogeneous.particle_counts.min(), 20)
        self.assertLessEqual(heterogeneous.particle_counts.max(), 240)
        self.assertGreater(len(np.unique(heterogeneous.particle_counts)), 20)
        self.assertLess(
            heterogeneous.particle_counts[anchor_rows & (powder == 1.0)].mean(),
            heterogeneous.particle_counts[anchor_rows & (powder == -1.0)].mean(),
        )
        self.assertGreater(
            heterogeneous.particle_counts[
                anchor_rows & (distance == 1.0)
            ].mean(),
            heterogeneous.particle_counts[
                anchor_rows & (distance == -1.0)
            ].mean(),
        )
        for balanced_truth, heterogeneous_truth in zip(
            balanced.truths,
            heterogeneous.truths,
        ):
            np.testing.assert_array_equal(
                balanced_truth.effect,
                heterogeneous_truth.effect,
            )
        for outcome in contract.outcome_names:
            for run_index, count in enumerate(
                heterogeneous.particle_counts
            ):
                heterogeneous_samples = (
                    heterogeneous.runs.particle_samples[outcome][run_index]
                )
                balanced_samples = (
                    balanced.runs.particle_samples[outcome][run_index]
                )
                self.assertEqual(len(heterogeneous_samples), count)
                shared_count = min(
                    len(balanced_samples),
                    len(heterogeneous_samples),
                )
                np.testing.assert_array_equal(
                    balanced_samples[:shared_count],
                    heterogeneous_samples[:shared_count],
                )

    def test_analytic_truths_match_frozen_location_and_shape_structure(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        truths = benchmark_quantile_truths(contract)
        truth_map = {
            (truth.treatment_name, truth.outcome): truth
            for truth in truths
        }
        expected_medians = {
            ("current_norm", "temperature_c"): 44.0,
            ("argon_norm", "temperature_c"): 10.0,
            ("powder_norm", "temperature_c"): -18.0,
            ("current_norm", "velocity_m_s"): 4.0,
            ("argon_norm", "velocity_m_s"): 15.0,
            ("distance_norm", "velocity_m_s"): -6.0,
            ("powder_norm", "particle_diameter_um"): -5.0,
        }

        self.assertEqual(len(truths), 12)
        self.assertEqual(sum(truth.is_active for truth in truths), 7)
        self.assertEqual(
            sum(
                truth.is_active and np.ptp(truth.effect) > 1e-10
                for truth in truths
            ),
            4,
        )
        for key, truth in truth_map.items():
            if key in expected_medians:
                self.assertAlmostEqual(
                    truth.effect[9],
                    expected_medians[key],
                )
            else:
                np.testing.assert_allclose(truth.effect, 0.0, atol=1e-12)

        baseline_quantiles = analytic_expected_run_quantiles(
            contract.mechanisms[0],
            np.zeros((1, 4)),
            np.asarray((0.05, 0.5, 0.95)),
        )[0]
        self.assertAlmostEqual(baseline_quantiles[1], 1800.0)
        self.assertAlmostEqual(
            baseline_quantiles[0] + baseline_quantiles[2],
            3600.0,
        )

    def test_module_reversal_truth_is_opposite_with_zero_aggregate(self) -> None:
        aggregate_truths, module_truths = module_reversal_quantile_truths(
            pu_dcgp_benchmark_contract()
        )
        aggregate = next(
            truth
            for truth in aggregate_truths
            if truth.treatment_name == "powder_norm"
            and truth.outcome == "particle_diameter_um"
        )

        self.assertEqual(
            tuple(truth.effect_multiplier for truth in module_truths),
            (1.0, -1.0),
        )
        self.assertAlmostEqual(module_truths[0].effect[9], -5.0)
        self.assertAlmostEqual(module_truths[1].effect[9], 5.0)
        np.testing.assert_allclose(
            module_truths[0].effect,
            -module_truths[1].effect,
        )
        np.testing.assert_allclose(aggregate.effect, 0.0, atol=1e-12)

    def test_module_reversal_changes_only_target_mechanism(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        baseline = generate_identified_heterogeneous_dataset(contract, 144, 2)
        reversed_modules = generate_module_sign_reversal_dataset(
            contract,
            144,
            2,
        )

        np.testing.assert_array_equal(
            baseline.runs.treatment_values,
            reversed_modules.runs.treatment_values,
        )
        np.testing.assert_array_equal(
            baseline.particle_counts,
            reversed_modules.particle_counts,
        )
        self.assertEqual(
            baseline.runs.doe_modules,
            reversed_modules.runs.doe_modules,
        )
        for outcome in ("temperature_c", "velocity_m_s"):
            for baseline_samples, reversed_samples in zip(
                baseline.runs.particle_samples[outcome],
                reversed_modules.runs.particle_samples[outcome],
            ):
                np.testing.assert_array_equal(
                    baseline_samples,
                    reversed_samples,
                )
        for run_index, module in enumerate(reversed_modules.runs.doe_modules):
            if module.startswith("DOE-1"):
                np.testing.assert_array_equal(
                    baseline.runs.particle_samples[
                        "particle_diameter_um"
                    ][run_index],
                    reversed_modules.runs.particle_samples[
                        "particle_diameter_um"
                    ][run_index],
                )

    def test_module_reversal_is_detected_for_powder_diameter(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        dataset = generate_module_sign_reversal_dataset(contract, 144, 0)
        config = PUDCGPConfig(
            treatment_columns=contract.treatment_names,
            outcome_columns=contract.outcome_names,
            quantile_grid=contract.quantile_grid,
        )
        estimand = DOEEstimand(
            estimand_id="powder_norm_minus1_to_plus1",
            treatment_name="powder_norm",
            reference_value=-1.0,
            intervention_value=1.0,
            claim_role="confirmatory",
            effect_direction="plus1_minus_minus1",
        )
        result = estimate_module_consistency(
            dataset.runs,
            config,
            estimand,
        )
        consistency = result.outcome_consistency[
            "particle_diameter_um"
        ]

        self.assertLess(
            consistency.module_mean_effects["DOE-1"],
            0.0,
        )
        self.assertGreater(
            consistency.module_mean_effects["DOE-2"],
            0.0,
        )
        self.assertFalse(consistency.direction_consistent)
        self.assertFalse(consistency.quantile_direction_consistent)
        self.assertLess(abs(consistency.module_balanced_mean_effect), 1.0)
        aggregate_truth = next(
            truth
            for truth in dataset.truths
            if truth.treatment_name == "powder_norm"
            and truth.outcome == "particle_diameter_um"
        )
        np.testing.assert_allclose(aggregate_truth.effect, 0.0, atol=1e-12)

    def test_module_reversal_pilot_meets_prespecified_failure_rates(self) -> None:
        audit = audit_module_reversal_generator(
            pu_dcgp_benchmark_contract()
        )

        self.assertGreaterEqual(
            audit.doe_1_sign_recovery_rate,
            audit.acceptance_rate,
        )
        self.assertGreaterEqual(
            audit.doe_2_sign_recovery_rate,
            audit.acceptance_rate,
        )
        self.assertGreaterEqual(
            audit.module_mean_gate_failure_rate,
            audit.acceptance_rate,
        )
        self.assertGreaterEqual(
            audit.module_quantile_gate_failure_rate,
            audit.acceptance_rate,
        )
        self.assertEqual(audit.target_eligible_support_rate, 1.0)
        self.assertEqual(audit.non_target_eligible_rate, 1.0)
        self.assertTrue(audit.passed)

    def test_overlap_design_retains_four_target_and_five_non_target_strata(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        for sample_size in (48, 96, 144):
            design = generate_insufficient_overlap_design(
                contract,
                sample_size,
                replicate_index=3,
            )
            runs = RunBatch(
                run_ids=tuple(
                    f"overlap-design-{index}" for index in range(sample_size)
                ),
                groups=("synthetic",) * sample_size,
                doe_modules=design.doe_modules,
                treatment_names=contract.treatment_names,
                treatment_values=design.treatment_values,
                controlled_process_names=(),
                controlled_process_values=np.empty((sample_size, 0)),
                context_names=("execution_order",),
                context_values=design.execution_order[:, None],
                particle_samples={
                    "velocity_m_s": tuple(
                        np.ones(2) for _ in range(sample_size)
                    )
                },
            )
            support = {}
            for treatment_name in contract.treatment_names:
                support[treatment_name] = audit_contrast_support(
                    runs,
                    DOEEstimand(
                        estimand_id=treatment_name,
                        treatment_name=treatment_name,
                        reference_value=-1.0,
                        intervention_value=1.0,
                        claim_role="confirmatory",
                        effect_direction="plus1_minus_minus1",
                    ),
                )
            self.assertEqual(len(support["argon_norm"].strata), 4)
            self.assertEqual(
                support["argon_norm"].support_level,
                "insufficient",
            )
            self.assertIn(
                "fewer than five exact-matching strata",
                support["argon_norm"].support_reasons,
            )
            self.assertTrue(
                all(
                    len(support[treatment].strata) == 5
                    and support[treatment].support_level != "insufficient"
                    for treatment in (
                        "current_norm",
                        "powder_norm",
                        "distance_norm",
                    )
                )
            )

    def test_overlap_transform_changes_only_selected_argon_settings(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        baseline = generate_identified_heterogeneous_dataset(contract, 144, 2)
        overlap = generate_insufficient_overlap_dataset(contract, 144, 2)
        changed = baseline.runs.treatment_values != overlap.runs.treatment_values

        self.assertEqual(np.sum(changed), 24)
        self.assertTrue(np.all(np.flatnonzero(np.any(changed, axis=0)) == 1))
        np.testing.assert_array_equal(
            baseline.particle_counts,
            overlap.particle_counts,
        )
        self.assertEqual(baseline.runs.doe_modules, overlap.runs.doe_modules)
        for baseline_samples, overlap_samples in zip(
            baseline.runs.particle_samples["particle_diameter_um"],
            overlap.runs.particle_samples["particle_diameter_um"],
        ):
            np.testing.assert_array_equal(baseline_samples, overlap_samples)
        target_truth = next(
            truth
            for truth in overlap.truths
            if truth.treatment_name == "argon_norm"
            and truth.outcome == "velocity_m_s"
        )
        self.assertAlmostEqual(target_truth.effect[9], 15.0)

    def test_overlap_failure_pilot_meets_structural_acceptance(self) -> None:
        audit = audit_overlap_failure_generator(
            pu_dcgp_benchmark_contract()
        )

        self.assertEqual(audit.average_target_matched_strata, 4.0)
        self.assertEqual(audit.target_insufficient_rate, 1.0)
        self.assertEqual(audit.target_only_strata_reason_rate, 1.0)
        self.assertEqual(audit.non_target_not_insufficient_rate, 1.0)
        self.assertTrue(audit.passed)

    def test_pilot_empirical_effects_converge_to_analytic_oracle(self) -> None:
        audit = audit_identified_generator_oracle(
            pu_dcgp_benchmark_contract()
        )

        self.assertEqual(audit.sample_size, 144)
        self.assertEqual(audit.replicate_count, 20)
        self.assertEqual(len(audit.entries), 12)
        self.assertLessEqual(
            audit.maximum_normalized_irmse,
            audit.acceptance_threshold,
        )
        self.assertTrue(audit.passed)

    def test_heterogeneous_count_transform_is_isolated_by_pairing(self) -> None:
        raw_audit = audit_identified_heterogeneous_generator_oracle(
            pu_dcgp_benchmark_contract()
        )
        paired_audit = audit_particle_count_isolation(
            pu_dcgp_benchmark_contract()
        )

        self.assertEqual(
            raw_audit.scenario_id,
            "identified_heterogeneous_particles",
        )
        self.assertLessEqual(
            paired_audit.maximum_normalized_irmse,
            paired_audit.acceptance_threshold,
        )
        self.assertTrue(paired_audit.passed)


class PostFormalDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _aggregate(
        scenario_id: str,
        sample_size: int,
        method_name: str,
        shape_error: float,
        coverage: float,
    ) -> BenchmarkAggregateRecord:
        return BenchmarkAggregateRecord(
            scenario_id=scenario_id,
            sample_size=sample_size,
            method_name=method_name,
            replicate_count=200,
            median_normalized_irmse=shape_error,
            median_shape_normalized_irmse=shape_error,
            simultaneous_coverage_rate=coverage,
            active_coverage_rate=coverage,
            shape_coverage_rate=coverage,
            normalized_mean_band_width=0.4,
            active_admission_rate=0.7,
            null_false_admission_rate=0.01,
            target_unsupported_admission_rate=None,
            median_runtime_seconds=1.0,
            normalized_mean_prediction_rmse=0.2,
            normalized_wasserstein_prediction_rmse=(
                None if method_name == "mean_gp" else 0.25
            ),
            median_prediction_validation_seconds=2.0,
        )

    def test_h1_and_h2_diagnostics_preserve_cell_evidence(self) -> None:
        records = []
        for scenario_id in (
            "identified_balanced_particles",
            "identified_heterogeneous_particles",
        ):
            coverage = (
                (0.90, 0.92)
                if scenario_id == "identified_balanced_particles"
                else (0.80, 0.85)
            )
            records.extend(
                (
                    self._aggregate(scenario_id, 48, "mean_gp", 0.20, 0.70),
                    self._aggregate(
                        scenario_id,
                        48,
                        "distribution_gp_no_pu",
                        0.10,
                        coverage[0],
                    ),
                    self._aggregate(
                        scenario_id,
                        48,
                        "pu_dcgp",
                        0.08,
                        coverage[1],
                    ),
                )
            )

        recovery = diagnose_shape_recovery(tuple(records), (48,))
        calibration = diagnose_coverage_calibration(tuple(records), (48,))

        self.assertEqual(len(recovery), 2)
        for cell in recovery:
            self.assertAlmostEqual(cell.no_pu_relative_reduction, 0.5)
            self.assertAlmostEqual(cell.pu_relative_reduction, 0.6)
        self.assertEqual(len(calibration), 1)
        self.assertAlmostEqual(
            calibration[0].balanced_error_worsening,
            -0.02,
        )
        self.assertAlmostEqual(
            calibration[0].heterogeneous_error_reduction,
            0.05,
        )

    def test_h3_and_h4_diagnostics_preserve_pooled_evidence(self) -> None:
        records = []
        for scenario_id in (
            "sequence_aligned_drift",
            "module_sign_reversal",
            "insufficient_overlap",
        ):
            records.extend(
                (
                    replace(
                        self._aggregate(
                            scenario_id,
                            48,
                            "pu_dcgp",
                            0.1,
                            0.9,
                        ),
                        target_unsupported_admission_rate=0.4,
                    ),
                    replace(
                        self._aggregate(
                            scenario_id,
                            48,
                            "support_gated_pu_dcgp",
                            0.1,
                            0.9,
                        ),
                        target_unsupported_admission_rate=0.1,
                    ),
                )
            )
        for scenario_id, power, false_rate in (
            ("identified_balanced_particles", 0.7, 0.01),
            ("identified_heterogeneous_particles", 0.9, 0.03),
        ):
            records.append(
                replace(
                    self._aggregate(
                        scenario_id,
                        144,
                        "support_gated_pu_dcgp",
                        0.1,
                        0.9,
                    ),
                    active_admission_rate=power,
                    null_false_admission_rate=false_rate,
                )
            )

        admission = diagnose_unsupported_admission(tuple(records), (48,))
        power = diagnose_retained_power(tuple(records))

        self.assertEqual(len(admission), 1)
        self.assertAlmostEqual(admission[0].ungated_rate, 0.4)
        self.assertAlmostEqual(admission[0].gated_rate, 0.1)
        self.assertAlmostEqual(admission[0].relative_reduction, 0.75)
        self.assertAlmostEqual(power.active_admission_power, 0.8)
        self.assertAlmostEqual(power.null_false_admission, 0.02)

    def test_prediction_diagnostics_keep_endpoints_separate(self) -> None:
        records = []
        for scenario_id in (
            "identified_balanced_particles",
            "identified_heterogeneous_particles",
        ):
            records.extend(
                (
                    replace(
                        self._aggregate(
                            scenario_id,
                            48,
                            "mean_gp",
                            0.2,
                            0.9,
                        ),
                        normalized_mean_prediction_rmse=0.25,
                    ),
                    replace(
                        self._aggregate(
                            scenario_id,
                            48,
                            "distribution_gp_no_pu",
                            0.1,
                            0.9,
                        ),
                        normalized_wasserstein_prediction_rmse=0.20,
                    ),
                    replace(
                        self._aggregate(
                            scenario_id,
                            48,
                            "pu_dcgp",
                            0.1,
                            0.9,
                        ),
                        normalized_mean_prediction_rmse=0.20,
                        normalized_wasserstein_prediction_rmse=0.18,
                    ),
                )
            )

        prediction = diagnose_prediction(tuple(records), (48,))

        self.assertEqual(len(prediction), 2)
        for cell in prediction:
            self.assertAlmostEqual(cell.pu_mean_rmse_relative_reduction, 0.20)
            self.assertAlmostEqual(
                cell.pu_wasserstein_relative_reduction,
                0.10,
            )

    def test_complete_postformal_payload_joins_all_evidence(self) -> None:
        records = []
        for scenario_id in (
            "identified_balanced_particles",
            "identified_heterogeneous_particles",
        ):
            records.extend(
                self._aggregate(scenario_id, 48, method, error, coverage)
                for method, error, coverage in (
                    ("mean_gp", 0.20, 0.70),
                    ("distribution_gp_no_pu", 0.10, 0.85),
                    ("pu_dcgp", 0.08, 0.87),
                    ("support_gated_pu_dcgp", 0.08, 0.87),
                )
            )
        for scenario_id in (
            "sequence_aligned_drift",
            "module_sign_reversal",
            "insufficient_overlap",
        ):
            records.extend(
                replace(
                    self._aggregate(
                        scenario_id,
                        48,
                        method,
                        0.10,
                        0.90,
                    ),
                    target_unsupported_admission_rate=rate,
                )
                for method, rate in (
                    ("pu_dcgp", 0.4),
                    ("support_gated_pu_dcgp", 0.1),
                )
            )
        decisions = tuple(
            BenchmarkHypothesisDecision(
                hypothesis_id=f"H{index}",
                status=status,
                evidence={},
                reason="test",
            )
            for index, status in enumerate(
                ("pass", "fail", "pass", "fail"),
                start=1,
            )
        )

        diagnostics = build_postformal_diagnostics(
            tuple(records),
            decisions,
            (48,),
            retained_power_sample_size=48,
        )

        self.assertEqual(
            diagnostics.hypothesis_statuses,
            (("H1", "pass"), ("H2", "fail"), ("H3", "pass"), ("H4", "fail")),
        )
        self.assertEqual(len(diagnostics.shape_recovery), 2)
        self.assertEqual(len(diagnostics.coverage_calibration), 1)
        self.assertEqual(len(diagnostics.unsupported_admission), 1)
        self.assertEqual(len(diagnostics.prediction), 2)
        recovery_table = render_shape_recovery_table(diagnostics)
        calibration_table = render_coverage_calibration_table(diagnostics)
        self.assertEqual(recovery_table.markdown.count("\n"), 3)
        self.assertIn("50.0%", recovery_table.markdown)
        self.assertEqual(calibration_table.markdown.count("\n"), 2)
        self.assertIn("Heterogeneous error reduction", calibration_table.markdown)
        admission_table = render_unsupported_admission_table(diagnostics)
        power_table = render_retained_power_table(diagnostics)
        prediction_table = render_prediction_table(diagnostics)
        self.assertIn("75.0%", admission_table.markdown)
        self.assertIn("0.7000", power_table.markdown)
        self.assertEqual(prediction_table.markdown.count("\n"), 3)
        self.assertIn("Wasserstein reduction", prediction_table.markdown)
        tables = build_postformal_paper_tables(diagnostics)
        rendered = render_postformal_paper_tables(
            tables,
            diagnostics.hypothesis_statuses,
        )
        self.assertEqual(
            tuple(section.section_id for section in tables.sections),
            (
                "h1_shape_recovery",
                "h2_coverage_calibration",
                "h3_unsupported_admission",
                "h4_retained_power",
                "prediction_endpoints",
            ),
        )
        self.assertIn("H1 `pass`, H2 `fail`, H3 `pass`, H4 `fail`", rendered)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paper_tables.md"
            write_postformal_paper_tables(path, diagnostics)
            self.assertEqual(path.read_text(encoding="utf-8"), rendered)

    def test_postformal_cli_rejects_incomplete_summary(self) -> None:
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            output = Path(directory) / "tables.md"
            summary.write_text(
                json.dumps({"formal_complete": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete formal summary"):
                write_postformal_tables_main(
                    [str(summary), "--output", str(output)]
                )
            self.assertFalse(output.exists())


class BenchmarkMethodAdapterTest(unittest.TestCase):
    @staticmethod
    def _pilot_config(contract):
        return benchmark_pilot_config(contract)

    @staticmethod
    def _pilot_results():
        contract = pu_dcgp_benchmark_contract()
        dataset = generate_identified_balanced_dataset(contract, 48, 0)
        config = BenchmarkMethodAdapterTest._pilot_config(contract)
        return contract, dataset, fit_benchmark_point_effect_methods(
            dataset,
            contract,
            config,
        )

    def test_three_point_adapters_return_aligned_effect_contracts(self) -> None:
        contract, dataset, results = self._pilot_results()
        truth_keys = {
            (truth.estimand_id, truth.outcome) for truth in dataset.truths
        }

        self.assertEqual(
            tuple(result.method_name for result in results),
            ("mean_gp", "distribution_gp_no_pu", "pu_dcgp"),
        )
        for result in results:
            self.assertEqual(len(result.effects), 12)
            self.assertEqual(
                {
                    (effect.estimand_id, effect.outcome)
                    for effect in result.effects
                },
                truth_keys,
            )
            self.assertGreaterEqual(result.fit_seconds, 0.0)
            self.assertGreaterEqual(result.prediction_seconds, 0.0)
            for effect in result.effects:
                self.assertEqual(effect.point_effect.shape, (19,))
                self.assertEqual(effect.marginal_variance.shape, (19,))
                self.assertEqual(effect.effect_covariance.shape, (19, 19))
                self.assertTrue(np.all(np.isfinite(effect.point_effect)))
                self.assertTrue(np.all(effect.marginal_variance >= 0.0))
                np.testing.assert_allclose(
                    np.diag(effect.effect_covariance),
                    effect.marginal_variance,
                )
                np.testing.assert_allclose(
                    effect.effect_covariance,
                    effect.effect_covariance.T,
                )
                self.assertGreaterEqual(
                    np.linalg.eigvalsh(effect.effect_covariance).min(),
                    -1e-8,
                )
                self.assertEqual(effect.lower_bound.shape, (19,))
                self.assertEqual(effect.upper_bound.shape, (19,))
                self.assertTrue(np.all(effect.lower_bound <= effect.point_effect))
                self.assertTrue(np.all(effect.upper_bound >= effect.point_effect))
                self.assertEqual(
                    effect.interval_kind,
                    "gaussian_posterior_simultaneous_max_t",
                )

        mean_target = next(
            effect
            for effect in results[0].effects
            if effect.treatment_name == "powder_norm"
            and effect.outcome == "particle_diameter_um"
        )
        distribution_targets = tuple(
            next(
                effect
                for effect in result.effects
                if effect.treatment_name == "powder_norm"
                and effect.outcome == "particle_diameter_um"
            )
            for result in results[1:]
        )
        self.assertAlmostEqual(np.ptp(mean_target.point_effect), 0.0)
        self.assertTrue(
            all(
                np.ptp(effect.point_effect) > 1.0
                for effect in distribution_targets
            )
        )

    def test_gate_adapter_changes_reporting_metadata_only(self) -> None:
        _, _, results = self._pilot_results()
        pu_result = results[2]
        all_pass = EffectAdmissionEvidence(*([True] * len(GATE_NAMES)))
        decisions = []
        for index, effect in enumerate(pu_result.effects):
            evidence = (
                EffectAdmissionEvidence(
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                )
                if index == 0
                else all_pass
            )
            decisions.append(
                decide_effect_admission(
                    DOEEstimand(
                        estimand_id=effect.estimand_id,
                        treatment_name=effect.treatment_name,
                        reference_value=-1.0,
                        intervention_value=1.0,
                        claim_role="confirmatory",
                        effect_direction="plus1_minus_minus1",
                    ),
                    effect.outcome,
                    "insufficient" if index == 0 else "eligible",
                    (),
                    evidence,
                    float(effect.point_effect[9]),
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                )
            )
        gated = apply_admission_decisions(pu_result, tuple(decisions))

        self.assertEqual(gated.method_name, "support_gated_pu_dcgp")
        self.assertFalse(gated.effects[0].reported)
        self.assertEqual(
            gated.effects[0].admission_status,
            "insufficient_support",
        )
        self.assertTrue(all(effect.reported for effect in gated.effects[1:]))
        for source, wrapped in zip(pu_result.effects, gated.effects):
            np.testing.assert_array_equal(
                source.point_effect,
                wrapped.point_effect,
            )
            np.testing.assert_array_equal(
                source.marginal_variance,
                wrapped.marginal_variance,
            )
            np.testing.assert_array_equal(
                source.effect_covariance,
                wrapped.effect_covariance,
            )

    def test_point_method_pilot_audit_is_contract_only(self) -> None:
        audit = audit_point_method_adapters(pu_dcgp_benchmark_contract())

        self.assertEqual(audit.sample_size, 48)
        self.assertEqual(audit.replicate_index, 0)
        self.assertEqual(audit.aligned_effect_count, 12)
        self.assertFalse(audit.comparison_authorized)
        self.assertTrue(audit.passed)
        self.assertTrue(
            all(entry.intervals_available for entry in audit.entries)
        )

    def test_known_truth_metric_plumbing_is_aligned(self) -> None:
        contract, dataset, results = self._pilot_results()

        for result in results:
            metrics = evaluate_benchmark_method(result, dataset, contract)
            self.assertEqual(len(metrics.effect_metrics), 12)
            self.assertEqual(
                sum(metric.is_shape_effect for metric in metrics.effect_metrics),
                4,
            )
            self.assertGreaterEqual(metrics.simultaneous_coverage_rate, 0.0)
            self.assertLessEqual(metrics.simultaneous_coverage_rate, 1.0)
            self.assertGreater(metrics.normalized_mean_band_width, 0.0)
            self.assertGreaterEqual(metrics.runtime_seconds, 0.0)

    def test_grouped_prediction_metrics_cover_the_method_contract(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        dataset = generate_identified_balanced_dataset(contract, 48, 0)
        metrics = evaluate_benchmark_predictions(
            dataset,
            contract,
            self._pilot_config(contract),
            n_folds=3,
        )

        self.assertEqual(set(metrics), set(contract.methods))
        self.assertIsNone(metrics["mean_gp"].normalized_wasserstein_rmse)
        for method_name in contract.methods:
            self.assertGreater(metrics[method_name].normalized_mean_rmse, 0.0)
        for method_name in (
            "distribution_gp_no_pu",
            "pu_dcgp",
            "support_gated_pu_dcgp",
        ):
            self.assertGreater(
                metrics[method_name].normalized_wasserstein_rmse,
                0.0,
            )
        self.assertEqual(
            metrics["support_gated_pu_dcgp"].normalized_mean_rmse,
            metrics["pu_dcgp"].normalized_mean_rmse,
        )
        self.assertEqual(
            metrics[
                "support_gated_pu_dcgp"
            ].normalized_wasserstein_rmse,
            metrics["pu_dcgp"].normalized_wasserstein_rmse,
        )

    def test_five_scenario_band_audit_is_smoke_only(self) -> None:
        audit = audit_method_band_pilot(pu_dcgp_benchmark_contract())

        self.assertEqual(audit.scenario_count, 5)
        self.assertEqual(len(audit.entries), 20)
        self.assertFalse(audit.comparison_authorized)
        self.assertTrue(audit.passed)
        gated_failures = [
            entry
            for entry in audit.entries
            if entry.method_name == "support_gated_pu_dcgp"
            and entry.target_unsupported_admitted is not None
        ]
        self.assertEqual(len(gated_failures), 3)
        self.assertFalse(
            any(entry.target_unsupported_admitted for entry in gated_failures)
        )

    def test_synthetic_gates_detect_three_prespecified_failures(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        config = self._pilot_config(contract)
        cases = (
            (
                generate_sequence_aligned_drift_dataset,
                "current_norm",
                "temperature_c",
                "sequence_sign",
                "abstain",
            ),
            (
                generate_module_sign_reversal_dataset,
                "powder_norm",
                "particle_diameter_um",
                "module_mean_direction",
                "abstain",
            ),
            (
                generate_insufficient_overlap_dataset,
                "argon_norm",
                "velocity_m_s",
                "structural_support",
                "insufficient_support",
            ),
        )
        observations = []
        for generator, treatment, outcome, failed_gate, status in cases:
            dataset = generator(contract, 48, 0)
            pu_result = fit_benchmark_point_effect_methods(
                dataset,
                contract,
                config,
            )[2]
            decisions = evaluate_synthetic_admission_decisions(
                dataset,
                contract,
                config,
                pu_result,
            )
            observations.extend(
                synthetic_admission_observations(dataset, decisions)
            )
            target = next(
                decision
                for decision in decisions
                if decision.estimand.treatment_name == treatment
                and decision.outcome == outcome
            )

            self.assertEqual(target.status, status)
            self.assertIn(failed_gate, target.failed_gates)
        diagnostics = aggregate_gate_power_diagnostics(tuple(observations))
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(
            all(
                set(diagnostic.active_failed_gate_rates) == set(GATE_NAMES)
                for diagnostic in diagnostics
            )
        )

    def test_replicate_runner_aggregates_without_authoring_claims(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        records = run_benchmark_replicates(
            contract,
            self._pilot_config(contract),
            sample_size=48,
            replicate_indices=(20, 21),
            scenario_ids=("identified_balanced_particles",),
        )
        aggregates = aggregate_benchmark_records(records)

        self.assertEqual(len(records), 8)
        self.assertEqual(len(aggregates), 4)
        self.assertTrue(
            all(aggregate.replicate_count == 2 for aggregate in aggregates)
        )
        self.assertEqual(
            {aggregate.method_name for aggregate in aggregates},
            set(contract.methods),
        )
        audit = audit_formal_checkpoint_records(
            records,
            contract,
            formal_benchmark_plan(contract, benchmark_formal_config(contract)),
        )
        self.assertTrue(audit.integrity_passed)
        self.assertFalse(audit.formal_complete)
        self.assertEqual(audit.completed_dataset_count, 2)
        self.assertEqual(audit.missing_dataset_count, 2998)
        report = render_formal_benchmark_report(records, contract)
        self.assertIn("Formal complete: `false`", report)
        self.assertIn("| H1 | not_evaluable |", report)
        signature = benchmark_run_signature(
            contract,
            self._pilot_config(contract),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            append_checkpoint_records(path, signature, records)
            append_checkpoint_records(path, signature, records[:4])
            loaded = load_checkpoint_records(path, signature)
            complete = completed_dataset_keys(loaded, contract.methods)

            self.assertEqual(len(loaded), 8)
            self.assertEqual(len(complete), 2)
            with self.assertRaises(ValueError):
                load_checkpoint_records(path, "different-signature")
            first_shard = Path(directory) / "first.jsonl"
            second_shard = Path(directory) / "second.jsonl"
            merged_path = Path(directory) / "merged.jsonl"
            append_checkpoint_records(first_shard, signature, records[:4])
            append_checkpoint_records(second_shard, signature, records[4:])
            merged = merge_checkpoint_shards(
                (first_shard, second_shard),
                merged_path,
                signature,
            )
            self.assertEqual(len(merged), 8)
            self.assertEqual(
                len(completed_dataset_keys(merged, contract.methods)),
                2,
            )
            summary_path = Path(directory) / "merged.summary.json"
            write_benchmark_summary(
                summary_path,
                merged,
                contract,
                signature,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["completed_dataset_count"], 2)
            self.assertEqual(summary["expected_dataset_count"], 3000)
            self.assertFalse(summary["formal_complete"])
            self.assertTrue(
                all(
                    decision["status"] == "not_evaluable"
                    for decision in summary["hypotheses"]
                )
            )
            conflicting = Path(directory) / "conflicting.jsonl"
            append_checkpoint_records(
                conflicting,
                signature,
                (replace(records[0], median_normalized_irmse=999.0),),
            )
            with self.assertRaises(ValueError):
                merge_checkpoint_shards(
                    (first_shard, conflicting),
                    Path(directory) / "conflict-merge.jsonl",
                    signature,
                )

    def test_checkpointed_runner_skips_completed_dataset(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        config = self._pilot_config(contract)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume.jsonl"
            first = run_checkpointed_benchmark(
                path,
                contract,
                config,
                sample_sizes=(48,),
                replicate_indices=(0,),
                scenario_ids=("identified_balanced_particles",),
            )
            size_after_first = path.stat().st_size
            second = run_checkpointed_benchmark(
                path,
                contract,
                config,
                sample_sizes=(48,),
                replicate_indices=(0,),
                scenario_ids=("identified_balanced_particles",),
            )

            self.assertEqual(len(first), 4)
            self.assertEqual(second, first)
            self.assertEqual(path.stat().st_size, size_after_first)

    def test_formal_plan_matches_frozen_execution_axes(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        config = benchmark_formal_config(contract)
        plan = formal_benchmark_plan(contract, config)

        self.assertEqual(plan.dataset_count, 3000)
        self.assertEqual(plan.method_record_count, 12000)
        self.assertEqual(plan.sample_sizes, (48, 96, 144))
        self.assertEqual(len(plan.replicate_indices), 200)
        self.assertEqual(plan.replicate_indices[0], 20)
        self.assertEqual(plan.replicate_indices[-1], 219)
        self.assertEqual(plan.gp_hyperparameter_combinations, 60)
        self.assertEqual(config.bootstrap_replicates, 200)
        self.assertEqual(config.posterior_band_draws, 10000)
        self.assertEqual(config.benchmark_cv_folds, 5)

    def test_hypothesis_evaluator_requires_formal_cells_and_frozen_thresholds(self) -> None:
        contract = pu_dcgp_benchmark_contract()
        supported = {
            "identified_balanced_particles",
            "identified_heterogeneous_particles",
        }
        failure = {
            "sequence_aligned_drift",
            "module_sign_reversal",
            "insufficient_overlap",
        }
        aggregates = []
        for scenario in contract.scenarios:
            for sample_size in scenario.sample_sizes:
                for method_name in contract.methods:
                    is_mean = method_name == "mean_gp"
                    is_no_pu = method_name == "distribution_gp_no_pu"
                    is_pu = method_name in {
                        "pu_dcgp",
                        "support_gated_pu_dcgp",
                    }
                    coverage = 0.90
                    if scenario.scenario_id == "identified_balanced_particles":
                        coverage = 0.92 if is_no_pu else 0.93 if is_pu else 0.70
                    if scenario.scenario_id == "identified_heterogeneous_particles":
                        coverage = 0.80 if is_no_pu else 0.90 if is_pu else 0.70
                    target_rate = None
                    if scenario.scenario_id in failure:
                        target_rate = (
                            0.02
                            if method_name == "support_gated_pu_dcgp"
                            else 0.40
                        )
                    aggregates.append(
                        BenchmarkAggregateRecord(
                            scenario_id=scenario.scenario_id,
                            sample_size=sample_size,
                            method_name=method_name,
                            replicate_count=200,
                            median_normalized_irmse=0.10,
                            median_shape_normalized_irmse=(
                                0.20 if is_mean else 0.15
                            ),
                            simultaneous_coverage_rate=coverage,
                            active_coverage_rate=coverage,
                            shape_coverage_rate=coverage,
                            normalized_mean_band_width=0.40,
                            active_admission_rate=(
                                0.85
                                if method_name == "support_gated_pu_dcgp"
                                and scenario.scenario_id in supported
                                else 0.70
                            ),
                            null_false_admission_rate=(
                                0.02
                                if method_name == "support_gated_pu_dcgp"
                                else 0.04
                            ),
                            target_unsupported_admission_rate=target_rate,
                            median_runtime_seconds=1.0,
                            normalized_mean_prediction_rmse=0.20,
                            normalized_wasserstein_prediction_rmse=(
                                None if is_mean else 0.25
                            ),
                            median_prediction_validation_seconds=2.0,
                        )
                    )

        decisions = evaluate_benchmark_hypotheses(tuple(aggregates), contract)
        self.assertEqual(
            tuple(decision.status for decision in decisions),
            ("pass", "pass", "pass", "pass"),
        )
        pilot_decisions = evaluate_benchmark_hypotheses(
            tuple(replace(record, replicate_count=5) for record in aggregates),
            contract,
        )
        self.assertTrue(
            all(decision.status == "not_evaluable" for decision in pilot_decisions)
        )
        low_power = tuple(
            replace(record, active_admission_rate=0.75)
            if record.sample_size == 144
            and record.scenario_id in supported
            and record.method_name == "support_gated_pu_dcgp"
            else record
            for record in aggregates
        )
        self.assertEqual(
            evaluate_benchmark_hypotheses(low_power, contract)[3].status,
            "fail",
        )

    def test_quantile_contrast_uses_reference_intervention_covariance(self) -> None:
        flat_covariance = np.array(
            [
                [2.0, 0.2, 1.0, 0.1],
                [0.2, 3.0, 0.1, 2.0],
                [1.0, 0.1, 5.0, 0.4],
                [0.1, 2.0, 0.4, 7.0],
            ]
        )
        prediction = JointDistributionPrediction(
            quantile_grid=np.array([0.25, 0.75]),
            means={"temperature_c": np.array([[1.0, 2.0], [4.0, 8.0]])},
            covariances={
                "temperature_c": flat_covariance.reshape(2, 2, 2, 2)
            },
        )

        contrast = average_paired_quantile_contrast(
            prediction,
            "temperature_c",
            stratum_count=1,
        )

        np.testing.assert_allclose(contrast.point_effect, [3.0, 6.0])
        np.testing.assert_allclose(
            contrast.covariance,
            [[5.0, 0.4], [0.4, 6.0]],
        )
        np.testing.assert_allclose(contrast.marginal_variance, [5.0, 6.0])


class ManifestDataSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ManifestDataSource(groups=("A",))
        cls.manifest = cls.source.read_manifest()
        cls.data = cls.source.load()

    def test_manifest_matches_executed_run_structure(self) -> None:
        runs = self.manifest["runs"]
        a_runs = [run for run in runs if run["group"] == "A"]

        self.assertEqual(len(a_runs), 150)
        self.assertEqual(
            [run["design_sequence"] for run in a_runs],
            list(range(1, 151)),
        )
        self.assertEqual(
            [run["execution_order"] for run in a_runs],
            list(range(1, 151)),
        )
        self.assertTrue(
            all(
                int(Path(run["dpv_csv"]).stem) == run["design_sequence"]
                for run in a_runs
            )
        )
        self.assertTrue(
            all(
                run["spray_distance_changed"]
                == (run["spray_distance_mm"] != 100)
                for run in a_runs
            )
        )
        data_root = (self.source.manifest_path.parent / self.manifest["data_root"]).resolve()
        self.assertTrue(all((data_root / run["dpv_csv"]).is_file() for run in a_runs))
        self.assertTrue(
            all((data_root / run["process_export"]).is_file() for run in a_runs)
        )

    def test_loader_reads_jointly_valid_particle_triplets(self) -> None:
        self.assertEqual(len(self.data.run_ids), 150)
        self.assertEqual(self.data.groups.count("A"), 150)
        self.assertEqual(
            self.data.controlled_process_names,
            ("hydrogen_setting", "powder_carrier_gas_setting"),
        )
        np.testing.assert_allclose(
            self.data.controlled_process_values,
            np.tile([2.5, 10.0], (150, 1)),
        )
        self.assertEqual(
            self.data.doe_modules,
            tuple(
                run["doe_module"]
                for run in self.manifest["runs"]
                if run["group"] == "A"
            ),
        )

        outcomes = self.data.particle_samples
        for run_index in range(150):
            lengths = {len(samples[run_index]) for samples in outcomes.values()}
            self.assertEqual(len(lengths), 1)
            self.assertGreater(next(iter(lengths)), 0)
            for samples in outcomes.values():
                self.assertTrue(np.all(samples[run_index] > 0))


class DistributionEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a_data = ManifestDataSource(groups=("A",)).load()
        cls.config = PUDCGPConfig(bootstrap_replicates=40)
        cls.encoder = BootstrapWassersteinFPCAEncoder(cls.config)
        cls.a_representation = cls.encoder.fit_transform(cls.a_data)

    def test_two_components_preserve_location_and_shape(self) -> None:
        for state in self.encoder.states.values():
            self.assertEqual(state.n_components, 2)
            self.assertGreaterEqual(
                state.cumulative_explained_variance,
                self.config.fpca_variance_target,
            )

    def test_a_representation_is_aligned_with_runs(self) -> None:
        self.assertEqual(len(self.a_representation.run_ids), 150)
        for outcome, state in self.encoder.states.items():
            self.assertEqual(
                self.a_representation.scores[outcome].shape,
                (150, state.n_components),
            )
            self.assertTrue(
                np.all(self.a_representation.score_variances[outcome] >= 0)
            )

    def test_reconstruction_is_monotone_and_retains_distribution_variation(self) -> None:
        zero_variances = {
            outcome: np.zeros_like(scores)
            for outcome, scores in self.a_representation.scores.items()
        }
        reconstructed = self.encoder.inverse_transform(
            ScorePrediction(
                means=self.a_representation.scores,
                variances=zero_variances,
            )
        )
        empirical = self.encoder.empirical_quantiles(self.a_data)

        for outcome, curves in empirical.items():
            predicted = reconstructed.means[outcome]
            state = self.encoder.states[outcome]
            residual = np.average(
                np.square(curves - predicted),
                axis=1,
                weights=state.quadrature_weights,
            ).mean()
            mean_only = np.average(
                np.square(curves - state.mean_quantile),
                axis=1,
                weights=state.quadrature_weights,
            ).mean()
            self.assertTrue(np.all(np.diff(predicted, axis=1) >= 0))
            self.assertLess(residual / mean_only, 0.01)

    def test_bootstrap_variances_are_reproducible(self) -> None:
        repeated = self.encoder.transform(self.a_data)
        for outcome in self.encoder.states:
            np.testing.assert_allclose(
                repeated.score_variances[outcome],
                self.a_representation.score_variances[outcome],
            )

    def test_joint_reconstruction_matches_marginal_diagonal(self) -> None:
        means = {
            outcome: scores[:3]
            for outcome, scores in self.a_representation.scores.items()
        }
        marginal_score_variances = {
            outcome: np.ones_like(scores)
            for outcome, scores in means.items()
        }
        score_covariances = {
            outcome: np.repeat(
                np.eye(3)[None, :, :],
                scores.shape[1],
                axis=0,
            )
            for outcome, scores in means.items()
        }
        marginal = self.encoder.inverse_transform(
            ScorePrediction(means=means, variances=marginal_score_variances)
        )
        joint = self.encoder.inverse_transform_joint(
            JointScorePrediction(means=means, covariances=score_covariances)
        )

        for outcome in means:
            covariance = joint.covariances[outcome]
            point_count, quantile_count = joint.means[outcome].shape
            flat_covariance = covariance.reshape(
                point_count * quantile_count,
                point_count * quantile_count,
            )
            np.testing.assert_allclose(joint.means[outcome], marginal.means[outcome])
            np.testing.assert_allclose(
                np.diag(flat_covariance).reshape(point_count, quantile_count),
                marginal.variances[outcome],
            )
            np.testing.assert_allclose(flat_covariance, flat_covariance.T)
            self.assertGreaterEqual(
                np.linalg.eigvalsh(flat_covariance).min(),
                -1e-10,
            )

    def test_synthetic_location_and_scale_family_is_reconstructed(self) -> None:
        base_particles = np.linspace(-2.0, 2.0, 81)
        particle_samples = tuple(
            location + scale * base_particles
            for scale in (0.7, 1.3)
            for location in (0.0, 1.0, 2.0, 3.0)
        )
        runs = RunBatch(
            run_ids=tuple(f"synthetic-{index}" for index in range(8)),
            groups=("synthetic",) * 8,
            doe_modules=("synthetic",) * 8,
            treatment_names=(
                "current_a",
                "argon_flow_scfh",
                "powder_feed_g_min",
                "spray_distance_mm",
            ),
            treatment_values=np.zeros((8, 4)),
            controlled_process_names=(
                "hydrogen_setting",
                "powder_carrier_gas_setting",
            ),
            controlled_process_values=np.tile([2.5, 10.0], (8, 1)),
            context_names=("execution_order", "measurement_position_mm"),
            context_values=np.zeros((8, 2)),
            particle_samples={"temperature_c": particle_samples},
        )
        encoder = BootstrapWassersteinFPCAEncoder(
            PUDCGPConfig(bootstrap_replicates=20)
        )
        representation = encoder.fit_transform(runs)
        reconstructed = encoder.inverse_transform(
            ScorePrediction(
                means=representation.scores,
                variances={
                    "temperature_c": np.zeros_like(
                        representation.scores["temperature_c"]
                    )
                },
            )
        )

        np.testing.assert_allclose(
            reconstructed.means["temperature_c"],
            encoder.empirical_quantiles(runs)["temperature_c"],
            atol=1e-12,
        )


class RepeatedSettingAuditTest(unittest.TestCase):
    def test_a_group_repeated_setting_audit_is_complete(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        result = audit_repeated_settings(runs, PUDCGPConfig())

        self.assertEqual((result.repeated_runs, result.repeated_settings), (111, 27))
        self.assertEqual(set(result.same_setting_mean_metrics), set(runs.particle_samples))
        for outcome in runs.particle_samples:
            self.assertTrue(np.isfinite(result.global_mean_metrics[outcome].rmse))
            self.assertTrue(
                np.isfinite(
                    result.same_setting_distribution_metrics[
                        outcome
                    ].wasserstein_rmse
                )
            )

    def test_module_audit_uses_held_out_settings(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        results = audit_module_predictions(
            runs,
            PUDCGPConfig(
                gp_lengthscale_candidates=(1.0,),
                gp_context_lengthscale_candidates=(8.0,),
                gp_signal_variance_candidates=(1.0,),
                gp_noise_variance_candidates=(0.2,),
            ),
            max_folds=3,
        )

        self.assertEqual(len(results), 6)
        center_module = next(result for result in results if result.unique_settings == 1)
        self.assertEqual(center_module.repeated_runs, 18)
        self.assertIsNone(center_module.mean_gp_metrics)
        for result in results:
            if result.mean_gp_metrics is not None:
                self.assertTrue(
                    all(
                        np.isfinite(metrics.rmse)
                        for metrics in result.mean_gp_metrics.values()
                    )
                )

    def test_exact_matching_support_covers_all_four_factors(self) -> None:
        runs = ManifestDataSource(groups=("A",)).load()
        results = audit_matched_factor_support(runs)

        self.assertEqual(
            [result.treatment_name for result in results],
            list(runs.treatment_names),
        )
        self.assertEqual(
            [result.matched_strata for result in results],
            [18, 19, 17, 17],
        )
        self.assertEqual(results[-1].max_levels_per_stratum, 5)


class TemporalDriftAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = ManifestDataSource(groups=("A",)).load()

    def test_center_repeats_support_a_local_order_slope(self) -> None:
        center_indices = np.flatnonzero(
            np.char.startswith(
                np.asarray(self.runs.doe_modules).astype(str),
                "DOE-0",
            )
        )
        result = audit_shared_linear_drift(
            self.runs,
            PUDCGPConfig(),
            center_indices,
            "DOE-0 center repeats",
        )

        self.assertEqual((result.runs, result.repeated_settings), (18, 1))
        for outcome in ("temperature_c", "particle_diameter_um"):
            self.assertLess(
                result.drift_mean_metrics[outcome].rmse,
                result.constant_mean_metrics[outcome].rmse,
            )

    def test_center_order_slope_does_not_generalize(self) -> None:
        noncenter_indices = np.flatnonzero(
            ~np.char.startswith(
                np.asarray(self.runs.doe_modules).astype(str),
                "DOE-0",
            )
        )
        result = audit_shared_linear_drift(
            self.runs,
            PUDCGPConfig(),
            noncenter_indices,
            "Repeated settings outside DOE-0",
        )

        self.assertEqual((result.runs, result.repeated_settings), (93, 27))
        for outcome in self.runs.particle_samples:
            self.assertGreater(
                result.drift_mean_metrics[outcome].rmse,
                result.constant_mean_metrics[outcome].rmse,
            )


class ModuleAnchorAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = ManifestDataSource(groups=("A",)).load()

    def test_only_three_settings_connect_doe_modules(self) -> None:
        anchors = find_shared_module_anchors(self.runs)

        self.assertEqual(len(anchors), 3)
        self.assertEqual(
            [anchor.treatment_setting[-1] for anchor in anchors],
            [80.0, 100.0, 120.0],
        )
        center = anchors[1]
        self.assertEqual(
            dict(center.module_run_counts),
            {"DOE-0": 18, "DOE-3": 6, "DOE-4": 3},
        )

    def test_doe1_to_doe4_offset_generalizes_only_for_velocity(self) -> None:
        result = audit_cross_module_offset(
            self.runs,
            PUDCGPConfig(),
            "DOE-1",
            "DOE-4",
        )

        self.assertEqual(len(result.anchor_settings), 2)
        for fold in result.folds:
            self.assertLess(
                fold.offset_rmse["velocity_m_s"],
                fold.baseline_rmse["velocity_m_s"],
            )
        self.assertLess(
            result.offset_mean_metrics["velocity_m_s"].rmse,
            result.baseline_mean_metrics["velocity_m_s"].rmse,
        )
        for outcome in ("temperature_c", "particle_diameter_um"):
            self.assertGreater(
                result.offset_mean_metrics[outcome].rmse,
                result.baseline_mean_metrics[outcome].rmse,
            )


class MeanBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = ManifestDataSource(groups=("A",)).load()
        cls.folds = grouped_setting_folds(cls.data.treatment_values, 5)
        cls.results = cross_validate_mean_baselines(cls.data)

    def test_replicated_settings_never_cross_fold_boundaries(self) -> None:
        fold_by_row = np.empty(len(self.data.run_ids), dtype=int)
        for fold_index, indices in enumerate(self.folds):
            fold_by_row[indices] = fold_index

        settings_to_folds: dict[tuple[float, ...], set[int]] = {}
        for setting, fold in zip(self.data.treatment_values, fold_by_row):
            settings_to_folds.setdefault(tuple(setting), set()).add(int(fold))

        self.assertEqual([len(indices) for indices in self.folds], [30] * 5)
        self.assertTrue(all(len(folds) == 1 for folds in settings_to_folds.values()))

    def test_all_baselines_return_finite_out_of_fold_predictions(self) -> None:
        self.assertEqual(
            set(self.results),
            {
                "global_mean",
                "linear_ridge",
                "quadratic_ridge",
                "quadratic_ridge_with_order",
            },
        )
        for result in self.results.values():
            self.assertEqual(result.predictions.shape, (150, 3))
            self.assertTrue(np.isfinite(result.predictions).all())
            self.assertTrue(
                all(
                    np.isfinite([metric.mae, metric.rmse, metric.r2]).all()
                    for metric in result.metrics.values()
                )
            )

    def test_quadratic_feature_map_recovers_a_quadratic_surface(self) -> None:
        treatments = np.array(
            [
                [x1, x2]
                for x1 in (-2.0, -1.0, 0.0, 1.0, 2.0)
                for x2 in (-1.5, -0.5, 0.5, 1.5)
            ]
        )
        target = (
            3.0
            + 2.0 * treatments[:, 0]
            - treatments[:, 1]
            + 0.7 * np.square(treatments[:, 0])
            + 0.4 * treatments[:, 0] * treatments[:, 1]
        )[:, None]
        model = PolynomialRidgeMeanModel(degree=2)
        model.fit(treatments, target, np.array([1e-10]))

        np.testing.assert_allclose(model.predict(treatments), target, atol=1e-9)


class SimultaneousBandTest(unittest.TestCase):
    def test_critical_values_match_analytic_gaussian_cases(self) -> None:
        one_dimensional = gaussian_simultaneous_band(
            np.zeros(1),
            np.ones((1, 1)),
            level=0.95,
            draw_count=40000,
            random_seed=2026,
        )
        perfectly_correlated = gaussian_simultaneous_band(
            np.zeros(2),
            np.ones((2, 2)),
            level=0.95,
            draw_count=40000,
            random_seed=2026,
        )
        independent = gaussian_simultaneous_band(
            np.zeros(2),
            np.eye(2),
            level=0.95,
            draw_count=40000,
            random_seed=2026,
        )
        marginal_critical = NormalDist().inv_cdf(0.975)
        independent_critical = NormalDist().inv_cdf(
            (1.0 + np.sqrt(0.95)) / 2.0
        )

        self.assertAlmostEqual(
            one_dimensional.critical_value,
            marginal_critical,
            delta=0.03,
        )
        self.assertAlmostEqual(
            perfectly_correlated.critical_value,
            marginal_critical,
            delta=0.03,
        )
        self.assertAlmostEqual(
            independent.critical_value,
            independent_critical,
            delta=0.03,
        )
        self.assertGreater(
            independent.critical_value,
            perfectly_correlated.critical_value,
        )

    def test_band_is_reproducible_and_keeps_zero_variance_fixed(self) -> None:
        point_effect = np.array([2.0, -1.0])
        covariance = np.array([[4.0, 0.0], [0.0, 0.0]])
        first = gaussian_simultaneous_band(
            point_effect,
            covariance,
            level=0.95,
            draw_count=10000,
            random_seed=91,
        )
        second = gaussian_simultaneous_band(
            point_effect,
            covariance,
            level=0.95,
            draw_count=10000,
            random_seed=91,
        )

        self.assertEqual(first.critical_value, second.critical_value)
        np.testing.assert_array_equal(first.lower_bound, second.lower_bound)
        np.testing.assert_array_equal(first.upper_bound, second.upper_bound)
        self.assertEqual(first.lower_bound[1], point_effect[1])
        self.assertEqual(first.upper_bound[1], point_effect[1])


class ExactGaussianProcessTest(unittest.TestCase):
    def test_matern_gp_recovers_a_smooth_synthetic_curve(self) -> None:
        predictors = np.linspace(-3.0, 3.0, 31)[:, None]
        targets = np.sin(predictors[:, 0])
        model = ExactGaussianProcessRegressor(PUDCGPConfig())
        model.fit(predictors, targets)

        means, variances = model.predict(predictors)

        self.assertLess(np.sqrt(np.mean(np.square(targets - means))), 0.15)
        self.assertTrue(np.all(variances >= 0))
        self.assertIsNotNone(model.hyperparameters)

    def test_known_observation_variance_is_accepted(self) -> None:
        predictors = np.linspace(0.0, 1.0, 12)[:, None]
        targets = np.square(predictors[:, 0])
        model = ExactGaussianProcessRegressor(PUDCGPConfig())
        model.fit(
            predictors,
            targets,
            observation_variance=np.linspace(0.001, 0.02, len(targets)),
        )

        means, variances = model.predict(np.array([[0.25], [0.75]]))
        self.assertEqual(means.shape, (2,))
        self.assertEqual(variances.shape, (2,))

    def test_joint_covariance_matches_marginals_and_is_psd(self) -> None:
        predictors = np.linspace(-3.0, 3.0, 31)[:, None]
        targets = np.sin(predictors[:, 0])
        test_predictors = np.linspace(-2.5, 2.5, 7)[:, None]
        model = ExactGaussianProcessRegressor(PUDCGPConfig())
        model.fit(predictors, targets)

        marginal_means, marginal_variances = model.predict(test_predictors)
        joint_means, joint_covariance = model.predict_joint(test_predictors)

        np.testing.assert_allclose(joint_means, marginal_means)
        np.testing.assert_allclose(
            np.diag(joint_covariance),
            marginal_variances,
        )
        np.testing.assert_allclose(joint_covariance, joint_covariance.T)
        self.assertGreaterEqual(
            np.linalg.eigvalsh(joint_covariance).min(),
            -1e-10,
        )


class GaussianProcessModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = subset_run_batch(
            ManifestDataSource(groups=("A",)).load(),
            np.arange(30),
        )
        cls.config = PUDCGPConfig(
            bootstrap_replicates=20,
            gp_lengthscale_candidates=(1.0, 4.0),
            gp_signal_variance_candidates=(1.0,),
            gp_noise_variance_candidates=(0.2, 1.0),
        )
        cls.encoder = BootstrapWassersteinFPCAEncoder(cls.config)
        cls.representation = cls.encoder.fit_transform(cls.runs)

    def test_mean_gp_predicts_all_three_outcomes(self) -> None:
        model = GaussianProcessMeanModel(self.config)
        model.fit(self.runs)
        means, variances = model.predict(
            self.runs.treatment_values[:4],
            self.runs.context_values[:4],
        )

        self.assertEqual(means.shape, (4, 3))
        self.assertEqual(variances.shape, (4, 3))
        self.assertTrue(np.all(variances >= 0))

        joint_means, joint_covariances = model.predict_joint(
            self.runs.treatment_values[:4],
            self.runs.context_values[:4],
        )
        np.testing.assert_allclose(joint_means, means)
        self.assertEqual(joint_covariances.shape, (3, 4, 4))
        np.testing.assert_allclose(
            np.stack([np.diag(covariance) for covariance in joint_covariances], axis=1),
            variances,
        )

    def test_distribution_gp_variants_predict_score_moments(self) -> None:
        prepared = PreparedData(
            runs=self.runs,
            distributions=self.representation,
        )
        for use_uncertainty in (False, True):
            model = GaussianProcessDistributionModel(
                self.config,
                use_uncertainty,
            )
            model.fit(prepared)
            prediction = model.predict(
                self.runs.treatment_values[:4],
                self.runs.context_values[:4],
            )
            for outcome, state in self.encoder.states.items():
                self.assertEqual(
                    prediction.means[outcome].shape,
                    (4, state.n_components),
                )
                self.assertTrue(np.all(prediction.variances[outcome] >= 0))

    def test_distribution_gp_joint_score_covariance_matches_marginals(self) -> None:
        prepared = PreparedData(
            runs=self.runs,
            distributions=self.representation,
        )
        model = GaussianProcessDistributionModel(self.config, True)
        model.fit(prepared)
        treatments = self.runs.treatment_values[:4]
        contexts = self.runs.context_values[:4]

        marginal = model.predict(treatments, contexts)
        joint = model.predict_joint(treatments, contexts)

        for outcome, state in self.encoder.states.items():
            self.assertEqual(
                joint.covariances[outcome].shape,
                (state.n_components, 4, 4),
            )
            np.testing.assert_allclose(joint.means[outcome], marginal.means[outcome])
            np.testing.assert_allclose(
                np.stack(
                    [
                        np.diag(component_covariance)
                        for component_covariance in joint.covariances[outcome]
                    ],
                    axis=1,
                ),
                marginal.variances[outcome],
            )

    def test_grouped_gp_evaluation_connects_on_real_runs(self) -> None:
        result = cross_validate_gp_models(
            self.runs,
            self.config,
            n_folds=3,
        )

        self.assertEqual(result.mean_gp.predictions.shape, (30, 3))
        self.assertEqual(
            set(result.distribution_models),
            {"distribution_fold_mean", "distribution_gp", "pu_dcgp"},
        )
        for model in result.distribution_models.values():
            for prediction in model.quantile_predictions.values():
                self.assertEqual(prediction.shape, (30, 19))
                self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
