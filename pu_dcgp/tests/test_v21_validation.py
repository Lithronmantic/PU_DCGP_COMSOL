
from hashlib import sha256
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.pu_dcgp_v21 import (
    H3ValidationConfig,
    PUDCGPV21Config,
    StructuralSelectionStressConfig,
    generate_structural_stress_dataset,
    append_structural_stress_result,
    aggregate_structural_stress_results,
    evaluate_structural_stress,
    load_v21_dataset_checkpoints,
    load_structural_stress_results,
    run_structural_stress_dataset,
    structural_stress_run_signature,
    append_v21_h3_checkpoint,
    load_v21_h3_checkpoints,
    pu_dcgp_v21_development_contract,
    run_v21_h3_dataset,
    v21_h3_run_signature,
    evaluate_v21_h3_validation,
    build_v21_readiness_decision,
    V21PhysicsAnnotationPlan,
    V21ReadinessDecision,
    build_v21_physics_annotation_plan,
    pu_dcgp_v21_formal_contract,
    v21_formal_plan,
    V21FormalDatasetCheckpoint,
    append_v21_formal_checkpoint,
    load_v21_formal_checkpoints,
    merge_v21_formal_shards,
    validate_v21_formal_checkpoint,
    v21_formal_run_signature,
    GPUAcceleratedStructuredExactICMGaussianProcessRegressor,
    StructuredExactICMGaussianProcessRegressor,
    V21BenchmarkDatasetResult,
    run_v21_formal_dataset,
    run_checkpointed_v21_formal_keys,
    validate_v21_formal_dataset_keys,
    audit_h8_proxy_physics,
    v21_proxy_direction_evidence,
    write_v21_proxy_physics_audit,
    compare_v21_proxy_consistency,
    write_v21_proxy_consistency_audit,
    audit_v21_formal_checkpoints,
    write_v21_formal_summary,
    build_v21_formal_batch_plan,
    load_v21_formal_batch_manifest,
    write_v21_formal_batch_manifest,
)
from experiments.pu_dcgp.physics_consistency import (
    frozen_existing_physics_evidence,
)
from experiments.pu_dcgp import (
    EffectAdmissionEvidence,
    a_group_doe_estimands,
    decide_effect_admission,
)
from experiments.pu_dcgp_v21.run_formal_evaluation import (
    main as run_formal_main,
)
from experiments.pu_dcgp_v21.run_formal_batch import (
    main as run_formal_batch_main,
    thread_limited_environment,
)
from experiments.pu_dcgp_v21.run_formal_batch_shard import (
    main as run_formal_batch_shard_main,
)
from experiments.pu_dcgp_v21.formal_batch_reporting import (
    audit_v21_formal_batch_progress,
)
from experiments.pu_dcgp_v21.formal_release import (
    V21FormalRelease,
    V21FormalReleasedDecision,
    load_v21_formal_release,
)
from experiments.pu_dcgp_v21.formal_postanalysis import (
    build_v21_formal_postanalysis,
)
from experiments.pu_dcgp_v21.formal_paper_tables import (
    build_v21_formal_paper_tables,
    render_v21_formal_paper_tables,
)
from experiments.pu_dcgp_v21.formal_structure_reporting import (
    aggregate_v21_structure_frequencies,
    load_complete_v21_structure_frequencies,
)
from experiments.pu_dcgp_v21.formal_claim_map import (
    build_v21_formal_claim_map,
    render_v21_formal_claim_map,
)
from experiments.pu_dcgp_v21.formal_finalize import (
    build_v21_final_artifact_payloads,
    finalize_v21_formal_evidence,
)

import numpy as np


class PostSupportedValidationConfigTests(unittest.TestCase):
    def test_structural_stress_seed_is_disjoint_and_two_sided(self):
        stress = StructuralSelectionStressConfig()
        method = PUDCGPV21Config()

        self.assertNotIn(
            stress.random_seed,
            (
                method.development_benchmark_random_seed,
                method.formal_benchmark_random_seed,
            ),
        )
        self.assertEqual(
            stress.scenario_ids,
            ("true_diagonal_latent", "true_full_latent"),
        )
        self.assertEqual(stress.screen_replicate_count, 5)
        self.assertEqual(stress.complete_replicate_count, 20)

    def test_h3_contract_retains_failure_scenarios_and_thresholds(self):
        h3 = H3ValidationConfig()
        method = PUDCGPV21Config()

        self.assertEqual(
            h3.benchmark_random_seed,
            method.development_benchmark_random_seed,
        )
        self.assertEqual(
            h3.unsupported_false_admission_max,
            method.unsupported_false_admission_max,
        )
        self.assertEqual(
            h3.unsupported_relative_reduction,
            method.unsupported_relative_reduction,
        )
        self.assertEqual(len(h3.scenario_ids), 3)

    def test_backend_policy_uses_gpu_only_where_beneficial(self):
        for config in (
            StructuralSelectionStressConfig(),
            H3ValidationConfig(),
        ):
            self.assertEqual(config.backend_for(48), "cpu")
            self.assertEqual(config.backend_for(96), "gpu")
            self.assertEqual(config.backend_for(144), "gpu")


class StructuralStressDatasetTests(unittest.TestCase):
    def test_generator_is_reproducible_and_retains_full_measurement_blocks(self):
        config = StructuralSelectionStressConfig()
        first = generate_structural_stress_dataset(
            config,
            "true_full_latent",
            48,
            0,
        )
        second = generate_structural_stress_dataset(
            config,
            "true_full_latent",
            48,
            0,
        )

        np.testing.assert_allclose(first.predictors, second.predictors)
        np.testing.assert_allclose(first.targets, second.targets)
        np.testing.assert_allclose(
            first.observation_covariances,
            second.observation_covariances,
        )
        self.assertTrue(
            np.all(
                np.abs(first.observation_covariances[:, 0, 1]) > 0.0
            )
        )

    def test_true_structures_are_paired_and_distinct(self):
        config = StructuralSelectionStressConfig()
        diagonal = generate_structural_stress_dataset(
            config,
            "true_diagonal_latent",
            48,
            0,
        )
        full = generate_structural_stress_dataset(
            config,
            "true_full_latent",
            48,
            0,
        )

        np.testing.assert_allclose(diagonal.predictors, full.predictors)
        np.testing.assert_allclose(
            diagonal.observation_covariances,
            full.observation_covariances,
        )
        np.testing.assert_allclose(
            diagonal.true_coregionalization,
            np.eye(config.component_count),
        )
        self.assertGreater(
            abs(full.true_coregionalization[0, 1]),
            0.0,
        )
        self.assertGreater(
            np.linalg.eigvalsh(full.true_coregionalization).min(),
            0.0,
        )

    def test_single_cpu_result_contains_both_candidate_evidence(self):
        stress = StructuralSelectionStressConfig()
        model = PUDCGPV21Config(optimizer_max_iterations=5)

        result = run_structural_stress_dataset(
            stress,
            model,
            "true_diagonal_latent",
            48,
            0,
        )

        self.assertEqual(result.backend, "cpu")
        self.assertIn(result.selected_structure, ("diagonal", "full"))
        self.assertTrue(np.isfinite(result.diagonal_bic))
        self.assertTrue(np.isfinite(result.full_bic))
        self.assertGreater(result.runtime_seconds, 0.0)


class StructuralStressCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.stress = StructuralSelectionStressConfig()
        self.model = PUDCGPV21Config(optimizer_max_iterations=5)
        self.result = run_structural_stress_dataset(
            self.stress,
            self.model,
            "true_diagonal_latent",
            48,
            0,
        )

    def test_one_result_round_trips_atomically(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stress.jsonl"
            append_structural_stress_result(
                path,
                self.result,
                self.stress,
                self.model,
            )

            loaded = load_structural_stress_results(
                path,
                self.stress,
                self.model,
            )

        self.assertEqual(loaded, (self.result,))

    def test_signature_changes_with_stress_or_model_configuration(self):
        signature = structural_stress_run_signature(
            self.stress,
            self.model,
        )

        self.assertNotEqual(
            signature,
            structural_stress_run_signature(
                replace(self.stress, true_full_correlation=0.70),
                self.model,
            ),
        )
        self.assertNotEqual(
            signature,
            structural_stress_run_signature(
                self.stress,
                replace(self.model, optimizer_max_iterations=6),
            ),
        )


class StructuralStressScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = StructuralSelectionStressConfig()
        path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v21"
            / "data"
            / "structural_stress_records.jsonl"
        )
        cls.results = load_structural_stress_results(
            path,
            cls.config,
            PUDCGPV21Config(),
        )
        cls.screen_aggregates = aggregate_structural_stress_results(
            tuple(
                result
                for result in cls.results
                if result.replicate_index
                < cls.config.screen_replicate_count
            )
        )
        cls.complete_aggregates = aggregate_structural_stress_results(
            cls.results
        )

    def test_five_replicate_screen_meets_two_sided_contract(self):
        decision = evaluate_structural_stress(
            self.screen_aggregates,
            self.config,
            "screen",
        )

        self.assertTrue(decision.threshold_met)
        self.assertEqual(decision.status, "screen_threshold_met")
        self.assertEqual(
            decision.evidence["true_full_selection_rate_n48"],
            0.60,
        )
        self.assertEqual(
            decision.evidence["minimum_true_diagonal_selection_rate"],
            1.0,
        )

    def test_screen_is_not_mislabelled_as_complete(self):
        decision = evaluate_structural_stress(
            self.screen_aggregates,
            self.config,
            "complete",
        )

        self.assertIsNone(decision.threshold_met)
        self.assertEqual(decision.status, "not_evaluable")

    def test_twenty_replicate_result_meets_complete_contract(self):
        decision = evaluate_structural_stress(
            self.complete_aggregates,
            self.config,
            "complete",
        )

        self.assertTrue(decision.threshold_met)
        self.assertEqual(decision.status, "complete_threshold_met")
        self.assertEqual(
            decision.evidence["true_full_selection_rate_n48"],
            0.90,
        )
        self.assertEqual(
            decision.evidence[
                "minimum_true_full_selection_rate_n96_n144"
            ],
            0.95,
        )


class H3CheckpointSkeletonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = pu_dcgp_v21_development_contract()
        cls.h3 = H3ValidationConfig()
        cls.model = PUDCGPV21Config(
            quantile_grid=cls.contract.quantile_grid,
            particle_bootstrap_replicates=20,
            posterior_band_draws=500,
            calibration_folds=3,
            calibration_band_draws=100,
            optimize_joint_hyperparameters=False,
        )
        cls.checkpoint = run_v21_h3_dataset(
            cls.contract,
            cls.model,
            cls.h3,
            "sequence_aligned_drift",
            48,
            0,
        )

    def test_one_dataset_has_three_methods_backend_and_full_audit(self):
        checkpoint = self.checkpoint

        self.assertEqual(checkpoint.backend, "cpu")
        self.assertEqual(
            tuple(record.method_name for record in checkpoint.records),
            self.h3.method_names,
        )
        self.assertTrue(
            all(
                record.target_unsupported_admitted is not None
                for record in checkpoint.records
            )
        )
        self.assertEqual(
            len(checkpoint.structure_selections),
            len(self.contract.outcome_names)
            * (1 + self.model.calibration_folds),
        )

    def test_h3_checkpoint_round_trip_and_signature(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "h3.jsonl"
            append_v21_h3_checkpoint(
                path,
                self.checkpoint,
                self.contract,
                self.model,
                self.h3,
            )
            loaded = load_v21_h3_checkpoints(
                path,
                self.contract,
                self.model,
                self.h3,
            )

        self.assertEqual(loaded, (self.checkpoint,))
        self.assertEqual(
            self.checkpoint.run_signature,
            v21_h3_run_signature(
                self.contract,
                self.model,
                self.h3,
            ),
        )


class H3ScreenResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = pu_dcgp_v21_development_contract()
        cls.model = PUDCGPV21Config(
            quantile_grid=cls.contract.quantile_grid
        )
        cls.h3 = H3ValidationConfig()
        path = (
            Path(__file__).resolve().parents[2]
            / "pu_dcgp_v21"
            / "data"
            / "h3_validation_records.jsonl"
        )
        cls.checkpoints = load_v21_h3_checkpoints(
            path,
            cls.contract,
            cls.model,
            cls.h3,
        )
        cls.screen_checkpoints = tuple(
            checkpoint
            for checkpoint in cls.checkpoints
            if checkpoint.dataset_key[2]
            < cls.h3.screen_replicate_count
        )

    def test_five_replicate_screen_supports_expansion_only(self):
        decision = evaluate_v21_h3_validation(
            self.screen_checkpoints,
            self.h3,
            "screen",
        )

        self.assertIsNone(decision.threshold_met)
        self.assertTrue(decision.expansion_recommended)
        self.assertEqual(
            decision.status,
            "screen_direction_supported",
        )
        self.assertEqual(
            decision.evidence[
                "maximum_gated_unsupported_admission"
            ],
            0.0,
        )

    def test_screen_is_not_a_complete_h3_decision(self):
        decision = evaluate_v21_h3_validation(
            self.screen_checkpoints,
            self.h3,
            "complete",
        )

        self.assertIsNone(decision.threshold_met)
        self.assertEqual(decision.status, "not_evaluable")

    def test_twenty_replicate_h3_meets_complete_threshold(self):
        decision = evaluate_v21_h3_validation(
            self.checkpoints,
            self.h3,
            "complete",
        )

        self.assertTrue(decision.threshold_met)
        self.assertEqual(decision.status, "complete_threshold_met")
        self.assertEqual(
            decision.evidence[
                "maximum_gated_unsupported_admission"
            ],
            0.0,
        )
        self.assertEqual(
            decision.evidence[
                "minimum_relative_reduction_from_ungated"
            ],
            1.0,
        )


class V21ReadinessTests(unittest.TestCase):
    def test_complete_validation_is_ready_but_physics_stays_read_only(self):
        contract = pu_dcgp_v21_development_contract()
        model = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        stress_config = StructuralSelectionStressConfig()
        h3_config = H3ValidationConfig()
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        supported = load_v21_dataset_checkpoints(
            data / "development_v21_records.jsonl",
            contract,
            model,
        )
        stress = load_structural_stress_results(
            data / "structural_stress_records.jsonl",
            stress_config,
            model,
        )
        h3 = load_v21_h3_checkpoints(
            data / "h3_validation_records.jsonl",
            contract,
            model,
            h3_config,
        )

        decision = build_v21_readiness_decision(
            supported,
            stress,
            h3,
            contract,
            model,
            stress_config,
            h3_config,
        )

        self.assertTrue(decision.statistical_validation_passed)
        self.assertTrue(decision.selector_sanity_passed)
        self.assertTrue(decision.ready_for_locked_formal_evaluation)
        self.assertTrue(decision.ready_for_read_only_physics_annotation)
        self.assertFalse(decision.physics_may_change_estimator)
        self.assertEqual(decision.formal_replicate_indices_used, 0)


class V21PhysicsBoundaryScaffoldTests(unittest.TestCase):
    def test_interface_is_importable_and_keeps_readiness_attached(self):
        self.assertEqual(
            tuple(V21PhysicsAnnotationPlan.__dataclass_fields__),
            (
                "readiness",
                "status",
                "evidence",
                "allowed_roles",
                "prohibited_roles",
            ),
        )

    def test_ready_plan_is_read_only_and_unready_plan_is_blocked(self):
        ready = V21ReadinessDecision(
            statistical_validation_passed=True,
            selector_sanity_passed=True,
            ready_for_locked_formal_evaluation=True,
            ready_for_read_only_physics_annotation=True,
            physics_may_change_estimator=False,
            formal_replicate_indices_used=0,
            hypothesis_statuses={},
            remaining_requirements=(),
        )
        evidence = frozen_existing_physics_evidence()

        plan = build_v21_physics_annotation_plan(ready, evidence)
        blocked = build_v21_physics_annotation_plan(
            replace(
                ready,
                ready_for_read_only_physics_annotation=False,
            ),
            evidence,
        )

        self.assertEqual(plan.status, "eligible_read_only")
        self.assertEqual(blocked.status, "blocked")
        self.assertIs(plan.readiness, ready)
        self.assertIs(plan.evidence, evidence)
        self.assertIn("falsification", plan.allowed_roles)
        self.assertIn("estimator_input", plan.prohibited_roles)

    def test_h8_proxy_resolves_five_directions_but_not_diameter(self):
        audit = audit_h8_proxy_physics()
        evidence = v21_proxy_direction_evidence(audit)
        resolved = {
            (item.estimand_id, item.outcome, item.direction)
            for item in evidence
        }

        self.assertEqual(audit.row_count, 96)
        self.assertGreater(
            min(item.cv_r2 for item in audit.target_audits),
            0.94,
        )
        self.assertEqual(audit.particle_diameter_constant_um, 40.0)
        self.assertEqual(
            resolved,
            {
                ("current_600_to_800", "temperature_c", 1),
                ("argon_80_to_120", "temperature_c", 1),
                ("powder_10_to_30", "temperature_c", -1),
                ("argon_80_to_120", "velocity_m_s", 1),
                ("powder_10_to_30", "velocity_m_s", -1),
            },
        )
        self.assertNotIn(
            "particle_diameter_um",
            {item.outcome for item in evidence},
        )

        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "audit.json"
            markdown_path = Path(directory) / "audit.md"
            write_v21_proxy_physics_audit(
                audit,
                json_path,
                markdown_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertIn("Five scalar directions are resolved", markdown)
        self.assertIn("no diameter evidence is admitted", markdown)

        passed = EffectAdmissionEvidence(*([True] * 7))
        decisions = (
            decide_effect_admission(
                a_group_doe_estimands()[0],
                "temperature_c",
                "conditional",
                (),
                passed,
                -2.5,
                -5.0,
                1.0,
                -6.0,
                2.0,
            ),
            decide_effect_admission(
                a_group_doe_estimands()[1],
                "temperature_c",
                "conditional",
                (),
                passed,
                5.4,
                1.0,
                9.0,
                0.5,
                10.0,
            ),
        )
        comparison = compare_v21_proxy_consistency(
            decisions,
            evidence,
        )
        self.assertEqual(comparison.direction_consistent_count, 1)
        self.assertEqual(comparison.direction_conflicting_count, 1)

        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "comparison.json"
            markdown_path = Path(directory) / "comparison.md"
            write_v21_proxy_consistency_audit(
                comparison,
                json_path,
                markdown_path,
            )
            comparison_markdown = markdown_path.read_text(
                encoding="utf-8"
            )
        self.assertIn("direction-consistent", comparison_markdown)
        self.assertIn("direction-conflicting", comparison_markdown)


class V21FormalPlanTests(unittest.TestCase):
    def test_formal_axes_are_disjoint_complete_and_backend_frozen(self):
        development = pu_dcgp_v21_development_contract()
        formal = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=formal.quantile_grid)
        plan = v21_formal_plan(formal, config)

        self.assertNotEqual(formal.random_seed, development.random_seed)
        self.assertEqual(formal.methods, development.methods)
        self.assertEqual(plan.replicate_indices, tuple(range(20, 220)))
        self.assertEqual(plan.dataset_count, 3000)
        self.assertEqual(plan.method_record_count, 24000)
        self.assertEqual(plan.structure_selection_count, 54000)
        self.assertEqual(plan.selected_cpu_dataset_count, 1000)
        self.assertEqual(plan.selected_gpu_dataset_count, 2000)
        self.assertEqual(plan.selected_backend_for(48), "cpu")
        self.assertEqual(plan.selected_backend_for(96), "gpu")
        self.assertEqual(plan.selected_backend_for(144), "gpu")


class V21FormalCheckpointScaffoldTests(unittest.TestCase):
    def test_signature_binds_formal_axes_and_backend_policy(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)

        signature = v21_formal_run_signature(contract, config, plan)
        changed = v21_formal_run_signature(
            contract,
            config,
            replace(plan, selected_gpu_dataset_count=1999),
        )

        self.assertEqual(len(signature), 64)
        self.assertNotEqual(signature, changed)
        self.assertEqual(
            tuple(V21FormalDatasetCheckpoint.__dataclass_fields__),
            (
                "run_signature",
                "dataset_key",
                "selected_backend",
                "records",
                "structure_selections",
            ),
        )

    def test_one_formal_dataset_round_trips_with_backend_validation(self):
        development = pu_dcgp_v21_development_contract()
        formal = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=formal.quantile_grid)
        plan = v21_formal_plan(formal, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        source = next(
            checkpoint
            for checkpoint in load_v21_dataset_checkpoints(
                data / "development_v21_records.jsonl",
                development,
                config,
            )
            if checkpoint.dataset_key[:2]
            == ("identified_balanced_particles", 48)
        )
        key = ("identified_balanced_particles", 48, 20)
        checkpoint = V21FormalDatasetCheckpoint(
            run_signature=v21_formal_run_signature(
                formal,
                config,
                plan,
            ),
            dataset_key=key,
            selected_backend="cpu",
            records=tuple(
                replace(
                    record,
                    replicate_index=20,
                )
                for record in source.records
            ),
            structure_selections=source.structure_selections,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "formal.jsonl"
            append_v21_formal_checkpoint(
                path,
                checkpoint,
                formal,
                config,
                plan,
            )
            loaded = load_v21_formal_checkpoints(
                path,
                formal,
                config,
                plan,
            )

        self.assertEqual(loaded, (checkpoint,))
        with self.assertRaisesRegex(ValueError, "backend"):
            validate_v21_formal_checkpoint(
                replace(checkpoint, selected_backend="gpu"),
                formal,
                config,
                plan,
            )


class V21FormalRunnerTests(unittest.TestCase):
    def test_dataset_runner_injects_the_frozen_selected_backend(self):
        development = pu_dcgp_v21_development_contract()
        formal = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=formal.quantile_grid)
        plan = v21_formal_plan(formal, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        sources = load_v21_dataset_checkpoints(
            data / "development_v21_records.jsonl",
            development,
            config,
        )
        expected_classes = {
            48: StructuredExactICMGaussianProcessRegressor,
            96: GPUAcceleratedStructuredExactICMGaussianProcessRegressor,
        }
        for sample_size, expected_class in expected_classes.items():
            source = next(
                checkpoint
                for checkpoint in sources
                if checkpoint.dataset_key[:2]
                == ("identified_balanced_particles", sample_size)
            )
            result = V21BenchmarkDatasetResult(
                records=tuple(
                    replace(record, replicate_index=20)
                    for record in source.records
                ),
                structure_selections=source.structure_selections,
            )
            with self.subTest(sample_size=sample_size), patch(
                "experiments.pu_dcgp_v21.formal_runner."
                "run_v21_benchmark_dataset",
                return_value=result,
            ) as mocked:
                checkpoint = run_v21_formal_dataset(
                    formal,
                    config,
                    plan,
                    "identified_balanced_particles",
                    sample_size,
                    20,
                )

            self.assertIs(
                mocked.call_args.kwargs["selected_model_class"],
                expected_class,
            )
            self.assertEqual(
                checkpoint.selected_backend,
                "cpu" if sample_size == 48 else "gpu",
            )

    def test_cli_is_dry_run_by_default(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "formal.jsonl"
            result = run_formal_main(
                [
                    "--checkpoint",
                    str(path),
                    "--replicate-start",
                    "20",
                    "--replicate-stop",
                    "21",
                    "--sample-size",
                    "96",
                    "--scenario",
                    "identified_balanced_particles",
                ]
            )

            self.assertEqual(result, 0)
            self.assertFalse(path.exists())

    def test_exact_key_runner_skips_completed_nonrectangular_keys(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        sources = load_v21_formal_checkpoints(
            data / "formal_v21_records.jsonl",
            contract,
            config,
            plan,
        )
        selected = (sources[0], sources[-1])
        keys = tuple(checkpoint.dataset_key for checkpoint in selected)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "exact.jsonl"
            append_v21_formal_checkpoint(
                path,
                selected[0],
                contract,
                config,
                plan,
            )
            with patch(
                "experiments.pu_dcgp_v21.formal_runner."
                "run_v21_formal_dataset",
                return_value=selected[1],
            ) as mocked:
                checkpoints = run_checkpointed_v21_formal_keys(
                    path,
                    contract,
                    config,
                    plan,
                    keys,
                )

        self.assertEqual(checkpoints, selected)
        mocked.assert_called_once_with(
            contract,
            config,
            plan,
            *selected[1].dataset_key,
        )

    def test_exact_key_selection_rejects_duplicates_and_outside_axes(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)
        valid = ("identified_balanced_particles", 48, 20)

        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_v21_formal_dataset_keys((valid, valid), plan)
        with self.assertRaisesRegex(ValueError, "replicate"):
            validate_v21_formal_dataset_keys(
                (("identified_balanced_particles", 48, 220),),
                plan,
            )


class V21FormalReportingTests(unittest.TestCase):
    def test_partial_formal_checkpoint_is_valid_but_not_evaluable(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        checkpoints = (
            load_v21_formal_checkpoints(
                data / "formal_v21_records.jsonl",
                contract,
                config,
                plan,
            )[0],
        )
        integrity = audit_v21_formal_checkpoints(
            checkpoints,
            plan,
        )

        self.assertEqual(integrity.dataset_count, 1)
        self.assertTrue(integrity.integrity_passed)
        self.assertFalse(integrity.phase_complete)
        self.assertEqual(
            integrity.missing_dataset_count,
            plan.dataset_count - integrity.dataset_count,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            write_v21_formal_summary(
                path,
                checkpoints,
                contract,
                config,
                plan,
                v21_formal_run_signature(contract, config, plan),
            )
            summary = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(summary["integrity"]["integrity_passed"])
        self.assertFalse(summary["integrity"]["phase_complete"])
        self.assertEqual(
            {
                decision["status"]
                for decision in summary["threshold_evaluations"]
            },
            {"not_evaluable"},
        )

    def test_formal_shards_merge_idempotently_and_reject_conflicts(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        checkpoints = load_v21_formal_checkpoints(
            data / "formal_v21_records.jsonl",
            contract,
            config,
            plan,
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            for checkpoint in checkpoints[:1]:
                append_v21_formal_checkpoint(
                    first,
                    checkpoint,
                    contract,
                    config,
                    plan,
                )
            for checkpoint in checkpoints[1:]:
                append_v21_formal_checkpoint(
                    second,
                    checkpoint,
                    contract,
                    config,
                    plan,
                )
            merged = merge_v21_formal_shards(
                (first, second),
                output,
                contract,
                config,
                plan,
            )
            repeated = merge_v21_formal_shards(
                (first, second),
                output,
                contract,
                config,
                plan,
            )

            conflict_path = root / "conflict.jsonl"
            first_checkpoint = checkpoints[0]
            conflicting_record = replace(
                first_checkpoint.records[0],
                runtime_seconds=(
                    first_checkpoint.records[0].runtime_seconds + 1.0
                ),
            )
            append_v21_formal_checkpoint(
                conflict_path,
                replace(
                    first_checkpoint,
                    records=(
                        conflicting_record,
                        *first_checkpoint.records[1:],
                    ),
                ),
                contract,
                config,
                plan,
            )
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                merge_v21_formal_shards(
                    (first, conflict_path),
                    root / "conflict_merged.jsonl",
                    contract,
                    config,
                    plan,
                )

        self.assertEqual(merged, checkpoints)
        self.assertEqual(repeated, checkpoints)


class V21FormalBatchPlanTests(unittest.TestCase):
    def test_batch_plan_excludes_completed_keys_and_round_trips(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        formal_plan = v21_formal_plan(contract, config)
        signature = v21_formal_run_signature(
            contract,
            config,
            formal_plan,
        )
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        completed = {
            checkpoint.dataset_key
            for checkpoint in load_v21_formal_checkpoints(
                data / "formal_v21_records.jsonl",
                contract,
                config,
                formal_plan,
            )
        }
        scenarios = formal_plan.scenario_ids[:2]
        sample_sizes = (96,)
        replicates = tuple(range(20, 25))
        target_keys = {
            (scenario, 96, replicate)
            for replicate in replicates
            for scenario in scenarios
        }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch_plan = build_v21_formal_batch_plan(
                formal_plan,
                signature,
                completed,
                root / "shards",
                scenario_ids=scenarios,
                sample_sizes=sample_sizes,
                replicate_indices=replicates,
                replicates_per_shard=2,
            )
            manifest = root / "batch.json"
            write_v21_formal_batch_manifest(manifest, batch_plan)
            loaded = load_v21_formal_batch_manifest(manifest)

        scheduled = {
            key for shard in batch_plan.shards for key in shard.dataset_keys
        }
        self.assertEqual(loaded, batch_plan)
        self.assertEqual(scheduled, target_keys - completed)
        self.assertFalse(scheduled & completed)
        self.assertEqual(
            batch_plan.completed_dataset_count,
            len(target_keys & completed),
        )
        self.assertEqual(
            batch_plan.scheduled_dataset_count,
            len(target_keys - completed),
        )
        self.assertTrue(
            all(shard.selected_backend == "gpu" for shard in batch_plan.shards)
        )

    def test_batch_plan_rejects_invalid_concurrency_and_axes(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        formal_plan = v21_formal_plan(contract, config)
        signature = v21_formal_run_signature(
            contract,
            config,
            formal_plan,
        )

        with self.assertRaisesRegex(ValueError, "max_workers"):
            build_v21_formal_batch_plan(
                formal_plan,
                signature,
                set(),
                Path("shards"),
                max_workers=0,
            )
        with self.assertRaisesRegex(ValueError, "sample size"):
            build_v21_formal_batch_plan(
                formal_plan,
                signature,
                set(),
                Path("shards"),
                sample_sizes=(192,),
            )

    def test_batch_and_shard_clis_are_dry_run_by_default(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "formal.jsonl"
            shard_directory = root / "shards"
            manifest = root / "manifest.json"
            result = run_formal_batch_main(
                [
                    "--output",
                    str(output),
                    "--shard-directory",
                    str(shard_directory),
                    "--manifest",
                    str(manifest),
                    "--replicate-start",
                    "20",
                    "--replicate-stop",
                    "22",
                    "--sample-size",
                    "96",
                    "--scenario",
                    "identified_balanced_particles",
                    "--replicates-per-shard",
                    "1",
                ]
            )
            batch_plan = load_v21_formal_batch_manifest(manifest)
            shard_result = run_formal_batch_shard_main(
                [
                    "--manifest",
                    str(manifest),
                    "--shard-id",
                    batch_plan.shards[0].shard_id,
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(shard_result, 0)
        self.assertEqual(batch_plan.scheduled_dataset_count, 2)
        self.assertEqual(len(batch_plan.shards), 2)
        self.assertFalse(output.exists())
        self.assertFalse(batch_plan.shards[0].checkpoint_path.exists())

    def test_thread_limited_environment_caps_each_numeric_pool(self):
        environment = thread_limited_environment(1)

        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            self.assertEqual(environment[variable], "1")

    def test_live_progress_distinguishes_staged_from_merged_data(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        formal_plan = v21_formal_plan(contract, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        source = load_v21_formal_checkpoints(
            data / "formal_v21_records.jsonl",
            contract,
            config,
            formal_plan,
        )[0]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch_plan = build_v21_formal_batch_plan(
                formal_plan,
                v21_formal_run_signature(
                    contract,
                    config,
                    formal_plan,
                ),
                set(),
                root / "shards",
                scenario_ids=(source.dataset_key[0],),
                sample_sizes=(source.dataset_key[1],),
                replicate_indices=(source.dataset_key[2],),
                replicates_per_shard=1,
            )
            manifest = root / "manifest.json"
            output = root / "main.jsonl"
            write_v21_formal_batch_manifest(manifest, batch_plan)
            append_v21_formal_checkpoint(
                batch_plan.shards[0].checkpoint_path,
                source,
                contract,
                config,
                formal_plan,
            )
            staged = audit_v21_formal_batch_progress(
                manifest,
                output,
            )
            merge_v21_formal_shards(
                (batch_plan.shards[0].checkpoint_path,),
                output,
                contract,
                config,
                formal_plan,
            )
            merged = audit_v21_formal_batch_progress(
                manifest,
                output,
            )

        self.assertEqual(staged.total_completed_dataset_count, 1)
        self.assertEqual(staged.staged_unmerged_dataset_count, 1)
        self.assertEqual(staged.merged_dataset_count, 0)
        self.assertEqual(merged.staged_unmerged_dataset_count, 0)
        self.assertEqual(merged.merged_dataset_count, 1)
        self.assertTrue(merged.integrity_passed)


class V21FormalReleaseTests(unittest.TestCase):
    def test_partial_summary_is_not_released(self):
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            partial_summary = root / "partial_summary.json"
            payload = json.loads(
                (data / "formal_v21_records.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["integrity"]["phase_complete"] = False
            partial_summary.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            output = root / "release"

            with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
                load_v21_formal_release(partial_summary)
            with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
                load_complete_v21_structure_frequencies(
                    partial_summary,
                    data / "formal_v21_records.jsonl",
                )
            with self.assertRaisesRegex(
                ValueError,
                "incomplete or invalid",
            ):
                finalize_v21_formal_evidence(
                    partial_summary,
                    data / "formal_v21_records.jsonl",
                    output,
                )
            self.assertFalse(output.exists())

    def test_final_release_hashes_match_disk_bytes(self):
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"

        with TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            manifest_path = finalize_v21_formal_evidence(
                data / "formal_v21_records.summary.json",
                data / "formal_v21_records.jsonl",
                output,
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            for name, expected_digest in manifest[
                "artifacts_sha256"
            ].items():
                self.assertEqual(
                    sha256((output / name).read_bytes()).hexdigest(),
                    expected_digest,
                    name,
                )

    def test_structure_frequency_aggregation_is_descriptive(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)
        data = Path(__file__).resolve().parents[2] / "pu_dcgp_v21" / "data"
        source = load_v21_formal_checkpoints(
            data / "formal_v21_records.jsonl",
            contract,
            config,
            plan,
        )[0]

        cells = aggregate_v21_structure_frequencies((source,))

        self.assertEqual(len(cells), 6)
        self.assertEqual(
            sum(cell.selection_count for cell in cells),
            18,
        )
        self.assertEqual(
            {cell.scope for cell in cells},
            {"full_data", "calibration_fold"},
        )
        self.assertTrue(
            all(0.0 <= cell.full_selection_rate <= 1.0 for cell in cells)
        )

    def test_failed_formal_decision_remains_an_explicit_limitation(self):
        release = V21FormalRelease(
            schema="pu_dcgp_v21_formal_release_v1",
            run_signature="test",
            dataset_count=3000,
            method_record_count=24000,
            structure_selection_count=54000,
            aggregate_cell_count=120,
            decisions=(
                V21FormalReleasedDecision(
                    hypothesis_id="H4-v2",
                    status="fail",
                    threshold_met=False,
                    evidence={
                        "supported_active_existence_power": 0.79,
                        "supported_null_false_admission": 0.01,
                        "supported_active_whole_curve_direction_power": 0.60,
                    },
                    reason="Frozen criterion evaluated.",
                ),
            ),
        )

        claim = build_v21_formal_claim_map(release)[0]

        self.assertEqual(claim.role, "explicit limitation")
        self.assertIn("does not meet", claim.statement)

    def test_complete_axis_and_boolean_decisions_release(self):
        contract = pu_dcgp_v21_formal_contract()
        config = PUDCGPV21Config(quantile_grid=contract.quantile_grid)
        plan = v21_formal_plan(contract, config)
        signature = v21_formal_run_signature(contract, config, plan)
        aggregates = [
            {
                "scenario_id": scenario_id,
                "sample_size": sample_size,
                "method_name": method_name,
                "replicate_count": len(plan.replicate_indices),
                "median_shape_normalized_irmse": 0.12,
                "simultaneous_coverage_rate": 0.95,
                "target_unsupported_admission_rate": 0.50,
                "active_admission_rate": 0.80,
                "null_false_admission_rate": 0.01,
                "active_whole_curve_direction_rate": 0.60,
            }
            for scenario_id in plan.scenario_ids
            for sample_size in plan.sample_sizes
            for method_name in contract.methods
        ]
        for aggregate in aggregates:
            method_name = aggregate["method_name"]
            scenario_id = aggregate["scenario_id"]
            if method_name == "mean_gp":
                aggregate["median_shape_normalized_irmse"] = 0.20
            elif method_name == "pu_dcgp_diagonal_v1":
                aggregate["median_shape_normalized_irmse"] = 0.10
            elif method_name == "joint_pu_dcgp_group_calibrated":
                aggregate["median_shape_normalized_irmse"] = 0.09
            elif (
                method_name
                == "joint_pu_dcgp_bic_selected_calibrated"
            ):
                aggregate["median_shape_normalized_irmse"] = 0.08
                if scenario_id == "identified_balanced_particles":
                    aggregate["simultaneous_coverage_rate"] = 0.95
                elif (
                    scenario_id
                    == "identified_heterogeneous_particles"
                ):
                    aggregate["simultaneous_coverage_rate"] = 0.94
                else:
                    aggregate[
                        "target_unsupported_admission_rate"
                    ] = 0.60
            elif method_name == "distribution_gp_no_pu":
                if scenario_id == "identified_balanced_particles":
                    aggregate["simultaneous_coverage_rate"] = 0.93
                elif (
                    scenario_id
                    == "identified_heterogeneous_particles"
                ):
                    aggregate["simultaneous_coverage_rate"] = 0.85
            elif (
                method_name
                == "support_gated_joint_pu_dcgp_bic_selected"
            ):
                if scenario_id in (
                    "sequence_aligned_drift",
                    "module_sign_reversal",
                    "insufficient_overlap",
                ):
                    aggregate[
                        "target_unsupported_admission_rate"
                    ] = 0.0
                elif aggregate["sample_size"] == 144:
                    aggregate["active_admission_rate"] = 0.85
                    aggregate["null_false_admission_rate"] = 0.01
                    aggregate[
                        "active_whole_curve_direction_rate"
                    ] = 0.70
        decision_evidence = {
            "S1": {
                "maximum_calibrated_to_diagonal_shape_irmse_ratio": 0.80,
                "minimum_shape_irmse_reduction_vs_mean_gp": 0.60,
            },
            "H2-v2": {
                "minimum_heterogeneous_coverage_error_reduction": 0.09,
                "maximum_balanced_coverage_error_worsening": -0.02,
            },
            "H3-v2": {
                "maximum_gated_unsupported_admission": 0.0,
                "minimum_relative_reduction_from_ungated": 1.0,
            },
            "H4-v2": {
                "supported_active_existence_power": 0.85,
                "supported_null_false_admission": 0.01,
                "supported_active_whole_curve_direction_power": 0.70,
            },
        }
        decisions = [
            {
                "hypothesis_id": hypothesis_id,
                "status": "pass",
                "threshold_met": True,
                "evidence": decision_evidence[hypothesis_id],
                "reason": "Frozen criterion evaluated.",
            }
            for hypothesis_id in ("S1", "H2-v2", "H3-v2", "H4-v2")
        ]
        payload = {
            "schema": "pu_dcgp_v21_formal_summary_v1",
            "phase": "formal",
            "run_signature": signature,
            "integrity": {
                "dataset_count": 3000,
                "method_record_count": 24000,
                "structure_selection_count": 54000,
                "expected_dataset_count": 3000,
                "expected_method_record_count": 24000,
                "expected_structure_selection_count": 54000,
                "missing_dataset_count": 0,
                "unexpected_dataset_count": 0,
                "duplicate_dataset_count": 0,
                "calibrated_gated_mismatch_count": 0,
                "selected_cpu_dataset_count": 1000,
                "selected_gpu_dataset_count": 2000,
                "integrity_passed": True,
                "phase_complete": True,
            },
            "aggregates": aggregates,
            "threshold_evaluations": decisions,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            release = load_v21_formal_release(path)
            postanalysis = build_v21_formal_postanalysis(path)
            tables = build_v21_formal_paper_tables(postanalysis)
            rendered = render_v21_formal_paper_tables(postanalysis)
            claims = build_v21_formal_claim_map(release)
            rendered_claims = render_v21_formal_claim_map(release)
            source = load_v21_formal_checkpoints(
                Path(__file__).resolve().parents[2]
                / "pu_dcgp_v21"
                / "data"
                / "formal_v21_records.jsonl",
                contract,
                config,
                plan,
            )[0]
            structure_cells = aggregate_v21_structure_frequencies(
                (source,)
            )
            artifact_payloads = build_v21_final_artifact_payloads(
                release,
                postanalysis,
                structure_cells,
            )

        self.assertEqual(release.dataset_count, 3000)
        self.assertEqual(release.aggregate_cell_count, 120)
        self.assertEqual(
            tuple(decision.status for decision in release.decisions),
            ("pass", "pass", "pass", "pass"),
        )
        self.assertAlmostEqual(
            postanalysis.shape_recovery[0].selected_to_diagonal_ratio,
            0.80,
        )
        self.assertAlmostEqual(
            postanalysis.shape_recovery[0].full_to_diagonal_ratio,
            0.90,
        )
        self.assertAlmostEqual(
            postanalysis.shape_recovery[0].selected_reduction_vs_mean_gp,
            0.60,
        )
        self.assertAlmostEqual(
            postanalysis.coverage_calibration[0]
            .heterogeneous_coverage_error_reduction,
            0.09,
        )
        self.assertAlmostEqual(
            postanalysis.unsupported_admission[0].relative_reduction,
            1.0,
        )
        self.assertAlmostEqual(
            postanalysis.retained_power.active_existence_power,
            0.85,
        )
        self.assertEqual(len(tables.sections), 4)
        self.assertIn("Selected/diagonal", rendered)
        self.assertIn("H2-v2: simultaneous coverage", rendered)
        self.assertNotIn("Held-out prediction", rendered)
        self.assertEqual(
            tuple(claim.role for claim in claims),
            (
                "supported claim",
                "supported claim",
                "supported claim",
                "supported claim",
            ),
        )
        self.assertIn("cannot compensate", rendered_claims)
        self.assertEqual(
            set(artifact_payloads),
            {
                "formal_v21_claim_map.md",
                "formal_v21_paper_tables.md",
                "formal_v21_structure_frequencies.md",
                "formal_v21_structure_frequencies.json",
            },
        )
