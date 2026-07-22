"""Contracts for the PU-DCGP v2.2 selective-guarantee scaffold."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pu_dcgp.benchmark_generator import (
    SyntheticQuantileEffectTruth,
)
from experiments.pu_dcgp.benchmark_methods import (
    BenchmarkEffectEstimate,
    BenchmarkMethodResult,
)
from experiments.pu_dcgp_v22 import (
    CalibrationCurveBlock,
    CurveBundle,
    CurveBundlePrediction,
    ExchangeableScoreOrbitAudit,
    ExchangeabilityContract,
    FamilywiseCurveCertificate,
    SelectiveCertificateStatus,
    SelectiveTargetKind,
    SplitConformalFamilywiseCalibrator,
    PUDCGPV22SelectiveWorkflow,
    V22SelectiveBenchmarkPlan,
    apply_structural_abstention,
    audit_a_group_selective_feasibility,
    audit_exchangeable_score_orbit,
    calibration_curve_block_from_benchmark,
    curve_bundle_prediction_from_benchmark,
    audit_selective_preconditions,
    minimum_calibration_blocks,
    map_familywise_band_to_claims,
    split_conformal_rank,
    whole_family_nonconformity_score,
    render_a_group_selective_feasibility,
    v22_selective_benchmark_contract,
)
from experiments.pu_dcgp_v22.selective_benchmark_batch import (
    curve_block_from_payload,
    curve_block_to_payload,
    finalize_selective_benchmark,
    run_or_resume_selective_benchmark,
    selective_benchmark_plan_payload,
    selective_benchmark_plan_signature,
    summarize_selective_benchmark,
    validate_selective_benchmark_release,
)


class V22SelectiveContractTests(unittest.TestCase):
    def setUp(self):
        self.claim_ids = ("current_temperature", "argon_velocity")
        self.quantile_grid = tuple(np.linspace(0.05, 0.95, 19))
        self.shape = (len(self.claim_ids), len(self.quantile_grid))

    def test_exchangeability_contract_records_finite_sample_unit(self):
        contract = ExchangeabilityContract(
            calibration_unit="independent_experimental_block",
            target_kind=SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE,
            claim_ids=self.claim_ids,
            quantile_grid=self.quantile_grid,
            alpha=0.05,
            calibration_block_count=19,
            proper_training_disjoint=True,
            prediction_rule_frozen_before_calibration=True,
            calibration_targets_hidden_from_fit=True,
            complete_target_bundle=True,
            exchangeability_justified=True,
            randomized_or_identified=False,
        )

        self.assertEqual(contract.calibration_block_count, 19)
        self.assertEqual(contract.alpha, 0.05)
        self.assertEqual(
            SelectiveCertificateStatus.PRECONDITION_FAILED.value,
            "precondition_failed",
        )

    def test_curve_block_requires_matching_complete_axes(self):
        target = CurveBundle(
            self.claim_ids,
            self.quantile_grid,
            np.zeros(self.shape),
        )
        prediction = CurveBundlePrediction(
            self.claim_ids,
            self.quantile_grid,
            np.zeros(self.shape),
            np.ones(self.shape),
        )

        block = CalibrationCurveBlock("block-001", target, prediction)

        self.assertEqual(block.target.values.shape, self.shape)
        self.assertTrue(np.all(block.prediction.scale > 0.0))

    def test_prediction_rejects_nonpositive_scale(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            CurveBundlePrediction(
                self.claim_ids,
                self.quantile_grid,
                np.zeros(self.shape),
                np.zeros(self.shape),
            )

    def test_contract_rejects_duplicate_claims(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ExchangeabilityContract(
                calibration_unit="block",
                target_kind=SelectiveTargetKind.OBSERVABLE_CURVE_BUNDLE,
                claim_ids=("duplicate", "duplicate"),
                quantile_grid=self.quantile_grid,
                alpha=0.05,
                calibration_block_count=19,
                proper_training_disjoint=True,
                prediction_rule_frozen_before_calibration=True,
                calibration_targets_hidden_from_fit=True,
                complete_target_bundle=True,
                exchangeability_justified=True,
                randomized_or_identified=False,
            )

    def test_split_conformal_rank_has_exact_five_percent_boundary(self):
        self.assertEqual(split_conformal_rank(18, 0.05), 19)
        self.assertEqual(split_conformal_rank(19, 0.05), 19)
        self.assertEqual(minimum_calibration_blocks(0.05), 19)

    def test_valid_known_truth_contract_passes_precondition_audit(self):
        contract = ExchangeabilityContract(
            calibration_unit="independent_benchmark_dataset",
            target_kind=SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE,
            claim_ids=self.claim_ids,
            quantile_grid=self.quantile_grid,
            alpha=0.05,
            calibration_block_count=19,
            proper_training_disjoint=True,
            prediction_rule_frozen_before_calibration=True,
            calibration_targets_hidden_from_fit=True,
            complete_target_bundle=True,
            exchangeability_justified=True,
            randomized_or_identified=False,
        )

        audit = audit_selective_preconditions(contract)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.calibration_rank, 19)
        self.assertEqual(audit.failures, ())

    def test_a_group_like_contract_fails_explicitly(self):
        contract = ExchangeabilityContract(
            calibration_unit="matched_stratum",
            target_kind=(
                SelectiveTargetKind.RANDOMIZED_EFFECT_CURVE_BUNDLE
            ),
            claim_ids=self.claim_ids,
            quantile_grid=self.quantile_grid,
            alpha=0.05,
            calibration_block_count=9,
            proper_training_disjoint=False,
            prediction_rule_frozen_before_calibration=True,
            calibration_targets_hidden_from_fit=True,
            complete_target_bundle=False,
            exchangeability_justified=False,
            randomized_or_identified=False,
        )

        audit = audit_selective_preconditions(contract)

        self.assertFalse(audit.passed)
        self.assertEqual(
            audit.failures,
            (
                "proper_training_not_disjoint",
                "incomplete_target_bundle",
                "exchangeability_not_justified",
                "causal_identification_missing",
                "insufficient_calibration_blocks",
            ),
        )

    def test_calibration_target_leakage_forces_precondition_failure(self):
        contract = ExchangeabilityContract(
            calibration_unit="independent_benchmark_dataset",
            target_kind=SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE,
            claim_ids=self.claim_ids,
            quantile_grid=self.quantile_grid,
            alpha=0.05,
            calibration_block_count=19,
            proper_training_disjoint=True,
            prediction_rule_frozen_before_calibration=True,
            calibration_targets_hidden_from_fit=False,
            complete_target_bundle=True,
            exchangeability_justified=True,
            randomized_or_identified=False,
        )

        audit = audit_selective_preconditions(contract)

        self.assertFalse(audit.passed)
        self.assertIn("calibration_target_leakage", audit.failures)


class V22SplitConformalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.claim_ids = ("claim_a", "claim_b")
        self.quantile_grid = (0.25, 0.75)
        self.shape = (2, 2)

    def contract(self, block_count=3, alpha=0.25):
        return ExchangeabilityContract(
            calibration_unit="independent_block",
            target_kind=SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE,
            claim_ids=self.claim_ids,
            quantile_grid=self.quantile_grid,
            alpha=alpha,
            calibration_block_count=block_count,
            proper_training_disjoint=True,
            prediction_rule_frozen_before_calibration=True,
            calibration_targets_hidden_from_fit=True,
            complete_target_bundle=True,
            exchangeability_justified=True,
            randomized_or_identified=False,
        )

    def block(self, block_id, maximum_error):
        centre = np.zeros(self.shape)
        target = np.zeros(self.shape)
        target[1, 1] = maximum_error
        return CalibrationCurveBlock(
            block_id,
            CurveBundle(
                self.claim_ids,
                self.quantile_grid,
                target,
            ),
            CurveBundlePrediction(
                self.claim_ids,
                self.quantile_grid,
                centre,
                np.ones(self.shape),
            ),
        )

    def test_score_is_maximum_over_the_complete_family(self):
        block = self.block("block-1", 3.5)

        self.assertEqual(whole_family_nonconformity_score(block), 3.5)

    def test_hand_calculated_rank_radius_and_band(self):
        blocks = tuple(
            self.block(f"block-{index}", error)
            for index, error in enumerate((1.0, 3.0, 2.0))
        )
        future = CurveBundlePrediction(
            self.claim_ids,
            self.quantile_grid,
            np.asarray([[10.0, 20.0], [30.0, 40.0]]),
            np.asarray([[2.0, 1.0], [0.5, 4.0]]),
        )

        certificate = SplitConformalFamilywiseCalibrator().certify(
            self.contract(),
            blocks,
            future,
        )

        self.assertIs(
            certificate.status,
            SelectiveCertificateStatus.CERTIFIED,
        )
        self.assertEqual(certificate.calibration_rank, 3)
        self.assertEqual(certificate.radius, 3.0)
        np.testing.assert_allclose(
            certificate.lower,
            [[4.0, 17.0], [28.5, 28.0]],
        )
        np.testing.assert_allclose(
            certificate.upper,
            [[16.0, 23.0], [31.5, 52.0]],
        )

    def test_insufficient_blocks_return_no_band(self):
        blocks = (
            self.block("block-1", 1.0),
            self.block("block-2", 2.0),
        )
        future = blocks[0].prediction

        certificate = SplitConformalFamilywiseCalibrator().certify(
            self.contract(block_count=2),
            blocks,
            future,
        )

        self.assertIs(
            certificate.status,
            SelectiveCertificateStatus.PRECONDITION_FAILED,
        )
        self.assertIn(
            "insufficient_calibration_blocks",
            certificate.reasons,
        )
        self.assertIsNone(certificate.lower)
        self.assertTrue(np.isinf(certificate.radius))

    def test_declared_and_observed_block_counts_must_match(self):
        blocks = tuple(
            self.block(f"block-{index}", float(index))
            for index in range(3)
        )

        certificate = SplitConformalFamilywiseCalibrator().certify(
            self.contract(block_count=4),
            blocks,
            blocks[0].prediction,
        )

        self.assertIn(
            "calibration_block_count_mismatch",
            certificate.reasons,
        )


class V22ClaimMappingTests(unittest.TestCase):
    def test_joint_band_maps_existence_and_direction_separately(self):
        certificate = FamilywiseCurveCertificate(
            status=SelectiveCertificateStatus.CERTIFIED,
            alpha=0.05,
            calibration_rank=19,
            calibration_block_count=19,
            radius=2.0,
            claim_ids=(
                "uniform_positive",
                "local_positive",
                "uniform_negative",
                "unresolved",
            ),
            quantile_grid=(0.25, 0.75),
            lower=np.asarray(
                [
                    [1.0, 2.0],
                    [-1.0, 0.2],
                    [-3.0, -2.0],
                    [-1.0, -0.5],
                ]
            ),
            upper=np.asarray(
                [
                    [3.0, 4.0],
                    [1.0, 0.8],
                    [-1.0, -0.5],
                    [0.5, 1.0],
                ]
            ),
            reasons=(),
        )

        decisions = map_familywise_band_to_claims(
            certificate,
            mean_weights=(0.5, 0.5),
        )

        self.assertEqual(
            [
                (
                    decision.status,
                    decision.existence_reported,
                    decision.mean_direction,
                    decision.whole_curve_direction,
                )
                for decision in decisions
            ],
            [
                (SelectiveCertificateStatus.CERTIFIED, True, 1, 1),
                (SelectiveCertificateStatus.CERTIFIED, True, 0, 0),
                (SelectiveCertificateStatus.CERTIFIED, True, -1, -1),
                (SelectiveCertificateStatus.ABSTAIN, False, 0, 0),
            ],
        )

    def test_precondition_failure_cannot_issue_a_claim(self):
        certificate = FamilywiseCurveCertificate(
            status=SelectiveCertificateStatus.PRECONDITION_FAILED,
            alpha=0.05,
            calibration_rank=19,
            calibration_block_count=9,
            radius=float("inf"),
            claim_ids=("claim_a", "claim_b"),
            quantile_grid=(0.25, 0.75),
            lower=None,
            upper=None,
            reasons=("exchangeability_not_justified",),
        )

        decisions = map_familywise_band_to_claims(
            certificate,
            mean_weights=(0.5, 0.5),
        )

        self.assertTrue(
            all(
                decision.status
                is SelectiveCertificateStatus.PRECONDITION_FAILED
                and not decision.existence_reported
                for decision in decisions
            )
        )

    def test_mean_weights_are_fixed_convex_weights(self):
        certificate = FamilywiseCurveCertificate(
            status=SelectiveCertificateStatus.CERTIFIED,
            alpha=0.05,
            calibration_rank=19,
            calibration_block_count=19,
            radius=1.0,
            claim_ids=("claim",),
            quantile_grid=(0.25, 0.75),
            lower=np.zeros((1, 2)),
            upper=np.ones((1, 2)),
            reasons=(),
        )

        with self.assertRaisesRegex(ValueError, "sum to one"):
            map_familywise_band_to_claims(
                certificate,
                mean_weights=(1.0, 1.0),
            )


class V22SelectiveWorkflowTests(unittest.TestCase):
    def test_structural_layer_can_only_downgrade_a_certified_claim(self):
        decisions = (
            map_familywise_band_to_claims(
                FamilywiseCurveCertificate(
                    status=SelectiveCertificateStatus.CERTIFIED,
                    alpha=0.25,
                    calibration_rank=3,
                    calibration_block_count=3,
                    radius=1.0,
                    claim_ids=("eligible", "vetoed", "unresolved"),
                    quantile_grid=(0.25, 0.75),
                    lower=np.asarray(
                        [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]]
                    ),
                    upper=np.asarray(
                        [[3.0, 3.0], [4.0, 4.0], [1.0, 1.0]]
                    ),
                    reasons=(),
                ),
                mean_weights=(0.5, 0.5),
            )
        )

        updated = apply_structural_abstention(
            decisions,
            {"vetoed": ("insufficient_overlap",)},
        )

        self.assertIs(
            updated[0].status,
            SelectiveCertificateStatus.CERTIFIED,
        )
        self.assertIs(
            updated[1].status,
            SelectiveCertificateStatus.ABSTAIN,
        )
        self.assertEqual(
            updated[1].reasons,
            ("insufficient_overlap",),
        )
        self.assertIs(
            updated[2].status,
            SelectiveCertificateStatus.ABSTAIN,
        )

    def test_workflow_connects_calibration_mapping_and_veto(self):
        claim_ids = ("claim_a",)
        grid = (0.25, 0.75)
        contract = ExchangeabilityContract(
            calibration_unit="block",
            target_kind=SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE,
            claim_ids=claim_ids,
            quantile_grid=grid,
            alpha=0.25,
            calibration_block_count=3,
            proper_training_disjoint=True,
            prediction_rule_frozen_before_calibration=True,
            calibration_targets_hidden_from_fit=True,
            complete_target_bundle=True,
            exchangeability_justified=True,
            randomized_or_identified=False,
        )
        calibration_blocks = tuple(
            CalibrationCurveBlock(
                f"block-{index}",
                CurveBundle(
                    claim_ids,
                    grid,
                    np.full((1, 2), error),
                ),
                CurveBundlePrediction(
                    claim_ids,
                    grid,
                    np.zeros((1, 2)),
                    np.ones((1, 2)),
                ),
            )
            for index, error in enumerate((0.1, 0.2, 0.3))
        )
        future = CurveBundlePrediction(
            claim_ids,
            grid,
            np.full((1, 2), 2.0),
            np.ones((1, 2)),
        )

        result = PUDCGPV22SelectiveWorkflow().certify(
            contract,
            calibration_blocks,
            future,
            mean_weights=(0.5, 0.5),
            ineligible_claim_reasons={
                "claim_a": ("sequence_stability_failed",)
            },
        )

        self.assertIs(
            result.certificate.status,
            SelectiveCertificateStatus.CERTIFIED,
        )
        self.assertIs(
            result.decisions[0].status,
            SelectiveCertificateStatus.ABSTAIN,
        )


class V22FiniteSampleValidationTests(unittest.TestCase):
    def test_distinct_twenty_score_orbit_has_exact_95_percent_coverage(self):
        audit = audit_exchangeable_score_orbit(
            np.arange(20, dtype=float),
            alpha=0.05,
        )

        self.assertIsInstance(audit, ExchangeableScoreOrbitAudit)
        self.assertEqual(audit.calibration_rank, 19)
        self.assertEqual(audit.miscovered_count, 1)
        self.assertEqual(audit.coverage_rate, 0.95)
        self.assertTrue(audit.passed)

    def test_tied_scores_are_conservative(self):
        audit = audit_exchangeable_score_orbit(
            np.ones(20),
            alpha=0.05,
        )

        self.assertEqual(audit.coverage_rate, 1.0)
        self.assertTrue(audit.passed)

    def test_heavy_tailed_heteroskedastic_curve_orbit_is_covered(self):
        rng = np.random.default_rng(52026)
        claim_ids = ("claim_a", "claim_b")
        grid = (0.2, 0.5, 0.8)
        blocks = []
        for index in range(20):
            centre = rng.normal(size=(2, 3))
            scale = np.exp(rng.normal(scale=0.6, size=(2, 3)))
            target = centre + scale * rng.standard_t(df=2.0, size=(2, 3))
            blocks.append(
                CalibrationCurveBlock(
                    f"heavy-tail-{index:02d}",
                    CurveBundle(claim_ids, grid, target),
                    CurveBundlePrediction(
                        claim_ids,
                        grid,
                        centre,
                        scale,
                    ),
                )
            )
        scores = np.asarray(
            [whole_family_nonconformity_score(block) for block in blocks]
        )
        orbit = audit_exchangeable_score_orbit(scores, alpha=0.05)
        covered = 0
        for future_index, future in enumerate(blocks):
            contract = ExchangeabilityContract(
                calibration_unit="independent_curve_block",
                target_kind=SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE,
                claim_ids=claim_ids,
                quantile_grid=grid,
                alpha=0.05,
                calibration_block_count=19,
                proper_training_disjoint=True,
                prediction_rule_frozen_before_calibration=True,
                calibration_targets_hidden_from_fit=True,
                complete_target_bundle=True,
                exchangeability_justified=True,
                randomized_or_identified=False,
            )
            certificate = SplitConformalFamilywiseCalibrator().certify(
                contract,
                tuple(
                    block
                    for index, block in enumerate(blocks)
                    if index != future_index
                ),
                future.prediction,
            )
            covered += int(
                np.all(future.target.values >= certificate.lower)
                and np.all(future.target.values <= certificate.upper)
            )

        self.assertEqual(covered, orbit.covered_count)
        self.assertGreaterEqual(covered / 20, 0.95)

    def test_false_selected_direction_is_bounded_by_miscoverage(self):
        claim_ids = ("selected_claim",)
        grid = (0.25, 0.75)
        centre = np.full((1, 2), 10.0)
        scale = np.ones((1, 2))
        target_levels = np.asarray(
            [10.0 + value for value in np.linspace(-5.0, 5.0, 19)]
            + [-10.0]
        )
        false_reports = 0

        for future_index, future_level in enumerate(target_levels):
            calibration_blocks = tuple(
                CalibrationCurveBlock(
                    f"block-{index}",
                    CurveBundle(
                        claim_ids,
                        grid,
                        np.full((1, 2), target_level),
                    ),
                    CurveBundlePrediction(
                        claim_ids,
                        grid,
                        centre,
                        scale,
                    ),
                )
                for index, target_level in enumerate(target_levels)
                if index != future_index
            )
            contract = ExchangeabilityContract(
                calibration_unit="exchangeable_block",
                target_kind=(
                    SelectiveTargetKind.KNOWN_TRUTH_CURVE_BUNDLE
                ),
                claim_ids=claim_ids,
                quantile_grid=grid,
                alpha=0.05,
                calibration_block_count=19,
                proper_training_disjoint=True,
                prediction_rule_frozen_before_calibration=True,
                calibration_targets_hidden_from_fit=True,
                complete_target_bundle=True,
                exchangeability_justified=True,
                randomized_or_identified=False,
            )
            result = PUDCGPV22SelectiveWorkflow().certify(
                contract,
                calibration_blocks,
                CurveBundlePrediction(
                    claim_ids,
                    grid,
                    centre,
                    scale,
                ),
                mean_weights=(0.5, 0.5),
            )
            decision = result.decisions[0]
            false_reports += int(
                decision.whole_curve_direction == 1
                and future_level < 0.0
            )

        self.assertEqual(false_reports, 1)
        self.assertLessEqual(false_reports / target_levels.size, 0.05)


class V22AGroupFeasibilityTests(unittest.TestCase):
    def test_current_a_group_receives_explicit_precondition_failure(self):
        result = audit_a_group_selective_feasibility()

        self.assertEqual(result.run_count, 150)
        self.assertEqual(result.unique_setting_count, 66)
        self.assertEqual(result.claim_count, 12)
        self.assertEqual(result.matched_strata_per_claim, 9)
        self.assertFalse(result.audit.passed)
        self.assertEqual(
            result.audit.failures,
            (
                "proper_training_not_disjoint",
                "incomplete_target_bundle",
                "exchangeability_not_justified",
                "causal_identification_missing",
                "insufficient_calibration_blocks",
            ),
        )

    def test_a_group_report_preserves_the_v21_conditional_result(self):
        rendered = render_a_group_selective_feasibility(
            audit_a_group_selective_feasibility()
        )

        self.assertIn("`precondition_failed`", rendered)
        self.assertIn("not relabelled", rendered)
        self.assertIn("conditional empirical result", rendered)


class V22BenchmarkAdapterTests(unittest.TestCase):
    def effect(self, estimand_id, outcome, point, variance):
        grid = np.asarray([0.25, 0.75])
        return BenchmarkEffectEstimate(
            estimand_id=estimand_id,
            treatment_name="treatment",
            outcome=outcome,
            quantile_grid=grid,
            point_effect=np.asarray(point, dtype=float),
            marginal_variance=np.asarray(variance, dtype=float),
            effect_covariance=np.diag(variance),
            lower_bound=None,
            upper_bound=None,
            interval_kind="point",
            admission_status="no_claim",
            reported=False,
            failed_gates=(),
        )

    def method_result(self):
        return BenchmarkMethodResult(
            method_name="selected",
            scenario_id="known_truth",
            sample_size=48,
            replicate_index=0,
            effects=(
                self.effect("current", "temperature", [1.0, 2.0], [4.0, 9.0]),
                self.effect("argon", "velocity", [3.0, 4.0], [1.0, 16.0]),
            ),
            preparation_seconds=0.0,
            fit_seconds=0.0,
            prediction_seconds=0.0,
        )

    def test_method_effects_stack_into_curve_prediction(self):
        prediction = curve_bundle_prediction_from_benchmark(
            self.method_result()
        )

        self.assertEqual(
            prediction.claim_ids,
            ("current:temperature", "argon:velocity"),
        )
        np.testing.assert_allclose(
            prediction.centre,
            [[1.0, 2.0], [3.0, 4.0]],
        )
        np.testing.assert_allclose(
            prediction.scale,
            [[2.0, 3.0], [1.0, 4.0]],
        )

    def test_known_truth_joins_by_claim_not_tuple_position(self):
        grid = np.asarray([0.25, 0.75])
        truths = (
            SyntheticQuantileEffectTruth(
                estimand_id="argon",
                treatment_name="treatment",
                outcome="velocity",
                reference_value=0.0,
                intervention_value=1.0,
                quantile_grid=grid,
                effect=np.asarray([30.0, 40.0]),
                is_active=True,
            ),
            SyntheticQuantileEffectTruth(
                estimand_id="current",
                treatment_name="treatment",
                outcome="temperature",
                reference_value=0.0,
                intervention_value=1.0,
                quantile_grid=grid,
                effect=np.asarray([10.0, 20.0]),
                is_active=True,
            ),
        )

        block = calibration_curve_block_from_benchmark(
            "block-001",
            truths,
            self.method_result(),
        )

        np.testing.assert_allclose(
            block.target.values,
            [[10.0, 20.0], [30.0, 40.0]],
        )


class V22SelectiveBenchmarkPlanTests(unittest.TestCase):
    @staticmethod
    def benchmark_block(plan, replicate_index):
        claim_ids = ("current:temperature",)
        quantiles = (0.25, 0.75)
        return CalibrationCurveBlock(
            block_id=(
                f"{plan.scenario_id}:n{plan.sample_size}:"
                f"r{replicate_index:03d}"
            ),
            target=CurveBundle(
                claim_ids,
                quantiles,
                np.asarray([[1.0, 2.0]]) + replicate_index,
            ),
            prediction=CurveBundlePrediction(
                claim_ids,
                quantiles,
                np.asarray([[0.8, 2.2]]) + replicate_index,
                np.asarray([[0.5, 0.6]]),
            ),
        )

    def test_plan_uses_new_randomness_and_first_finite_orbit(self):
        plan = V22SelectiveBenchmarkPlan()
        contract = v22_selective_benchmark_contract(plan)

        self.assertEqual(plan.random_seed, 52026)
        self.assertNotIn(plan.random_seed, (22026, 32026, 42026))
        self.assertEqual(len(plan.replicate_indices), 20)
        self.assertEqual(plan.alpha, 0.05)
        self.assertEqual(plan.sample_size, 48)
        self.assertEqual(plan.selected_backend, "cpu")
        self.assertEqual(contract.random_seed, plan.random_seed)

    def test_plan_signature_is_stable_and_covers_frozen_fields(self):
        plan = V22SelectiveBenchmarkPlan()
        payload = selective_benchmark_plan_payload(plan)
        signature = selective_benchmark_plan_signature(plan)

        self.assertEqual(payload["replicate_indices"], list(range(20)))
        self.assertEqual(len(signature), 64)
        self.assertEqual(signature, selective_benchmark_plan_signature(plan))

    def test_curve_block_payload_round_trip_is_exact(self):
        claim_ids = ("current:temperature", "argon:velocity")
        quantiles = (0.25, 0.75)
        block = CalibrationCurveBlock(
            block_id="round-trip",
            target=CurveBundle(
                claim_ids,
                quantiles,
                np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            ),
            prediction=CurveBundlePrediction(
                claim_ids,
                quantiles,
                np.asarray([[0.8, 2.2], [2.5, 4.5]]),
                np.asarray([[0.5, 0.6], [0.7, 0.8]]),
            ),
        )

        restored = curve_block_from_payload(curve_block_to_payload(block))

        self.assertEqual(restored.block_id, block.block_id)
        self.assertEqual(restored.target.claim_ids, block.target.claim_ids)
        self.assertEqual(
            restored.target.quantile_grid,
            block.target.quantile_grid,
        )
        np.testing.assert_array_equal(
            restored.target.values,
            block.target.values,
        )
        np.testing.assert_array_equal(
            restored.prediction.centre,
            block.prediction.centre,
        )
        np.testing.assert_array_equal(
            restored.prediction.scale,
            block.prediction.scale,
        )

    def test_checkpointed_runner_resumes_without_refitting(self):
        plan = V22SelectiveBenchmarkPlan(replicate_indices=(0, 1))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch(
                "experiments.pu_dcgp_v22.selective_benchmark_batch."
                "run_v22_selective_benchmark_block",
                side_effect=lambda index, frozen: self.benchmark_block(
                    frozen,
                    index,
                ),
            ) as runner:
                first = run_or_resume_selective_benchmark(output, plan)
                second = run_or_resume_selective_benchmark(output, plan)

            self.assertTrue(first.complete)
            self.assertTrue(second.complete)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(
                sorted(path.name for path in (output / "blocks").glob("*.json")),
                ["block_000.json", "block_001.json"],
            )
            self.assertFalse(list(output.rglob("*.tmp")))
            live = json.loads(
                (output / "live_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(live["completed_indices"], [0, 1])
            self.assertEqual(live["remaining_indices"], [])

    def test_complete_outer_orbit_writes_familywise_audit(self):
        plan = V22SelectiveBenchmarkPlan()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch(
                "experiments.pu_dcgp_v22.selective_benchmark_batch."
                "run_v22_selective_benchmark_block",
                side_effect=lambda index, frozen: self.benchmark_block(
                    frozen,
                    index,
                ),
            ):
                run_or_resume_selective_benchmark(output, plan)
            summary = summarize_selective_benchmark(output, plan)

            self.assertEqual(summary.block_count, 20)
            self.assertEqual(summary.calibration_block_count, 19)
            self.assertEqual(summary.calibration_rank, 19)
            self.assertEqual(summary.coverage_rate, 1.0)
            self.assertEqual(summary.target_coverage, 0.95)
            self.assertEqual(summary.false_selected_report_count, 0)
            self.assertTrue(summary.false_report_implies_miscoverage)
            self.assertTrue(summary.passed)
            payload = json.loads(
                (output / "selective_benchmark_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(payload["holdouts"]), 20)
            release = finalize_selective_benchmark(output, plan)
            verified = validate_selective_benchmark_release(output, plan)
            self.assertEqual(len(release.artifact_sha256), 23)
            self.assertEqual(verified, release)

    def test_release_verifier_rejects_changed_checkpoint_bytes(self):
        plan = V22SelectiveBenchmarkPlan()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch(
                "experiments.pu_dcgp_v22.selective_benchmark_batch."
                "run_v22_selective_benchmark_block",
                side_effect=lambda index, frozen: self.benchmark_block(
                    frozen,
                    index,
                ),
            ):
                run_or_resume_selective_benchmark(output, plan)
            finalize_selective_benchmark(output, plan)
            changed = output / "blocks" / "block_000.json"
            changed.write_bytes(changed.read_bytes() + b" ")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_selective_benchmark_release(output, plan)


if __name__ == "__main__":
    unittest.main()
