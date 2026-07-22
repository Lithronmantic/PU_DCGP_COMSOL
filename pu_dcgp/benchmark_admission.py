
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .admission_gate import (
    EffectAdmissionDecision,
    EffectAdmissionEvidence,
    decide_effect_admission,
)
from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import SyntheticBenchmarkDataset
from .benchmark_methods import BenchmarkMethodResult
from .config import PUDCGPConfig
from .contrast_support import audit_contrast_support
from .estimands import DOEEstimand
from .matched_effects import estimate_matched_distribution_effects
from .module_consistency import estimate_module_consistency
from .sequence_sensitivity import estimate_sequence_adjusted_effects


@dataclass(frozen=True, slots=True)
class SyntheticAdmissionObservation:

    scenario_id: str
    sample_size: int
    replicate_index: int
    treatment_name: str
    outcome: str
    is_active: bool
    status: str
    failed_gates: tuple[str, ...]


def evaluate_synthetic_admission_decisions(
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
    pu_result: BenchmarkMethodResult,
) -> tuple[EffectAdmissionDecision, ...]:

    effect_map = {
        (effect.estimand_id, effect.outcome): effect
        for effect in pu_result.effects
    }
    weights = _trapezoid_weights(np.asarray(contract.quantile_grid, dtype=float))
    normal_critical = NormalDist().inv_cdf(
        0.5 + config.effect_interval_level / 2
    )
    decisions = []
    for treatment_name in contract.treatment_names:
        truth = next(
            truth
            for truth in dataset.truths
            if truth.treatment_name == treatment_name
        )
        estimand = DOEEstimand(
            estimand_id=truth.estimand_id,
            treatment_name=treatment_name,
            reference_value=truth.reference_value,
            intervention_value=truth.intervention_value,
            claim_role=(
                "exploratory" if treatment_name == "distance_norm" else "confirmatory"
            ),
            effect_direction="plus1_minus_minus1",
        )
        support = audit_contrast_support(dataset.runs, estimand)
        matched = estimate_matched_distribution_effects(
            dataset.runs,
            config,
            estimand,
        )
        sequence = estimate_sequence_adjusted_effects(
            dataset.runs,
            config,
            estimand,
        )
        modules = estimate_module_consistency(
            dataset.runs,
            config,
            estimand,
        )
        for outcome in contract.outcome_names:
            effect = effect_map[(estimand.estimand_id, outcome)]
            point_mean = float(weights @ effect.point_effect)
            mean_variance = float(
                weights @ effect.effect_covariance @ weights
            )
            mean_half_width = normal_critical * np.sqrt(max(mean_variance, 0.0))
            mean_lower = point_mean - mean_half_width
            mean_upper = point_mean + mean_half_width
            simultaneous_excludes_zero = bool(
                np.all(effect.lower_bound > 0.0)
                or np.all(effect.upper_bound < 0.0)
            )
            evidence = EffectAdmissionEvidence(
                structural_support=support.support_level != "insufficient",
                leave_one_stratum_sign=(
                    matched.aggregate_effects[outcome].leave_one_out_sign_stable
                ),
                sequence_sign=(
                    sequence.outcome_effects[outcome].mean_sign_retained
                ),
                module_mean_direction=(
                    modules.outcome_consistency[outcome].direction_consistent
                ),
                module_quantile_direction=(
                    modules.outcome_consistency[
                        outcome
                    ].quantile_direction_consistent
                ),
                mean_interval=mean_lower > 0.0 or mean_upper < 0.0,
                simultaneous_quantile_band=simultaneous_excludes_zero,
            )
            decisions.append(
                decide_effect_admission(
                    estimand=estimand,
                    outcome=outcome,
                    support_level=support.support_level,
                    conditional_reasons=(
                        support.support_reasons
                        if support.support_level == "conditional"
                        else ()
                    ),
                    evidence=evidence,
                    point_mean_effect=point_mean,
                    mean_lower_bound=mean_lower,
                    mean_upper_bound=mean_upper,
                    simultaneous_lower_min=float(np.min(effect.lower_bound)),
                    simultaneous_upper_max=float(np.max(effect.upper_bound)),
                )
            )
    return tuple(decisions)


def synthetic_admission_observations(
    dataset: SyntheticBenchmarkDataset,
    decisions: tuple[EffectAdmissionDecision, ...],
) -> tuple[SyntheticAdmissionObservation, ...]:

    truth_map = {
        (truth.estimand_id, truth.outcome): truth for truth in dataset.truths
    }
    return tuple(
        SyntheticAdmissionObservation(
            scenario_id=dataset.scenario_id,
            sample_size=dataset.sample_size,
            replicate_index=dataset.replicate_index,
            treatment_name=decision.estimand.treatment_name,
            outcome=decision.outcome,
            is_active=truth_map[
                (decision.estimand.estimand_id, decision.outcome)
            ].is_active,
            status=decision.status,
            failed_gates=decision.failed_gates,
        )
        for decision in decisions
    )


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
