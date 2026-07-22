
from dataclasses import dataclass

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import (
    generate_insufficient_overlap_dataset,
    generate_module_sign_reversal_dataset,
    generate_sequence_aligned_drift_dataset,
)
from .config import PUDCGPConfig
from .contrast_support import audit_contrast_support
from .estimands import DOEEstimand
from .matched_effects import estimate_matched_distribution_effects
from .module_consistency import estimate_module_consistency
from .sequence_sensitivity import estimate_sequence_adjusted_effects


@dataclass(frozen=True, slots=True)
class SequenceDriftGeneratorAudit:

    sample_size: int
    replicate_count: int
    true_mean_effect: float
    average_raw_effect: float
    average_adjusted_effect: float
    raw_reversal_rate: float
    adjusted_recovery_rate: float
    sequence_gate_failure_rate: float
    leave_one_out_stable_rate: float
    module_mean_consistency_rate: float
    module_quantile_consistency_rate: float
    target_conditional_support_rate: float
    non_target_eligible_rate: float
    acceptance_rate: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ModuleReversalGeneratorAudit:

    sample_size: int
    replicate_count: int
    true_doe_1_effect: float
    true_doe_2_effect: float
    average_doe_1_effect: float
    average_doe_2_effect: float
    average_pooled_effect: float
    average_absolute_pooled_effect: float
    doe_1_sign_recovery_rate: float
    doe_2_sign_recovery_rate: float
    module_mean_gate_failure_rate: float
    module_quantile_gate_failure_rate: float
    leave_one_out_stable_rate: float
    sequence_sign_retained_rate: float
    target_eligible_support_rate: float
    non_target_eligible_rate: float
    acceptance_rate: float
    passed: bool


@dataclass(frozen=True, slots=True)
class OverlapFailureGeneratorAudit:

    sample_size: int
    replicate_count: int
    true_target_effect: float
    average_empirical_effect: float
    target_sign_recovery_rate: float
    average_target_matched_strata: float
    target_insufficient_rate: float
    target_only_strata_reason_rate: float
    non_target_not_insufficient_rate: float
    non_target_eligible_rate: float
    passed: bool


def audit_sequence_drift_generator(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 144,
) -> SequenceDriftGeneratorAudit:

    config = PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        random_seed=contract.random_seed,
    )
    target = DOEEstimand(
        estimand_id="current_norm_minus1_to_plus1",
        treatment_name="current_norm",
        reference_value=-1.0,
        intervention_value=1.0,
        claim_role="confirmatory",
        effect_direction="plus1_minus_minus1",
    )
    raw_effects = []
    adjusted_effects = []
    raw_reversals = []
    adjusted_recoveries = []
    sequence_failures = []
    leave_one_out_stable = []
    module_mean_consistency = []
    module_quantile_consistency = []
    target_conditional = []
    non_target_eligible = []
    true_effect = 44.0
    for replicate_index in range(contract.pilot_replicate_count):
        dataset = generate_sequence_aligned_drift_dataset(
            contract,
            sample_size,
            replicate_index,
        )
        matched = estimate_matched_distribution_effects(
            dataset.runs,
            config,
            target,
        ).aggregate_effects["temperature_c"]
        adjusted = estimate_sequence_adjusted_effects(
            dataset.runs,
            config,
            target,
        ).outcome_effects["temperature_c"]
        raw_effects.append(matched.mean_difference)
        adjusted_effects.append(adjusted.adjusted_mean_effect)
        raw_reversals.append(matched.mean_difference * true_effect < 0.0)
        adjusted_recoveries.append(
            adjusted.adjusted_mean_effect * true_effect > 0.0
        )
        sequence_failures.append(not adjusted.mean_sign_retained)
        leave_one_out_stable.append(matched.leave_one_out_sign_stable)
        modules = estimate_module_consistency(
            dataset.runs,
            config,
            target,
        ).outcome_consistency["temperature_c"]
        module_mean_consistency.append(modules.direction_consistent)
        module_quantile_consistency.append(
            modules.quantile_direction_consistent
        )
        target_conditional.append(
            audit_contrast_support(dataset.runs, target).support_level
            == "conditional"
        )
        other_levels = []
        for treatment_name in contract.treatment_names[1:]:
            other_levels.append(
                audit_contrast_support(
                    dataset.runs,
                    DOEEstimand(
                        estimand_id=treatment_name,
                        treatment_name=treatment_name,
                        reference_value=-1.0,
                        intervention_value=1.0,
                        claim_role="confirmatory",
                        effect_direction="plus1_minus_minus1",
                    ),
                ).support_level
                == "eligible"
            )
        non_target_eligible.append(all(other_levels))

    acceptance_rate = 0.90
    reversal_rate = float(np.mean(raw_reversals))
    recovery_rate = float(np.mean(adjusted_recoveries))
    failure_rate = float(np.mean(sequence_failures))
    target_support_rate = float(np.mean(target_conditional))
    non_target_support_rate = float(np.mean(non_target_eligible))
    passed = (
        reversal_rate >= acceptance_rate
        and recovery_rate >= acceptance_rate
        and failure_rate >= acceptance_rate
        and target_support_rate == 1.0
        and non_target_support_rate == 1.0
    )
    return SequenceDriftGeneratorAudit(
        sample_size=sample_size,
        replicate_count=contract.pilot_replicate_count,
        true_mean_effect=true_effect,
        average_raw_effect=float(np.mean(raw_effects)),
        average_adjusted_effect=float(np.mean(adjusted_effects)),
        raw_reversal_rate=reversal_rate,
        adjusted_recovery_rate=recovery_rate,
        sequence_gate_failure_rate=failure_rate,
        leave_one_out_stable_rate=float(np.mean(leave_one_out_stable)),
        module_mean_consistency_rate=float(
            np.mean(module_mean_consistency)
        ),
        module_quantile_consistency_rate=float(
            np.mean(module_quantile_consistency)
        ),
        target_conditional_support_rate=target_support_rate,
        non_target_eligible_rate=non_target_support_rate,
        acceptance_rate=acceptance_rate,
        passed=passed,
    )


def audit_module_reversal_generator(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 144,
) -> ModuleReversalGeneratorAudit:

    config = PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        random_seed=contract.random_seed,
    )
    target = DOEEstimand(
        estimand_id="powder_norm_minus1_to_plus1",
        treatment_name="powder_norm",
        reference_value=-1.0,
        intervention_value=1.0,
        claim_role="confirmatory",
        effect_direction="plus1_minus_minus1",
    )
    doe_1_effects = []
    doe_2_effects = []
    pooled_effects = []
    mean_gate_failures = []
    quantile_gate_failures = []
    leave_one_out_stable = []
    sequence_sign_retained = []
    target_eligible = []
    non_target_eligible = []
    for replicate_index in range(contract.pilot_replicate_count):
        dataset = generate_module_sign_reversal_dataset(
            contract,
            sample_size,
            replicate_index,
        )
        consistency = estimate_module_consistency(
            dataset.runs,
            config,
            target,
        ).outcome_consistency["particle_diameter_um"]
        doe_1_effects.append(consistency.module_mean_effects["DOE-1"])
        doe_2_effects.append(consistency.module_mean_effects["DOE-2"])
        pooled_effects.append(consistency.pooled_mean_effect)
        mean_gate_failures.append(not consistency.direction_consistent)
        quantile_gate_failures.append(
            not consistency.quantile_direction_consistent
        )
        matched = estimate_matched_distribution_effects(
            dataset.runs,
            config,
            target,
        ).aggregate_effects["particle_diameter_um"]
        sequence = estimate_sequence_adjusted_effects(
            dataset.runs,
            config,
            target,
        ).outcome_effects["particle_diameter_um"]
        leave_one_out_stable.append(matched.leave_one_out_sign_stable)
        sequence_sign_retained.append(sequence.mean_sign_retained)
        target_eligible.append(
            audit_contrast_support(dataset.runs, target).support_level
            == "eligible"
        )
        other_levels = []
        for treatment_name in contract.treatment_names:
            if treatment_name == "powder_norm":
                continue
            other_levels.append(
                audit_contrast_support(
                    dataset.runs,
                    DOEEstimand(
                        estimand_id=treatment_name,
                        treatment_name=treatment_name,
                        reference_value=-1.0,
                        intervention_value=1.0,
                        claim_role="confirmatory",
                        effect_direction="plus1_minus_minus1",
                    ),
                ).support_level
                == "eligible"
            )
        non_target_eligible.append(all(other_levels))

    acceptance_rate = 0.90
    doe_1_recovery = float(np.mean(np.asarray(doe_1_effects) < 0.0))
    doe_2_recovery = float(np.mean(np.asarray(doe_2_effects) > 0.0))
    mean_failure_rate = float(np.mean(mean_gate_failures))
    quantile_failure_rate = float(np.mean(quantile_gate_failures))
    target_support_rate = float(np.mean(target_eligible))
    non_target_support_rate = float(np.mean(non_target_eligible))
    passed = (
        doe_1_recovery >= acceptance_rate
        and doe_2_recovery >= acceptance_rate
        and mean_failure_rate >= acceptance_rate
        and quantile_failure_rate >= acceptance_rate
        and target_support_rate == 1.0
        and non_target_support_rate == 1.0
    )
    pooled_array = np.asarray(pooled_effects)
    return ModuleReversalGeneratorAudit(
        sample_size=sample_size,
        replicate_count=contract.pilot_replicate_count,
        true_doe_1_effect=-5.0,
        true_doe_2_effect=5.0,
        average_doe_1_effect=float(np.mean(doe_1_effects)),
        average_doe_2_effect=float(np.mean(doe_2_effects)),
        average_pooled_effect=float(np.mean(pooled_array)),
        average_absolute_pooled_effect=float(np.mean(np.abs(pooled_array))),
        doe_1_sign_recovery_rate=doe_1_recovery,
        doe_2_sign_recovery_rate=doe_2_recovery,
        module_mean_gate_failure_rate=mean_failure_rate,
        module_quantile_gate_failure_rate=quantile_failure_rate,
        leave_one_out_stable_rate=float(np.mean(leave_one_out_stable)),
        sequence_sign_retained_rate=float(np.mean(sequence_sign_retained)),
        target_eligible_support_rate=target_support_rate,
        non_target_eligible_rate=non_target_support_rate,
        acceptance_rate=acceptance_rate,
        passed=passed,
    )


def audit_overlap_failure_generator(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 144,
) -> OverlapFailureGeneratorAudit:

    config = PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        random_seed=contract.random_seed,
    )
    target = DOEEstimand(
        estimand_id="argon_norm_minus1_to_plus1",
        treatment_name="argon_norm",
        reference_value=-1.0,
        intervention_value=1.0,
        claim_role="confirmatory",
        effect_direction="plus1_minus_minus1",
    )
    empirical_effects = []
    matched_strata = []
    target_insufficient = []
    only_strata_reason = []
    non_target_not_insufficient = []
    non_target_eligible = []
    true_effect = 15.0
    for replicate_index in range(contract.pilot_replicate_count):
        dataset = generate_insufficient_overlap_dataset(
            contract,
            sample_size,
            replicate_index,
        )
        support = audit_contrast_support(dataset.runs, target)
        matched_strata.append(len(support.strata))
        target_insufficient.append(support.support_level == "insufficient")
        only_strata_reason.append(
            support.support_reasons
            == ("fewer than five exact-matching strata",)
        )
        empirical_effects.append(
            estimate_matched_distribution_effects(
                dataset.runs,
                config,
                target,
            ).aggregate_effects["velocity_m_s"].mean_difference
        )
        other_levels = []
        for treatment_name in contract.treatment_names:
            if treatment_name == "argon_norm":
                continue
            other_levels.append(
                audit_contrast_support(
                    dataset.runs,
                    DOEEstimand(
                        estimand_id=treatment_name,
                        treatment_name=treatment_name,
                        reference_value=-1.0,
                        intervention_value=1.0,
                        claim_role="confirmatory",
                        effect_direction="plus1_minus_minus1",
                    ),
                ).support_level
            )
        non_target_not_insufficient.append(
            all(level != "insufficient" for level in other_levels)
        )
        non_target_eligible.append(
            all(level == "eligible" for level in other_levels)
        )

    empirical_array = np.asarray(empirical_effects)
    insufficient_rate = float(np.mean(target_insufficient))
    non_target_safe_rate = float(np.mean(non_target_not_insufficient))
    average_strata = float(np.mean(matched_strata))
    passed = (
        average_strata == 4.0
        and insufficient_rate == 1.0
        and non_target_safe_rate == 1.0
    )
    return OverlapFailureGeneratorAudit(
        sample_size=sample_size,
        replicate_count=contract.pilot_replicate_count,
        true_target_effect=true_effect,
        average_empirical_effect=float(np.mean(empirical_array)),
        target_sign_recovery_rate=float(
            np.mean(empirical_array * true_effect > 0.0)
        ),
        average_target_matched_strata=average_strata,
        target_insufficient_rate=insufficient_rate,
        target_only_strata_reason_rate=float(np.mean(only_strata_reason)),
        non_target_not_insufficient_rate=non_target_safe_rate,
        non_target_eligible_rate=float(np.mean(non_target_eligible)),
        passed=passed,
    )
