
from dataclasses import dataclass

from .config import PUDCGPConfig
from .contrast_support import audit_contrast_support
from .contracts import RunBatch
from .effect_uncertainty import bootstrap_matched_distribution_effect
from .estimands import DOEEstimand, a_group_doe_estimands
from .matched_effects import estimate_matched_distribution_effects
from .module_consistency import estimate_module_consistency
from .sequence_sensitivity import estimate_sequence_adjusted_effects


GATE_NAMES = (
    "structural_support",
    "leave_one_stratum_sign",
    "sequence_sign",
    "module_mean_direction",
    "module_quantile_direction",
    "mean_interval",
    "simultaneous_quantile_band",
)


@dataclass(frozen=True, slots=True)
class EffectAdmissionEvidence:

    structural_support: bool
    leave_one_stratum_sign: bool
    sequence_sign: bool
    module_mean_direction: bool
    module_quantile_direction: bool
    mean_interval: bool
    simultaneous_quantile_band: bool

    def as_dict(self) -> dict[str, bool]:

        return {name: getattr(self, name) for name in GATE_NAMES}


@dataclass(frozen=True, slots=True)
class EffectAdmissionDecision:

    estimand: DOEEstimand
    outcome: str
    status: str
    support_level: str
    evidence: EffectAdmissionEvidence
    passed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    conditional_reasons: tuple[str, ...]
    point_mean_effect: float
    mean_lower_bound: float
    mean_upper_bound: float
    simultaneous_lower_min: float
    simultaneous_upper_max: float


def decide_effect_admission(
    estimand: DOEEstimand,
    outcome: str,
    support_level: str,
    conditional_reasons: tuple[str, ...],
    evidence: EffectAdmissionEvidence,
    point_mean_effect: float,
    mean_lower_bound: float,
    mean_upper_bound: float,
    simultaneous_lower_min: float,
    simultaneous_upper_max: float,
) -> EffectAdmissionDecision:

    gate_results = evidence.as_dict()
    passed_gates = tuple(
        name for name in GATE_NAMES if gate_results[name]
    )
    failed_gates = tuple(
        name for name in GATE_NAMES if not gate_results[name]
    )
    if not evidence.structural_support:
        status = "insufficient_support"
    elif failed_gates:
        status = "abstain"
    elif estimand.claim_role == "exploratory":
        status = "exploratory_admit"
    elif support_level == "conditional":
        status = "conditional_admit"
    else:
        status = "admit"
    return EffectAdmissionDecision(
        estimand=estimand,
        outcome=outcome,
        status=status,
        support_level=support_level,
        evidence=evidence,
        passed_gates=passed_gates,
        failed_gates=failed_gates,
        conditional_reasons=conditional_reasons,
        point_mean_effect=point_mean_effect,
        mean_lower_bound=mean_lower_bound,
        mean_upper_bound=mean_upper_bound,
        simultaneous_lower_min=simultaneous_lower_min,
        simultaneous_upper_max=simultaneous_upper_max,
    )


def evaluate_effect_admission(
    runs: RunBatch,
    config: PUDCGPConfig,
    estimand: DOEEstimand,
    outcome: str,
) -> EffectAdmissionDecision:

    support = audit_contrast_support(runs, estimand)
    matched = estimate_matched_distribution_effects(runs, config, estimand)
    sequence = estimate_sequence_adjusted_effects(runs, config, estimand)
    modules = estimate_module_consistency(runs, config, estimand)
    uncertainty = bootstrap_matched_distribution_effect(
        runs,
        config,
        estimand,
        outcome,
    )
    matched_outcome = matched.aggregate_effects[outcome]
    sequence_outcome = sequence.outcome_effects[outcome]
    module_outcome = modules.outcome_consistency[outcome]
    mean_interval = uncertainty.mean_interval
    quantile_band = uncertainty.quantile_band
    evidence = EffectAdmissionEvidence(
        structural_support=support.support_level != "insufficient",
        leave_one_stratum_sign=matched_outcome.leave_one_out_sign_stable,
        sequence_sign=sequence_outcome.mean_sign_retained,
        module_mean_direction=module_outcome.direction_consistent,
        module_quantile_direction=(
            module_outcome.quantile_direction_consistent
        ),
        mean_interval=mean_interval.interval_excludes_zero,
        simultaneous_quantile_band=(
            quantile_band.simultaneous_excludes_zero_everywhere
        ),
    )
    conditional_reasons = (
        support.support_reasons
        if support.support_level == "conditional"
        else ()
    )
    return decide_effect_admission(
        estimand=estimand,
        outcome=outcome,
        support_level=support.support_level,
        conditional_reasons=conditional_reasons,
        evidence=evidence,
        point_mean_effect=mean_interval.point_estimate,
        mean_lower_bound=mean_interval.lower_bound,
        mean_upper_bound=mean_interval.upper_bound,
        simultaneous_lower_min=float(
            quantile_band.simultaneous_lower_bound.min()
        ),
        simultaneous_upper_max=float(
            quantile_band.simultaneous_upper_bound.max()
        ),
    )


def evaluate_all_effect_admissions(
    runs: RunBatch,
    config: PUDCGPConfig,
) -> tuple[EffectAdmissionDecision, ...]:

    return tuple(
        evaluate_effect_admission(runs, config, estimand, outcome)
        for estimand in a_group_doe_estimands()
        for outcome in config.outcome_columns
    )
