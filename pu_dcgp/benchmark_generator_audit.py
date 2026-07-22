
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import (
    SyntheticBenchmarkDataset,
    benchmark_quantile_truths,
    generate_identified_balanced_dataset,
    generate_identified_heterogeneous_dataset,
)
from .config import PUDCGPConfig
from .estimands import DOEEstimand
from .matched_effects import estimate_matched_distribution_effects


@dataclass(frozen=True, slots=True)
class GeneratorOracleAuditEntry:

    treatment_name: str
    outcome: str
    is_active: bool
    normalized_irmse: float
    median_bias: float


@dataclass(frozen=True, slots=True)
class GeneratorOracleAudit:

    scenario_id: str
    sample_size: int
    replicate_count: int
    acceptance_threshold: float
    entries: tuple[GeneratorOracleAuditEntry, ...]
    maximum_normalized_irmse: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ParticleCountIsolationAuditEntry:

    treatment_name: str
    outcome: str
    normalized_irmse: float
    median_change: float


@dataclass(frozen=True, slots=True)
class ParticleCountIsolationAudit:

    sample_size: int
    replicate_count: int
    acceptance_threshold: float
    entries: tuple[ParticleCountIsolationAuditEntry, ...]
    maximum_normalized_irmse: float
    passed: bool


def audit_identified_generator_oracle(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 144,
) -> GeneratorOracleAudit:

    return _audit_generator_oracle(
        contract,
        sample_size,
        generate_identified_balanced_dataset,
    )


def audit_identified_heterogeneous_generator_oracle(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 144,
) -> GeneratorOracleAudit:

    return _audit_generator_oracle(
        contract,
        sample_size,
        generate_identified_heterogeneous_dataset,
    )


def audit_particle_count_isolation(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 144,
) -> ParticleCountIsolationAudit:

    config = PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        random_seed=contract.random_seed,
    )
    differences: dict[tuple[str, str], list[np.ndarray]] = {
        (treatment, outcome): []
        for treatment in contract.treatment_names
        for outcome in contract.outcome_names
    }
    for replicate_index in range(contract.pilot_replicate_count):
        balanced = generate_identified_balanced_dataset(
            contract,
            sample_size,
            replicate_index,
        )
        heterogeneous = generate_identified_heterogeneous_dataset(
            contract,
            sample_size,
            replicate_index,
        )
        for treatment_name in contract.treatment_names:
            estimand = DOEEstimand(
                estimand_id=f"{treatment_name}_minus1_to_plus1",
                treatment_name=treatment_name,
                reference_value=-1.0,
                intervention_value=1.0,
                claim_role="confirmatory",
                effect_direction="plus1_minus_minus1",
            )
            balanced_effect = estimate_matched_distribution_effects(
                balanced.runs,
                config,
                estimand,
            )
            heterogeneous_effect = estimate_matched_distribution_effects(
                heterogeneous.runs,
                config,
                estimand,
            )
            for outcome in contract.outcome_names:
                differences[(treatment_name, outcome)].append(
                    heterogeneous_effect.aggregate_effects[
                        outcome
                    ].quantile_difference
                    - balanced_effect.aggregate_effects[
                        outcome
                    ].quantile_difference
                )

    outcome_scales = {
        mechanism.outcome: mechanism.baseline_scale
        for mechanism in contract.mechanisms
    }
    entries = []
    for key, paired_differences in differences.items():
        average_change = np.mean(paired_differences, axis=0)
        entries.append(
            ParticleCountIsolationAuditEntry(
                treatment_name=key[0],
                outcome=key[1],
                normalized_irmse=float(
                    np.sqrt(np.mean(np.square(average_change)))
                    / outcome_scales[key[1]]
                ),
                median_change=float(
                    average_change[len(average_change) // 2]
                ),
            )
        )
    threshold = 0.05
    maximum_error = max(entry.normalized_irmse for entry in entries)
    return ParticleCountIsolationAudit(
        sample_size=sample_size,
        replicate_count=contract.pilot_replicate_count,
        acceptance_threshold=threshold,
        entries=tuple(entries),
        maximum_normalized_irmse=maximum_error,
        passed=maximum_error <= threshold,
    )


def _audit_generator_oracle(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    generator: Callable[
        [SyntheticBenchmarkContract, int, int],
        SyntheticBenchmarkDataset,
    ],
) -> GeneratorOracleAudit:

    config = PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        random_seed=contract.random_seed,
    )
    truths = {
        (truth.treatment_name, truth.outcome): truth
        for truth in benchmark_quantile_truths(contract)
    }
    empirical: dict[tuple[str, str], list[np.ndarray]] = {
        key: [] for key in truths
    }
    for replicate_index in range(contract.pilot_replicate_count):
        dataset = generator(
            contract,
            sample_size,
            replicate_index,
        )
        for treatment_name in contract.treatment_names:
            result = estimate_matched_distribution_effects(
                dataset.runs,
                config,
                DOEEstimand(
                    estimand_id=f"{treatment_name}_minus1_to_plus1",
                    treatment_name=treatment_name,
                    reference_value=-1.0,
                    intervention_value=1.0,
                    claim_role="confirmatory",
                    effect_direction="plus1_minus_minus1",
                ),
            )
            for outcome in contract.outcome_names:
                empirical[(treatment_name, outcome)].append(
                    result.aggregate_effects[outcome].quantile_difference
                )

    outcome_scales = {
        mechanism.outcome: mechanism.baseline_scale
        for mechanism in contract.mechanisms
    }
    entries = []
    for key, truth in truths.items():
        average_effect = np.mean(empirical[key], axis=0)
        error = average_effect - truth.effect
        entries.append(
            GeneratorOracleAuditEntry(
                treatment_name=key[0],
                outcome=key[1],
                is_active=truth.is_active,
                normalized_irmse=float(
                    np.sqrt(np.mean(np.square(error)))
                    / outcome_scales[key[1]]
                ),
                median_bias=float(error[len(error) // 2]),
            )
        )
    threshold = 0.05
    maximum_error = max(entry.normalized_irmse for entry in entries)
    return GeneratorOracleAudit(
        scenario_id=generator(contract, sample_size, 0).scenario_id,
        sample_size=sample_size,
        replicate_count=contract.pilot_replicate_count,
        acceptance_threshold=threshold,
        entries=tuple(entries),
        maximum_normalized_irmse=maximum_error,
        passed=maximum_error <= threshold,
    )
