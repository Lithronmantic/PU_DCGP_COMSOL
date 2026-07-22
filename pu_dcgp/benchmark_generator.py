"""Known-truth synthetic data generator for the PU-DCGP benchmark."""

from dataclasses import dataclass, replace
from itertools import product
from statistics import NormalDist
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .benchmark_contract import (
    SyntheticBenchmarkContract,
    SyntheticOutcomeMechanism,
)
from .contracts import RunBatch


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticQuantileEffectTruth:
    """Analytic equal-anchor intervention effect for one outcome."""

    estimand_id: str
    treatment_name: str
    outcome: str
    reference_value: float
    intervention_value: float
    quantile_grid: FloatArray
    effect: FloatArray
    is_active: bool


@dataclass(frozen=True, slots=True)
class SyntheticModuleEffectTruth:
    """Analytic quantile effect within one synthetic DOE module."""

    doe_module: str
    estimand_id: str
    treatment_name: str
    outcome: str
    effect_multiplier: float
    quantile_grid: FloatArray
    effect: FloatArray


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkDataset:
    """One generated run batch paired with its analytic causal truths."""

    scenario_id: str
    sample_size: int
    replicate_index: int
    particle_counts: NDArray[np.int64]
    scenario_parameters: Mapping[str, float]
    runs: RunBatch
    truths: tuple[SyntheticQuantileEffectTruth, ...]
    module_truths: tuple[SyntheticModuleEffectTruth, ...]


@dataclass(frozen=True, slots=True)
class SyntheticRunDesign:
    """Randomized factorial anchors and interior settings for one replicate."""

    treatment_values: FloatArray
    doe_modules: tuple[str, ...]
    execution_order: FloatArray
    is_factorial_anchor: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SequenceGapSummary:
    """Matched anchor separation under one proposed execution order."""

    median_gap: float
    normalized_median_gap: float
    positive_gaps: int
    negative_gaps: int


def generate_identified_design(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticRunDesign:
    """Generate the frozen two-module anchor-plus-interior design."""

    scenario = contract.scenarios[0]
    if sample_size not in scenario.sample_sizes:
        raise ValueError("sample_size is not part of the frozen benchmark grid")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [contract.random_seed, sample_size, replicate_index, 1401]
        )
    )
    anchor_count = 2 * sample_size // 3
    interior_count = sample_size - anchor_count
    anchor_cells = np.asarray(
        tuple(product((-1.0, 1.0), repeat=len(contract.treatment_names))),
        dtype=float,
    )
    repetitions = anchor_count // len(anchor_cells)

    interior = np.empty((interior_count, len(contract.treatment_names)))
    for column in range(interior.shape[1]):
        strata = rng.permutation(interior_count)
        unit_values = (strata + rng.random(interior_count)) / interior_count
        interior[:, column] = -0.8 + 1.6 * unit_values

    for _ in range(1000):
        blocks = []
        for repeat_index in range(repetitions):
            order = rng.permutation(len(anchor_cells))
            blocks.append(
                (
                    anchor_cells[order],
                    (f"DOE-{repeat_index % 2 + 1} synthetic",) * len(order),
                    np.ones(len(order), dtype=bool),
                )
            )
        for start in range(0, interior_count, len(anchor_cells)):
            values = interior[start:start + len(anchor_cells)]
            order = rng.permutation(len(values))
            blocks.append(
                (
                    values[order],
                    tuple(
                        f"DOE-{index % 2 + 1} synthetic"
                        for index in range(len(values))
                    ),
                    np.zeros(len(values), dtype=bool),
                )
            )
        block_order = rng.permutation(len(blocks))
        ordered_blocks = [blocks[index] for index in block_order]
        treatment_values = np.vstack(
            [block[0] for block in ordered_blocks]
        )
        modules = tuple(
            module
            for block in ordered_blocks
            for module in block[1]
        )
        is_anchor = np.concatenate(
            [block[2] for block in ordered_blocks]
        )
        if _has_balanced_sequence_support(treatment_values, is_anchor):
            break
    else:
        raise RuntimeError("failed to construct the frozen balanced sequence")
    return SyntheticRunDesign(
        treatment_values=treatment_values,
        doe_modules=modules,
        execution_order=np.arange(1, sample_size + 1, dtype=float),
        is_factorial_anchor=is_anchor,
    )


def _has_balanced_sequence_support(
    treatment_values: FloatArray,
    is_anchor: NDArray[np.bool_],
) -> bool:
    """Check the clean design against the frozen sequence-support thresholds."""

    order = np.arange(1, len(treatment_values) + 1, dtype=float)
    span = order[-1] - order[0]
    anchor_rows = np.flatnonzero(is_anchor)
    for treatment_index in range(treatment_values.shape[1]):
        other_indices = [
            index for index in range(treatment_values.shape[1])
            if index != treatment_index
        ]
        grouped: dict[tuple[float, ...], list[int]] = {}
        for row in anchor_rows:
            key = tuple(treatment_values[row, other_indices])
            grouped.setdefault(key, []).append(row)
        gaps = []
        for rows in grouped.values():
            rows_array = np.asarray(rows, dtype=int)
            values = treatment_values[rows_array, treatment_index]
            gaps.append(
                order[rows_array[values == 1.0]].mean()
                - order[rows_array[values == -1.0]].mean()
            )
        gaps_array = np.asarray(gaps)
        if (
            not np.any(gaps_array > 0.0)
            or not np.any(gaps_array < 0.0)
            or np.median(np.abs(gaps_array)) / span > 0.10
        ):
            return False
    return True


def generate_sequence_aligned_design(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticRunDesign:
    """Order high-current anchors later without losing structural support."""

    base = generate_identified_design(contract, sample_size, replicate_index)
    current_index = contract.treatment_names.index("current_norm")
    base_order = np.arange(sample_size, dtype=float)
    for shift_fraction in np.linspace(0.04, 0.24, 81):
        score = (
            base_order
            + shift_fraction
            * sample_size
            * base.treatment_values[:, current_index]
        )
        permutation = np.argsort(score, kind="stable")
        values = base.treatment_values[permutation]
        is_anchor = base.is_factorial_anchor[permutation]
        summaries = tuple(
            _sequence_gap_summary(values, is_anchor, treatment_index)
            for treatment_index in range(values.shape[1])
        )
        target = summaries[current_index]
        non_targets = tuple(
            summary
            for index, summary in enumerate(summaries)
            if index != current_index
        )
        if (
            0.10 < target.normalized_median_gap < 0.25
            and target.positive_gaps == 8
            and all(
                summary.normalized_median_gap <= 0.10
                and summary.positive_gaps > 0
                and summary.negative_gaps > 0
                for summary in non_targets
            )
        ):
            return SyntheticRunDesign(
                treatment_values=values,
                doe_modules=tuple(
                    base.doe_modules[index] for index in permutation
                ),
                execution_order=np.arange(
                    1,
                    sample_size + 1,
                    dtype=float,
                ),
                is_factorial_anchor=is_anchor,
            )
    permutation = _balanced_sequence_aligned_permutation(base, current_index)
    return SyntheticRunDesign(
        treatment_values=base.treatment_values[permutation],
        doe_modules=tuple(base.doe_modules[index] for index in permutation),
        execution_order=np.arange(1, sample_size + 1, dtype=float),
        is_factorial_anchor=base.is_factorial_anchor[permutation],
    )


def _balanced_sequence_aligned_permutation(
    design: SyntheticRunDesign,
    current_index: int,
) -> NDArray[np.int64]:
    """Construct the frozen balanced fallback when a shifted order has no solution."""

    anchor_rows = np.flatnonzero(design.is_factorial_anchor)
    interior_rows = np.flatnonzero(~design.is_factorial_anchor)
    non_current_indices = [
        index for index in range(design.treatment_values.shape[1])
        if index != current_index
    ]
    anchor_patterns = (
        np.asarray((5, 0, 1, 4, 2, 6, 3, 7)),
        np.asarray((1, 6, 7, 2, 3, 4, 5, 0)),
    )
    pattern_positions = tuple(np.argsort(pattern) for pattern in anchor_patterns)
    repetitions = len(anchor_rows) // 16
    current_shift = 1.25 * repetitions + 0.25
    occurrence_counts: dict[tuple[float, int], int] = {}
    anchor_scores = []
    for row in anchor_rows:
        current_value = design.treatment_values[row, current_index]
        combination = sum(
            int(design.treatment_values[row, index] > 0.0) << bit
            for bit, index in enumerate(non_current_indices)
        )
        key = (current_value, combination)
        occurrence = occurrence_counts.get(key, 0)
        occurrence_counts[key] = occurrence + 1
        anchor_scores.append(
            occurrence * 8
            + pattern_positions[occurrence % 2][combination]
            + (current_shift if current_value > 0.0 else 0.0)
        )
    ordered_anchors = anchor_rows[
        np.argsort(np.asarray(anchor_scores), kind="stable")
    ]
    permutation = []
    anchor_position = 0
    interior_position = 0
    for slot in range(len(design.treatment_values)):
        if slot % 3 == 2:
            permutation.append(interior_rows[interior_position])
            interior_position += 1
        else:
            permutation.append(ordered_anchors[anchor_position])
            anchor_position += 1
    return np.asarray(permutation, dtype=np.int64)


def _sequence_gap_summary(
    treatment_values: FloatArray,
    is_anchor: NDArray[np.bool_],
    treatment_index: int,
) -> SequenceGapSummary:
    """Summarize matched anchor order gaps for one treatment."""

    order = np.arange(1, len(treatment_values) + 1, dtype=float)
    other_indices = [
        index for index in range(treatment_values.shape[1])
        if index != treatment_index
    ]
    grouped: dict[tuple[float, ...], list[int]] = {}
    for row in np.flatnonzero(is_anchor):
        key = tuple(treatment_values[row, other_indices])
        grouped.setdefault(key, []).append(row)
    gaps = []
    for rows in grouped.values():
        rows_array = np.asarray(rows, dtype=int)
        values = treatment_values[rows_array, treatment_index]
        gaps.append(
            order[rows_array[values == 1.0]].mean()
            - order[rows_array[values == -1.0]].mean()
        )
    gaps_array = np.asarray(gaps)
    median_gap = float(np.median(gaps_array))
    return SequenceGapSummary(
        median_gap=median_gap,
        normalized_median_gap=float(
            np.median(np.abs(gaps_array)) / (len(treatment_values) - 1)
        ),
        positive_gaps=int(np.sum(gaps_array > 0.0)),
        negative_gaps=int(np.sum(gaps_array < 0.0)),
    )


def analytic_expected_run_quantiles(
    mechanism: SyntheticOutcomeMechanism,
    treatment_values: FloatArray,
    quantile_grid: FloatArray,
) -> FloatArray:
    """Return the expected run-level particle quantile functions."""

    values = np.atleast_2d(np.asarray(treatment_values, dtype=float))
    grid = np.asarray(quantile_grid, dtype=float)
    location = (
        mechanism.baseline_location
        + values @ np.asarray(mechanism.location_linear)
        + values ** 3 @ np.asarray(mechanism.location_cubic)
    )
    particle_scale = mechanism.baseline_scale * np.exp(
        values @ np.asarray(mechanism.log_scale_linear)
    )
    normal_quantiles = np.asarray(
        [NormalDist().inv_cdf(float(probability)) for probability in grid]
    )
    return location[:, None] + particle_scale[:, None] * normal_quantiles


def benchmark_quantile_truths(
    contract: SyntheticBenchmarkContract,
) -> tuple[SyntheticQuantileEffectTruth, ...]:
    """Return equal-anchor quantile effects for all treatment-outcome pairs."""

    grid = np.asarray(contract.quantile_grid, dtype=float)
    truths = []
    factor_count = len(contract.treatment_names)
    for mechanism in contract.mechanisms:
        for treatment_index, treatment_name in enumerate(
            contract.treatment_names
        ):
            other_indices = tuple(
                index for index in range(factor_count)
                if index != treatment_index
            )
            stratum_effects = []
            for other_values in product((-1.0, 1.0), repeat=factor_count - 1):
                reference = np.zeros(factor_count)
                intervention = np.zeros(factor_count)
                reference[treatment_index] = -1.0
                intervention[treatment_index] = 1.0
                reference[list(other_indices)] = other_values
                intervention[list(other_indices)] = other_values
                quantiles = analytic_expected_run_quantiles(
                    mechanism,
                    np.vstack((reference, intervention)),
                    grid,
                )
                stratum_effects.append(quantiles[1] - quantiles[0])
            effect = np.mean(stratum_effects, axis=0)
            active = any(
                coefficient != 0.0
                for coefficient in (
                    mechanism.location_linear[treatment_index],
                    mechanism.location_cubic[treatment_index],
                    mechanism.log_scale_linear[treatment_index],
                )
            )
            truths.append(
                SyntheticQuantileEffectTruth(
                    estimand_id=f"{treatment_name}_minus1_to_plus1",
                    treatment_name=treatment_name,
                    outcome=mechanism.outcome,
                    reference_value=-1.0,
                    intervention_value=1.0,
                    quantile_grid=grid.copy(),
                    effect=np.asarray(effect, dtype=float),
                    is_active=active,
                )
            )
    return tuple(truths)


def module_reversal_quantile_truths(
    contract: SyntheticBenchmarkContract,
) -> tuple[
    tuple[SyntheticQuantileEffectTruth, ...],
    tuple[SyntheticModuleEffectTruth, ...],
]:
    """Return aggregate-zero and opposite module truths for the target pair."""

    base_truths = benchmark_quantile_truths(contract)
    target = next(
        truth
        for truth in base_truths
        if truth.treatment_name == "powder_norm"
        and truth.outcome == "particle_diameter_um"
    )
    multipliers = contract.scenarios[3].module_effect_multipliers
    module_truths = tuple(
        SyntheticModuleEffectTruth(
            doe_module=f"DOE-{index + 1}",
            estimand_id=target.estimand_id,
            treatment_name=target.treatment_name,
            outcome=target.outcome,
            effect_multiplier=multiplier,
            quantile_grid=target.quantile_grid.copy(),
            effect=multiplier * target.effect,
        )
        for index, multiplier in enumerate(multipliers)
    )
    aggregate_effect = np.mean(
        [truth.effect for truth in module_truths],
        axis=0,
    )
    aggregate_truths = tuple(
        replace(truth, effect=np.asarray(aggregate_effect, dtype=float))
        if truth.treatment_name == target.treatment_name
        and truth.outcome == target.outcome
        else truth
        for truth in base_truths
    )
    return aggregate_truths, module_truths


def design_matched_quantile_truths(
    contract: SyntheticBenchmarkContract,
    treatment_values: FloatArray,
) -> tuple[SyntheticQuantileEffectTruth, ...]:
    """Return equal-stratum truths for the exact arms present in one design."""

    values = np.asarray(treatment_values, dtype=float)
    grid = np.asarray(contract.quantile_grid, dtype=float)
    truths = []
    for mechanism in contract.mechanisms:
        for treatment_index, treatment_name in enumerate(
            contract.treatment_names
        ):
            other_indices = [
                index for index in range(values.shape[1])
                if index != treatment_index
            ]
            grouped: dict[tuple[float, ...], list[int]] = {}
            for row_index, row in enumerate(values):
                grouped.setdefault(tuple(row[other_indices]), []).append(
                    row_index
                )
            effects = []
            for rows in grouped.values():
                rows_array = np.asarray(rows, dtype=int)
                arm_values = values[rows_array, treatment_index]
                reference_rows = rows_array[arm_values == -1.0]
                intervention_rows = rows_array[arm_values == 1.0]
                if len(reference_rows) == 0 or len(intervention_rows) == 0:
                    continue
                quantiles = analytic_expected_run_quantiles(
                    mechanism,
                    np.vstack(
                        (
                            values[reference_rows[0]],
                            values[intervention_rows[0]],
                        )
                    ),
                    grid,
                )
                effects.append(quantiles[1] - quantiles[0])
            effect = np.mean(effects, axis=0)
            active = any(
                coefficient != 0.0
                for coefficient in (
                    mechanism.location_linear[treatment_index],
                    mechanism.location_cubic[treatment_index],
                    mechanism.log_scale_linear[treatment_index],
                )
            )
            truths.append(
                SyntheticQuantileEffectTruth(
                    estimand_id=f"{treatment_name}_minus1_to_plus1",
                    treatment_name=treatment_name,
                    outcome=mechanism.outcome,
                    reference_value=-1.0,
                    intervention_value=1.0,
                    quantile_grid=grid.copy(),
                    effect=np.asarray(effect, dtype=float),
                    is_active=active,
                )
            )
    return tuple(truths)


def generate_identified_balanced_dataset(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticBenchmarkDataset:
    """Generate the clean identified scenario with 80 particles per run."""

    design = generate_identified_design(
        contract,
        sample_size,
        replicate_index,
    )
    particle_count = contract.scenarios[0].particle_count_range[0]
    return _sample_identified_dataset(
        contract=contract,
        scenario_id="identified_balanced_particles",
        design=design,
        sample_size=sample_size,
        replicate_index=replicate_index,
        particle_counts=np.full(sample_size, particle_count, dtype=np.int64),
    )


def generate_treatment_dependent_particle_counts(
    contract: SyntheticBenchmarkContract,
    treatment_values: FloatArray,
    replicate_index: int = 0,
) -> NDArray[np.int64]:
    """Draw outcome-independent counts related to powder and distance settings."""

    values = np.asarray(treatment_values, dtype=float)
    sample_size = len(values)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [contract.random_seed, sample_size, replicate_index, 1403]
        )
    )
    powder_index = contract.treatment_names.index("powder_norm")
    distance_index = contract.treatment_names.index("distance_norm")
    log_count = (
        np.log(80.0)
        - 0.55 * values[:, powder_index]
        + 0.35 * values[:, distance_index]
        + rng.normal(0.0, 0.55, sample_size)
    )
    lower, upper = contract.scenarios[1].particle_count_range
    return np.clip(
        np.rint(np.exp(log_count)),
        lower,
        upper,
    ).astype(np.int64)


def generate_identified_heterogeneous_dataset(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticBenchmarkDataset:
    """Generate the identified scenario with treatment-dependent counts."""

    design = generate_identified_design(
        contract,
        sample_size,
        replicate_index,
    )
    counts = generate_treatment_dependent_particle_counts(
        contract,
        design.treatment_values,
        replicate_index,
    )
    return _sample_identified_dataset(
        contract=contract,
        scenario_id="identified_heterogeneous_particles",
        design=design,
        sample_size=sample_size,
        replicate_index=replicate_index,
        particle_counts=counts,
    )


def generate_sequence_aligned_drift_dataset(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticBenchmarkDataset:
    """Generate current-aligned order with a known opposing temperature drift."""

    design = generate_sequence_aligned_design(
        contract,
        sample_size,
        replicate_index,
    )
    current_index = contract.treatment_names.index("current_norm")
    gap = _sequence_gap_summary(
        design.treatment_values,
        design.is_factorial_anchor,
        current_index,
    )
    target_truth = next(
        truth
        for truth in benchmark_quantile_truths(contract)
        if truth.treatment_name == "current_norm"
        and truth.outcome == "temperature_c"
    )
    true_median_effect = float(
        target_truth.effect[len(target_truth.effect) // 2]
    )
    scenario = contract.scenarios[2]
    drift_contribution = (
        scenario.sequence_confounding_ratio * true_median_effect
    )
    drift_slope = drift_contribution / gap.median_gap
    temperature_offset = drift_slope * (
        design.execution_order - design.execution_order.mean()
    )
    counts = generate_treatment_dependent_particle_counts(
        contract,
        design.treatment_values,
        replicate_index,
    )
    return _sample_identified_dataset(
        contract=contract,
        scenario_id="sequence_aligned_drift",
        design=design,
        sample_size=sample_size,
        replicate_index=replicate_index,
        particle_counts=counts,
        outcome_location_offsets={"temperature_c": temperature_offset},
        scenario_parameters={
            "matched_sequence_gap": gap.median_gap,
            "normalized_sequence_gap": gap.normalized_median_gap,
            "true_median_effect": true_median_effect,
            "drift_contribution_at_matched_gap": drift_contribution,
            "drift_slope_per_run": drift_slope,
            "total_sequence_drift": drift_slope * (sample_size - 1),
        },
    )


def generate_module_sign_reversal_dataset(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticBenchmarkDataset:
    """Generate opposite powder-to-diameter effects in the two DOE modules."""

    design = generate_identified_design(
        contract,
        sample_size,
        replicate_index,
    )
    powder_index = contract.treatment_names.index("powder_norm")
    diameter_mechanism = next(
        mechanism
        for mechanism in contract.mechanisms
        if mechanism.outcome == "particle_diameter_um"
    )
    module_multiplier = np.asarray(
        [
            1.0 if module.startswith("DOE-1") else -1.0
            for module in design.doe_modules
        ]
    )
    powder_values = design.treatment_values[:, powder_index]
    powder_location = (
        diameter_mechanism.location_linear[powder_index] * powder_values
        + diameter_mechanism.location_cubic[powder_index]
        * powder_values ** 3
    )
    powder_log_scale = (
        diameter_mechanism.log_scale_linear[powder_index] * powder_values
    )
    counts = generate_treatment_dependent_particle_counts(
        contract,
        design.treatment_values,
        replicate_index,
    )
    aggregate_truths, module_truths = module_reversal_quantile_truths(
        contract
    )
    return _sample_identified_dataset(
        contract=contract,
        scenario_id="module_sign_reversal",
        design=design,
        sample_size=sample_size,
        replicate_index=replicate_index,
        particle_counts=counts,
        outcome_location_offsets={
            "particle_diameter_um": (
                module_multiplier - 1.0
            ) * powder_location
        },
        outcome_log_scale_offsets={
            "particle_diameter_um": (
                module_multiplier - 1.0
            ) * powder_log_scale
        },
        scenario_parameters={
            "doe_1_effect_multiplier": 1.0,
            "doe_2_effect_multiplier": -1.0,
        },
        truths=aggregate_truths,
        module_truths=module_truths,
    )


def generate_insufficient_overlap_design(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticRunDesign:
    """Reassign four argon intervention strata to the reference arm."""

    base = generate_identified_design(contract, sample_size, replicate_index)
    values = base.treatment_values.copy()
    argon_index = contract.treatment_names.index("argon_norm")
    other_indices = [
        contract.treatment_names.index(name)
        for name in ("current_norm", "powder_norm", "distance_norm")
    ]
    selected = (
        base.is_factorial_anchor
        & (values[:, argon_index] == 1.0)
        & (np.sum(values[:, other_indices] == 1.0, axis=1) <= 1)
    )
    values[selected, argon_index] = -1.0
    return SyntheticRunDesign(
        treatment_values=values,
        doe_modules=base.doe_modules,
        execution_order=base.execution_order.copy(),
        is_factorial_anchor=base.is_factorial_anchor.copy(),
    )


def generate_insufficient_overlap_dataset(
    contract: SyntheticBenchmarkContract,
    sample_size: int,
    replicate_index: int = 0,
) -> SyntheticBenchmarkDataset:
    """Generate the four-stratum argon overlap-failure scenario."""

    design = generate_insufficient_overlap_design(
        contract,
        sample_size,
        replicate_index,
    )
    counts = generate_treatment_dependent_particle_counts(
        contract,
        design.treatment_values,
        replicate_index,
    )
    repetitions = (2 * sample_size // 3) // 16
    return _sample_identified_dataset(
        contract=contract,
        scenario_id="insufficient_overlap",
        design=design,
        sample_size=sample_size,
        replicate_index=replicate_index,
        particle_counts=counts,
        scenario_parameters={
            "matched_strata_retained": 4.0,
            "reassigned_argon_intervention_runs": float(4 * repetitions),
        },
        truths=design_matched_quantile_truths(
            contract,
            design.treatment_values,
        ),
    )


def _sample_identified_dataset(
    contract: SyntheticBenchmarkContract,
    scenario_id: str,
    design: SyntheticRunDesign,
    sample_size: int,
    replicate_index: int,
    particle_counts: NDArray[np.int64],
    outcome_location_offsets: Mapping[str, FloatArray] | None = None,
    outcome_log_scale_offsets: Mapping[str, FloatArray] | None = None,
    scenario_parameters: Mapping[str, float] | None = None,
    truths: tuple[SyntheticQuantileEffectTruth, ...] | None = None,
    module_truths: tuple[SyntheticModuleEffectTruth, ...] = (),
) -> SyntheticBenchmarkDataset:
    """Sample the common outcome law at a supplied precision pattern."""

    particle_samples: dict[str, tuple[FloatArray, ...]] = {}
    for mechanism_index, mechanism in enumerate(contract.mechanisms):
        values = design.treatment_values
        location = (
            mechanism.baseline_location
            + values @ np.asarray(mechanism.location_linear)
            + values ** 3 @ np.asarray(mechanism.location_cubic)
        )
        if outcome_location_offsets and mechanism.outcome in outcome_location_offsets:
            location = location + np.asarray(
                outcome_location_offsets[mechanism.outcome],
                dtype=float,
            )
        log_scale = values @ np.asarray(mechanism.log_scale_linear)
        if (
            outcome_log_scale_offsets
            and mechanism.outcome in outcome_log_scale_offsets
        ):
            log_scale = log_scale + np.asarray(
                outcome_log_scale_offsets[mechanism.outcome],
                dtype=float,
            )
        scale = mechanism.baseline_scale * np.exp(log_scale)
        outcome_samples = []
        for run_index in range(sample_size):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        contract.random_seed,
                        sample_size,
                        replicate_index,
                        1402,
                        mechanism_index,
                        run_index,
                    ]
                )
            )
            run_shift = rng.normal(0.0, mechanism.run_location_sd)
            outcome_samples.append(
                location[run_index]
                + run_shift
                + scale[run_index]
                * rng.normal(size=int(particle_counts[run_index]))
            )
        particle_samples[mechanism.outcome] = tuple(outcome_samples)

    scenario_label = scenario_id.removeprefix("identified_").replace(
        "_particles", ""
    )
    run_ids = tuple(
        (
            f"syn-{scenario_label}-n{sample_size:03d}-"
            f"r{replicate_index:04d}-"
            f"run{index:03d}"
        )
        for index in range(1, sample_size + 1)
    )
    runs = RunBatch(
        run_ids=run_ids,
        groups=("synthetic",) * sample_size,
        doe_modules=design.doe_modules,
        treatment_names=contract.treatment_names,
        treatment_values=design.treatment_values,
        controlled_process_names=(
            "hydrogen_setting",
            "powder_carrier_gas_setting",
        ),
        controlled_process_values=np.tile([2.5, 10.0], (sample_size, 1)),
        context_names=("execution_order", "measurement_position_mm"),
        context_values=np.column_stack(
            (
                design.execution_order,
                np.full(sample_size, 100.0),
            )
        ),
        particle_samples=particle_samples,
    )
    return SyntheticBenchmarkDataset(
        scenario_id=scenario_id,
        sample_size=sample_size,
        replicate_index=replicate_index,
        particle_counts=np.asarray(particle_counts, dtype=np.int64),
        scenario_parameters=(
            {} if scenario_parameters is None else dict(scenario_parameters)
        ),
        runs=runs,
        truths=(benchmark_quantile_truths(contract) if truths is None else truths),
        module_truths=module_truths,
    )
