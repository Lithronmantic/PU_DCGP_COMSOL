"""Single-replicate contract audit for the benchmark point-method adapters."""

from dataclasses import dataclass

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import generate_identified_balanced_dataset
from .benchmark_methods import fit_benchmark_point_effect_methods
from .config import PUDCGPConfig


@dataclass(frozen=True, slots=True)
class PointMethodPilotEntry:
    """Descriptive error and runtime from one point-method smoke run."""

    method_name: str
    effect_count: int
    median_normalized_irmse: float
    maximum_normalized_irmse: float
    median_shape_effect_normalized_irmse: float
    powder_diameter_curve_range: float
    preparation_seconds: float
    fit_seconds: float
    prediction_seconds: float
    intervals_available: bool


@dataclass(frozen=True, slots=True)
class PointMethodPilotAudit:
    """Adapter integrity result; it is not a method-performance result."""

    scenario_id: str
    sample_size: int
    replicate_index: int
    expected_effect_count: int
    entries: tuple[PointMethodPilotEntry, ...]
    aligned_effect_count: int
    comparison_authorized: bool
    passed: bool


def audit_point_method_adapters(
    contract: SyntheticBenchmarkContract,
) -> PointMethodPilotAudit:
    """Run the frozen 48-run smoke case and audit only interface integrity."""

    dataset = generate_identified_balanced_dataset(contract, 48, 0)
    config = PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        bootstrap_replicates=40,
        gp_lengthscale_candidates=(0.5, 1.0, 2.0),
        gp_context_lengthscale_candidates=(8.0,),
        gp_signal_variance_candidates=(1.0,),
        gp_noise_variance_candidates=(0.05, 0.2, 0.5),
        random_seed=contract.random_seed,
    )
    results = fit_benchmark_point_effect_methods(dataset, contract, config)
    truth_map = {
        (truth.estimand_id, truth.outcome): truth for truth in dataset.truths
    }
    outcome_scales = {
        mechanism.outcome: mechanism.baseline_scale
        for mechanism in contract.mechanisms
    }
    entries = []
    for result in results:
        errors = []
        shape_errors = []
        for effect in result.effects:
            truth = truth_map[(effect.estimand_id, effect.outcome)]
            error = float(
                np.sqrt(np.mean(np.square(effect.point_effect - truth.effect)))
                / outcome_scales[effect.outcome]
            )
            errors.append(error)
            if np.ptp(truth.effect) > 1e-12:
                shape_errors.append(error)
        powder_diameter = next(
            effect
            for effect in result.effects
            if effect.treatment_name == "powder_norm"
            and effect.outcome == "particle_diameter_um"
        )
        entries.append(
            PointMethodPilotEntry(
                method_name=result.method_name,
                effect_count=len(result.effects),
                median_normalized_irmse=float(np.median(errors)),
                maximum_normalized_irmse=float(max(errors)),
                median_shape_effect_normalized_irmse=float(
                    np.median(shape_errors)
                ),
                powder_diameter_curve_range=float(
                    np.ptp(powder_diameter.point_effect)
                ),
                preparation_seconds=result.preparation_seconds,
                fit_seconds=result.fit_seconds,
                prediction_seconds=result.prediction_seconds,
                intervals_available=all(
                    effect.lower_bound is not None
                    and effect.upper_bound is not None
                    for effect in result.effects
                ),
            )
        )

    expected_effect_count = len(dataset.truths)
    aligned_effect_count = min(entry.effect_count for entry in entries)
    passed = (
        tuple(entry.method_name for entry in entries)
        == ("mean_gp", "distribution_gp_no_pu", "pu_dcgp")
        and all(entry.effect_count == expected_effect_count for entry in entries)
        and all(entry.intervals_available for entry in entries)
        and entries[0].powder_diameter_curve_range < 1e-12
        and all(
            entry.powder_diameter_curve_range > 1.0 for entry in entries[1:]
        )
    )
    return PointMethodPilotAudit(
        scenario_id=dataset.scenario_id,
        sample_size=dataset.sample_size,
        replicate_index=dataset.replicate_index,
        expected_effect_count=expected_effect_count,
        entries=tuple(entries),
        aligned_effect_count=aligned_effect_count,
        comparison_authorized=False,
        passed=passed,
    )
