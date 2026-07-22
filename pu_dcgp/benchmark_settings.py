"""Shared pilot and formal settings for benchmark execution."""

from dataclasses import dataclass

from .benchmark_contract import SyntheticBenchmarkContract
from .config import PUDCGPConfig


@dataclass(frozen=True, slots=True)
class FormalBenchmarkPlan:
    """Frozen execution axes and expected checkpoint size."""

    scenario_ids: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    replicate_indices: tuple[int, ...]
    dataset_count: int
    method_record_count: int
    gp_hyperparameter_combinations: int


def benchmark_pilot_config(
    contract: SyntheticBenchmarkContract,
) -> PUDCGPConfig:
    """Return the fixed lightweight grid used only before formal execution."""

    return PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        bootstrap_replicates=40,
        posterior_band_draws=10000,
        benchmark_cv_folds=3,
        gp_lengthscale_candidates=(0.5, 1.0, 2.0),
        gp_context_lengthscale_candidates=(8.0,),
        gp_signal_variance_candidates=(1.0,),
        gp_noise_variance_candidates=(0.05, 0.2, 0.5),
        random_seed=contract.random_seed,
    )


def benchmark_formal_config(
    contract: SyntheticBenchmarkContract,
) -> PUDCGPConfig:
    """Return the full frozen configuration used for formal comparison."""

    return PUDCGPConfig(
        treatment_columns=contract.treatment_names,
        outcome_columns=contract.outcome_names,
        quantile_grid=contract.quantile_grid,
        bootstrap_replicates=200,
        posterior_band_draws=10000,
        benchmark_cv_folds=5,
        random_seed=contract.random_seed,
    )


def formal_benchmark_plan(
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
) -> FormalBenchmarkPlan:
    """Derive the complete formal axes from the frozen contract."""

    scenario_ids = tuple(scenario.scenario_id for scenario in contract.scenarios)
    sample_sizes = contract.scenarios[0].sample_sizes
    if any(scenario.sample_sizes != sample_sizes for scenario in contract.scenarios):
        raise ValueError("Formal scenarios do not share one sample-size axis")
    replicate_indices = tuple(
        range(
            contract.pilot_replicate_count,
            contract.pilot_replicate_count + contract.replicate_count,
        )
    )
    dataset_count = len(scenario_ids) * len(sample_sizes) * len(replicate_indices)
    gp_combinations = (
        len(config.gp_lengthscale_candidates)
        * len(config.gp_context_lengthscale_candidates)
        * len(config.gp_signal_variance_candidates)
        * len(config.gp_noise_variance_candidates)
    )
    return FormalBenchmarkPlan(
        scenario_ids=scenario_ids,
        sample_sizes=sample_sizes,
        replicate_indices=replicate_indices,
        dataset_count=dataset_count,
        method_record_count=dataset_count * len(contract.methods),
        gp_hyperparameter_combinations=gp_combinations,
    )
