"""Frozen scope for the first PU-DCGP implementation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PUDCGPConfig:
    """Scientific scope shared by the code and the manuscript."""

    treatment_columns: tuple[str, ...] = (
        "current_a",
        "argon_flow_scfh",
        "powder_feed_g_min",
        "spray_distance_mm",
    )
    controlled_process_columns: tuple[str, ...] = (
        "hydrogen_setting",
        "powder_carrier_gas_setting",
    )
    controlled_process_values: tuple[float, ...] = (2.5, 10.0)
    derived_process_columns: tuple[str, ...] = (
        "hydrogen_to_argon_ratio",
    )
    context_columns: tuple[str, ...] = (
        "execution_order",
        "measurement_position_mm",
    )
    outcome_columns: tuple[str, ...] = (
        "temperature_c",
        "velocity_m_s",
        "particle_diameter_um",
    )
    confirmatory_treatments: tuple[str, ...] = (
        "current_a",
        "argon_flow_scfh",
        "powder_feed_g_min",
    )
    exploratory_treatments: tuple[str, ...] = ("spray_distance_mm",)
    primary_group: str = "A"
    analysis_groups: tuple[str, ...] = ("A",)
    quantile_grid: tuple[float, ...] = tuple(index / 20 for index in range(1, 20))
    fpca_variance_target: float = 0.995
    fpca_min_components: int = 2
    bootstrap_replicates: int = 200
    effect_bootstrap_replicates: int = 2000
    effect_interval_level: float = 0.95
    posterior_band_draws: int = 10000
    benchmark_cv_folds: int = 5
    gp_lengthscale_candidates: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    gp_context_lengthscale_candidates: tuple[float, ...] = (1.0, 8.0, 64.0)
    gp_signal_variance_candidates: tuple[float, ...] = (1.0,)
    gp_noise_variance_candidates: tuple[float, ...] = (0.05, 0.2, 0.5, 1.0)
    random_seed: int = 2026
