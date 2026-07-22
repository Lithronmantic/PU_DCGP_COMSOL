
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticOutcomeMechanism:

    outcome: str
    baseline_location: float
    baseline_scale: float
    run_location_sd: float
    location_linear: tuple[float, ...]
    location_cubic: tuple[float, ...]
    log_scale_linear: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:

    scenario_id: str
    purpose: str
    sample_sizes: tuple[int, ...]
    particle_count_mode: str
    particle_count_range: tuple[int, int]
    overlap_mode: str
    sequence_mode: str
    module_mode: str
    target_treatment: str | None
    target_outcome: str | None
    sequence_confounding_ratio: float
    module_effect_multipliers: tuple[float, float]
    matched_strata_retained: int | None
    expected_active_gate_behavior: str


@dataclass(frozen=True, slots=True)
class BenchmarkMetric:

    metric_id: str
    target: str
    aggregation: str


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkContract:

    random_seed: int
    pilot_replicate_count: int
    replicate_count: int
    interval_level: float
    treatment_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    quantile_grid: tuple[float, ...]
    methods: tuple[str, ...]
    mechanisms: tuple[SyntheticOutcomeMechanism, ...]
    scenarios: tuple[BenchmarkScenario, ...]
    metrics: tuple[BenchmarkMetric, ...]
    hypotheses: tuple[str, ...]


def pu_dcgp_benchmark_contract() -> SyntheticBenchmarkContract:

    treatment_names = (
        "current_norm",
        "argon_norm",
        "powder_norm",
        "distance_norm",
    )
    outcome_names = (
        "temperature_c",
        "velocity_m_s",
        "particle_diameter_um",
    )
    mechanisms = (
        SyntheticOutcomeMechanism(
            outcome="temperature_c",
            baseline_location=1800.0,
            baseline_scale=60.0,
            run_location_sd=20.0,
            location_linear=(18.0, 5.0, -7.0, 0.0),
            location_cubic=(4.0, 0.0, -2.0, 0.0),
            log_scale_linear=(0.12, 0.0, 0.08, 0.0),
        ),
        SyntheticOutcomeMechanism(
            outcome="velocity_m_s",
            baseline_location=160.0,
            baseline_scale=12.0,
            run_location_sd=4.0,
            location_linear=(2.0, 6.0, 0.0, -3.0),
            location_cubic=(0.0, 1.5, 0.0, 0.0),
            log_scale_linear=(0.0, 0.12, 0.0, 0.0),
        ),
        SyntheticOutcomeMechanism(
            outcome="particle_diameter_um",
            baseline_location=40.0,
            baseline_scale=5.0,
            run_location_sd=1.5,
            location_linear=(0.0, 0.0, -2.0, 0.0),
            location_cubic=(0.0, 0.0, -0.5, 0.0),
            log_scale_linear=(0.0, 0.0, -0.15, 0.0),
        ),
    )
    common_sizes = (48, 96, 144)
    scenarios = (
        BenchmarkScenario(
            scenario_id="identified_balanced_particles",
            purpose="recover known location and shape effects under clean support",
            sample_sizes=common_sizes,
            particle_count_mode="balanced",
            particle_count_range=(80, 80),
            overlap_mode="factorial_anchors_plus_interior",
            sequence_mode="randomized",
            module_mode="homogeneous",
            target_treatment=None,
            target_outcome=None,
            sequence_confounding_ratio=0.0,
            module_effect_multipliers=(1.0, 1.0),
            matched_strata_retained=None,
            expected_active_gate_behavior="admit_active",
        ),
        BenchmarkScenario(
            scenario_id="identified_heterogeneous_particles",
            purpose="test particle-uncertainty calibration at unequal precision",
            sample_sizes=common_sizes,
            particle_count_mode="treatment_dependent_heterogeneous",
            particle_count_range=(20, 240),
            overlap_mode="factorial_anchors_plus_interior",
            sequence_mode="randomized",
            module_mode="homogeneous",
            target_treatment=None,
            target_outcome=None,
            sequence_confounding_ratio=0.0,
            module_effect_multipliers=(1.0, 1.0),
            matched_strata_retained=None,
            expected_active_gate_behavior="admit_active",
        ),
        BenchmarkScenario(
            scenario_id="sequence_aligned_drift",
            purpose="reject an apparent effect created or reversed by run order",
            sample_sizes=common_sizes,
            particle_count_mode="heterogeneous",
            particle_count_range=(20, 240),
            overlap_mode="factorial_anchors_plus_interior",
            sequence_mode="intervention_later_with_opposing_drift",
            module_mode="homogeneous",
            target_treatment="current_norm",
            target_outcome="temperature_c",
            sequence_confounding_ratio=-1.5,
            module_effect_multipliers=(1.0, 1.0),
            matched_strata_retained=None,
            expected_active_gate_behavior="abstain_sequence_instability",
        ),
        BenchmarkScenario(
            scenario_id="module_sign_reversal",
            purpose="reject a pooled effect that changes sign across modules",
            sample_sizes=common_sizes,
            particle_count_mode="heterogeneous",
            particle_count_range=(20, 240),
            overlap_mode="factorial_anchors_plus_interior",
            sequence_mode="randomized_within_module",
            module_mode="opposite_effect_direction",
            target_treatment="powder_norm",
            target_outcome="particle_diameter_um",
            sequence_confounding_ratio=0.0,
            module_effect_multipliers=(1.0, -1.0),
            matched_strata_retained=None,
            expected_active_gate_behavior="abstain_module_inconsistency",
        ),
        BenchmarkScenario(
            scenario_id="insufficient_overlap",
            purpose="withhold effects when exact reference support is removed",
            sample_sizes=common_sizes,
            particle_count_mode="heterogeneous",
            particle_count_range=(20, 240),
            overlap_mode="missing_matched_arms",
            sequence_mode="randomized",
            module_mode="homogeneous",
            target_treatment="argon_norm",
            target_outcome="velocity_m_s",
            sequence_confounding_ratio=0.0,
            module_effect_multipliers=(1.0, 1.0),
            matched_strata_retained=4,
            expected_active_gate_behavior="insufficient_support",
        ),
    )
    metrics = (
        BenchmarkMetric(
            metric_id="mean_prediction_rmse",
            target="held-out-setting run means",
            aggregation="replicate mean by scenario, method, and sample size",
        ),
        BenchmarkMetric(
            metric_id="wasserstein_prediction_rmse",
            target="held-out-setting particle distributions",
            aggregation="replicate mean by scenario, method, and sample size",
        ),
        BenchmarkMetric(
            metric_id="quantile_effect_irmse",
            target="known intervention-minus-reference quantile-effect curve",
            aggregation="integrated RMSE then replicate median",
        ),
        BenchmarkMetric(
            metric_id="simultaneous_band_coverage",
            target="known full quantile-effect curve",
            aggregation="coverage proportion and absolute error from 0.95",
        ),
        BenchmarkMetric(
            metric_id="active_admission_rate",
            target="nonzero supported treatment-outcome effects",
            aggregation="proportion by identified scenario and sample size",
        ),
        BenchmarkMetric(
            metric_id="null_false_admission_rate",
            target="zero treatment-outcome effects",
            aggregation="proportion by scenario, method, and sample size",
        ),
        BenchmarkMetric(
            metric_id="unsupported_false_admission_rate",
            target="effects in sequence, module, or overlap failure regimes",
            aggregation="proportion by failure regime and sample size",
        ),
    )
    hypotheses = (
        "H1: on active nonconstant quantile effects, a distribution model must "
        "reduce median quantile-effect IRMSE by at least 10% versus the mean GP",
        "H2: under heterogeneous particle counts, PU-DCGP must reduce absolute "
        "0.95 simultaneous-coverage error by at least 0.02 versus the no-PU "
        "distribution GP, without worsening balanced-count error by more than 0.02",
        "H3: gating must keep unsupported false admission at or below 0.05 and "
        "reduce it by at least 50% versus ungated PU-DCGP",
        "H4: at 144 supported runs, gated PU-DCGP must retain at least 0.80 "
        "admission power for active effects while keeping null false admission "
        "at or below 0.05",
    )
    return SyntheticBenchmarkContract(
        random_seed=2026,
        pilot_replicate_count=20,
        replicate_count=200,
        interval_level=0.95,
        treatment_names=treatment_names,
        outcome_names=outcome_names,
        quantile_grid=tuple(index / 20 for index in range(1, 20)),
        methods=(
            "mean_gp",
            "distribution_gp_no_pu",
            "pu_dcgp",
            "support_gated_pu_dcgp",
        ),
        mechanisms=mechanisms,
        scenarios=scenarios,
        metrics=metrics,
        hypotheses=hypotheses,
    )
