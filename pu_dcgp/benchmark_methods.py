
from dataclasses import dataclass, replace
from time import perf_counter

from numpy.typing import NDArray
import numpy as np

from .admission_gate import EffectAdmissionDecision
from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import SyntheticBenchmarkDataset
from .config import PUDCGPConfig
from .contrast_uncertainty import average_paired_quantile_contrast
from .contracts import PreparedData, RunBatch
from .distribution_encoder import BootstrapWassersteinFPCAEncoder
from .gaussian_process import (
    GaussianProcessDistributionModel,
    GaussianProcessMeanModel,
)
from .simultaneous_band import gaussian_simultaneous_band


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BenchmarkEffectEstimate:

    estimand_id: str
    treatment_name: str
    outcome: str
    quantile_grid: FloatArray
    point_effect: FloatArray
    marginal_variance: FloatArray
    effect_covariance: FloatArray
    lower_bound: FloatArray | None
    upper_bound: FloatArray | None
    interval_kind: str
    admission_status: str
    reported: bool
    failed_gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkMethodResult:

    method_name: str
    scenario_id: str
    sample_size: int
    replicate_index: int
    effects: tuple[BenchmarkEffectEstimate, ...]
    preparation_seconds: float
    fit_seconds: float
    prediction_seconds: float


def fit_benchmark_point_effect_methods(
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
) -> tuple[BenchmarkMethodResult, ...]:

    mean_result = _fit_mean_gp_effects(dataset, contract, config)

    preparation_start = perf_counter()
    encoder = BootstrapWassersteinFPCAEncoder(config)
    representation = encoder.fit_transform(dataset.runs)
    preparation_seconds = perf_counter() - preparation_start
    prepared = PreparedData(runs=dataset.runs, distributions=representation)
    distribution_results = tuple(
        _fit_distribution_gp_effects(
            dataset,
            contract,
            config,
            encoder,
            prepared,
            method_name,
            use_particle_uncertainty,
            preparation_seconds,
        )
        for method_name, use_particle_uncertainty in (
            ("distribution_gp_no_pu", False),
            ("pu_dcgp", True),
        )
    )
    return (mean_result,) + distribution_results


def apply_admission_decisions(
    pu_result: BenchmarkMethodResult,
    decisions: tuple[EffectAdmissionDecision, ...],
) -> BenchmarkMethodResult:

    decision_map = {
        (decision.estimand.estimand_id, decision.outcome): decision
        for decision in decisions
    }
    admitted_statuses = {
        "admit",
        "conditional_admit",
        "exploratory_admit",
    }
    effects = []
    for effect in pu_result.effects:
        decision = decision_map[(effect.estimand_id, effect.outcome)]
        effects.append(
            replace(
                effect,
                admission_status=decision.status,
                reported=decision.status in admitted_statuses,
                failed_gates=decision.failed_gates,
            )
        )
    return replace(
        pu_result,
        method_name="support_gated_pu_dcgp",
        effects=tuple(effects),
    )


def _fit_mean_gp_effects(
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
) -> BenchmarkMethodResult:
    fit_start = perf_counter()
    model = GaussianProcessMeanModel(config)
    model.fit(dataset.runs)
    fit_seconds = perf_counter() - fit_start
    prediction_start = perf_counter()
    estimates = {}
    for treatment_name in contract.treatment_names:
        reference, intervention, contexts = _matched_contrast_points(
            dataset.runs,
            treatment_name,
        )
        joint_mean, joint_covariance = model.predict_joint(
            np.vstack([reference, intervention]),
            np.vstack([contexts, contexts]),
        )
        stratum_count = len(reference)
        contrast_weights = np.concatenate(
            [
                np.full(stratum_count, -1.0 / stratum_count),
                np.full(stratum_count, 1.0 / stratum_count),
            ]
        )
        estimates[treatment_name] = (
            contrast_weights @ joint_mean,
            np.asarray(
                [
                    contrast_weights
                    @ outcome_covariance
                    @ contrast_weights
                    for outcome_covariance in joint_covariance
                ]
            ),
        )
    outcome_index = {
        outcome: index for index, outcome in enumerate(model.outcome_names)
    }
    grid = np.asarray(contract.quantile_grid, dtype=float)
    effects = tuple(
        _effect_estimate(
            truth.estimand_id,
            truth.treatment_name,
            truth.outcome,
            grid,
            np.full(
                len(grid),
                estimates[truth.treatment_name][0][outcome_index[truth.outcome]],
            ),
            np.full(
                len(grid),
                estimates[truth.treatment_name][1][outcome_index[truth.outcome]],
            ),
            np.full(
                (len(grid), len(grid)),
                estimates[truth.treatment_name][1][outcome_index[truth.outcome]],
            ),
            config.effect_interval_level,
            config.posterior_band_draws,
            config.random_seed,
        )
        for truth in dataset.truths
    )
    prediction_seconds = perf_counter() - prediction_start
    return BenchmarkMethodResult(
        method_name="mean_gp",
        scenario_id=dataset.scenario_id,
        sample_size=dataset.sample_size,
        replicate_index=dataset.replicate_index,
        effects=effects,
        preparation_seconds=0.0,
        fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds,
    )


def _fit_distribution_gp_effects(
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
    encoder: BootstrapWassersteinFPCAEncoder,
    prepared: PreparedData,
    method_name: str,
    use_particle_uncertainty: bool,
    preparation_seconds: float,
) -> BenchmarkMethodResult:
    fit_start = perf_counter()
    model = GaussianProcessDistributionModel(
        config,
        use_particle_uncertainty=use_particle_uncertainty,
    )
    model.fit(prepared)
    fit_seconds = perf_counter() - fit_start
    prediction_start = perf_counter()
    estimates = {}
    for treatment_name in contract.treatment_names:
        reference, intervention, contexts = _matched_contrast_points(
            dataset.runs,
            treatment_name,
        )
        joint_distribution = encoder.inverse_transform_joint(
            model.predict_joint(
                np.vstack([reference, intervention]),
                np.vstack([contexts, contexts]),
            )
        )
        stratum_count = len(reference)
        estimates[treatment_name] = {
            outcome: average_paired_quantile_contrast(
                joint_distribution,
                outcome,
                stratum_count,
            )
            for outcome in contract.outcome_names
        }
    effects = tuple(
        _effect_estimate(
            truth.estimand_id,
            truth.treatment_name,
            truth.outcome,
            np.asarray(contract.quantile_grid, dtype=float),
            estimates[truth.treatment_name][truth.outcome].point_effect,
            estimates[truth.treatment_name][truth.outcome].marginal_variance,
            estimates[truth.treatment_name][truth.outcome].covariance,
            config.effect_interval_level,
            config.posterior_band_draws,
            config.random_seed,
        )
        for truth in dataset.truths
    )
    prediction_seconds = perf_counter() - prediction_start
    return BenchmarkMethodResult(
        method_name=method_name,
        scenario_id=dataset.scenario_id,
        sample_size=dataset.sample_size,
        replicate_index=dataset.replicate_index,
        effects=effects,
        preparation_seconds=preparation_seconds,
        fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds,
    )


def _effect_estimate(
    estimand_id: str,
    treatment_name: str,
    outcome: str,
    quantile_grid: FloatArray,
    point_effect: FloatArray,
    marginal_variance: FloatArray,
    effect_covariance: FloatArray,
    interval_level: float,
    band_draw_count: int,
    random_seed: int,
) -> BenchmarkEffectEstimate:
    band = gaussian_simultaneous_band(
        point_effect,
        effect_covariance,
        interval_level,
        band_draw_count,
        random_seed,
    )
    return BenchmarkEffectEstimate(
        estimand_id=estimand_id,
        treatment_name=treatment_name,
        outcome=outcome,
        quantile_grid=np.asarray(quantile_grid, dtype=float),
        point_effect=np.asarray(point_effect, dtype=float),
        marginal_variance=np.asarray(marginal_variance, dtype=float),
        effect_covariance=np.asarray(effect_covariance, dtype=float),
        lower_bound=band.lower_bound,
        upper_bound=band.upper_bound,
        interval_kind="gaussian_posterior_simultaneous_max_t",
        admission_status="not_applicable",
        reported=True,
        failed_gates=(),
    )


def _matched_contrast_points(
    runs: RunBatch,
    treatment_name: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    treatment_index = runs.treatment_names.index(treatment_name)
    other_indices = [
        index for index in range(len(runs.treatment_names))
        if index != treatment_index
    ]
    grouped: dict[tuple[float, ...], list[int]] = {}
    for row_index, row in enumerate(runs.treatment_values):
        grouped.setdefault(tuple(row[other_indices]), []).append(row_index)
    reference_points = []
    intervention_points = []
    for rows in grouped.values():
        rows_array = np.asarray(rows, dtype=int)
        arm_values = runs.treatment_values[rows_array, treatment_index]
        if not np.any(arm_values == -1.0) or not np.any(arm_values == 1.0):
            continue
        point = runs.treatment_values[rows_array[0]].copy()
        reference = point.copy()
        intervention = point.copy()
        reference[treatment_index] = -1.0
        intervention[treatment_index] = 1.0
        reference_points.append(reference)
        intervention_points.append(intervention)
    reference_array = np.asarray(reference_points, dtype=float)
    intervention_array = np.asarray(intervention_points, dtype=float)
    common_context = np.mean(runs.context_values, axis=0)
    contexts = np.tile(common_context, (len(reference_array), 1))
    return reference_array, intervention_array, contexts
