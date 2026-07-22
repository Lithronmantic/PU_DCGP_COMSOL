
from dataclasses import dataclass

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import SyntheticBenchmarkDataset
from .benchmark_methods import BenchmarkMethodResult


@dataclass(frozen=True, slots=True)
class BenchmarkSelectionMetrics:

    method_name: str
    scenario_id: str
    active_admission_rate: float
    null_false_admission_rate: float
    target_unsupported_admitted: bool | None
    admitted_count: int


def evaluate_benchmark_selection(
    result: BenchmarkMethodResult,
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
) -> BenchmarkSelectionMetrics:

    truth_map = {
        (truth.estimand_id, truth.outcome): truth for truth in dataset.truths
    }
    gated = result.method_name == "support_gated_pu_dcgp"
    admitted = np.asarray(
        [
            effect.reported
            if gated
            else (
                np.all(effect.lower_bound > 0.0)
                or np.all(effect.upper_bound < 0.0)
            )
            for effect in result.effects
        ],
        dtype=bool,
    )
    active = np.asarray(
        [
            truth_map[(effect.estimand_id, effect.outcome)].is_active
            for effect in result.effects
        ],
        dtype=bool,
    )
    scenario = next(
        scenario
        for scenario in contract.scenarios
        if scenario.scenario_id == dataset.scenario_id
    )
    target_admitted = None
    if scenario.target_treatment is not None:
        target_index = next(
            index
            for index, effect in enumerate(result.effects)
            if effect.treatment_name == scenario.target_treatment
            and effect.outcome == scenario.target_outcome
        )
        target_admitted = bool(admitted[target_index])
    return BenchmarkSelectionMetrics(
        method_name=result.method_name,
        scenario_id=result.scenario_id,
        active_admission_rate=float(np.mean(admitted[active])),
        null_false_admission_rate=float(np.mean(admitted[~active])),
        target_unsupported_admitted=target_admitted,
        admitted_count=int(np.sum(admitted)),
    )
