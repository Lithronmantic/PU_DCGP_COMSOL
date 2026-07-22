
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET_CONTRACT_PATH = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)
OUT_DIR = HERE / "h11_outputs" / "effective_exit_parameterization"
CONTRACT_PATH = OUT_DIR / "h11_effective_exit_parameterization_contract.json"


@dataclass(frozen=True)
class AGroupDOEPoint:

    current_a: float
    argon_scfh: float
    spray_distance_mm: float
    powder_feed_g_min: float

    def validate(self, contract: "EffectiveExitParameterizationContract") -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("DOE settings must be finite")
        bounds = contract.factor_bounds()
        for name, value in values.items():
            lower, upper = bounds[name]
            if not lower <= float(value) <= upper:
                raise ValueError(
                    f"{name}={value} lies outside executed A range [{lower}, {upper}]"
                )


@dataclass(frozen=True)
class EffectiveExitParameterizationContract:

    current_min_a: float = 600.0
    current_max_a: float = 800.0
    argon_min_scfh: float = 80.0
    argon_max_scfh: float = 120.0
    spray_distance_min_mm: float = 80.0
    spray_distance_max_mm: float = 120.0
    powder_feed_min_g_min: float = 10.0
    powder_feed_max_g_min: float = 30.0
    hydrogen_setting: float = 2.5
    dpv_plane_mm: float = 100.0
    workpiece_present_in_dpv_branch: bool = False
    exit_main_effects_only_in_first_layer: bool = True
    current_to_effective_exit: bool = True
    argon_to_effective_exit: bool = True
    powder_feed_to_effective_exit: bool = False
    spray_distance_to_effective_exit: bool = False
    powder_feed_to_particle_loading: bool = True
    spray_distance_retained_in_discrepancy_layer: bool = True
    h2_argon_ratio_as_independent_feature: bool = False
    grouped_held_out_by_four_factor_setting: bool = True
    all_a_runs_retained: bool = True
    particle_operator_available: bool = False
    dpv_operator_available: bool = False
    jointly_calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        expected_bounds = (
            (self.current_min_a, self.current_max_a, 600.0, 800.0),
            (self.argon_min_scfh, self.argon_max_scfh, 80.0, 120.0),
            (
                self.spray_distance_min_mm,
                self.spray_distance_max_mm,
                80.0,
                120.0,
            ),
            (
                self.powder_feed_min_g_min,
                self.powder_feed_max_g_min,
                10.0,
                30.0,
            ),
        )
        if any((lower, upper) != (expected_lower, expected_upper) for (
            lower,
            upper,
            expected_lower,
            expected_upper,
        ) in expected_bounds):
            raise ValueError("Executed A factor ranges cannot be changed")
        if self.hydrogen_setting != 2.5:
            raise ValueError("The fixed hydrogen setting must remain 2.5")
        if self.dpv_plane_mm != 100.0:
            raise ValueError("The A-group DPV plane must remain fixed at 100 mm")
        if self.workpiece_present_in_dpv_branch:
            raise ValueError("The A-group DPV branch has no workpiece")
        if not self.exit_main_effects_only_in_first_layer:
            raise ValueError("The first exit layer is the prespecified main-effects basis")
        if not (self.current_to_effective_exit and self.argon_to_effective_exit):
            raise ValueError("Current and argon are the admitted exit covariates")
        if self.powder_feed_to_effective_exit:
            raise ValueError("Powder feed enters loading, not the exit boundary")
        if self.spray_distance_to_effective_exit:
            raise ValueError("Spray distance is absent from fixed-DPV free-jet equations")
        if not self.powder_feed_to_particle_loading:
            raise ValueError("Powder feed must remain a particle-loading treatment")
        if not self.spray_distance_retained_in_discrepancy_layer:
            raise ValueError("The executed spray-distance treatment must remain in analysis")
        if self.h2_argon_ratio_as_independent_feature:
            raise ValueError("Fixed-H2/argon ratio duplicates the argon predictor")
        if not (
            self.grouped_held_out_by_four_factor_setting
            and self.all_a_runs_retained
        ):
            raise ValueError("Grouped held-out validation must retain all A runs")
        if self.jointly_calibrated or self.paper_prediction_allowed:
            raise ValueError("The parameterization contract cannot claim calibration")

    def factor_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "current_a": (self.current_min_a, self.current_max_a),
            "argon_scfh": (self.argon_min_scfh, self.argon_max_scfh),
            "spray_distance_mm": (
                self.spray_distance_min_mm,
                self.spray_distance_max_mm,
            ),
            "powder_feed_g_min": (
                self.powder_feed_min_g_min,
                self.powder_feed_max_g_min,
            ),
        }

    @property
    def calibration_release_allowed(self) -> bool:
        return self.particle_operator_available and self.dpv_operator_available

    def unresolved_gates(self) -> list[str]:
        gates: list[str] = []
        if not self.particle_operator_available:
            gates.append("validated_particle_operator")
        if not self.dpv_operator_available:
            gates.append("validated_dpv_observation_operator")
        gates.extend(
            [
                "training_fold_only_joint_calibration",
                "grouped_held_out_predictive_validation",
                "parameter_identifiability_and_prior_sensitivity",
            ]
        )
        return gates


def _standardize(value: float, lower: float, upper: float) -> float:
    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    return (value - midpoint) / half_range


def standardized_coordinates(
    point: AGroupDOEPoint,
    contract: EffectiveExitParameterizationContract | None = None,
) -> dict[str, float]:
    active = contract or EffectiveExitParameterizationContract()
    active.validate()
    point.validate(active)
    bounds = active.factor_bounds()
    values = asdict(point)
    return {
        name: _standardize(float(values[name]), *bounds[name])
        for name in bounds
    }


def pathway_design(
    point: AGroupDOEPoint,
    contract: EffectiveExitParameterizationContract | None = None,
) -> dict[str, dict[str, float]]:

    active = contract or EffectiveExitParameterizationContract()
    coordinates = standardized_coordinates(point, active)
    return {
        "effective_exit_temperature": {
            "intercept": 1.0,
            "current": coordinates["current_a"],
            "argon": coordinates["argon_scfh"],
        },
        "effective_exit_velocity": {
            "intercept": 1.0,
            "current": coordinates["current_a"],
            "argon": coordinates["argon_scfh"],
        },
        "particle_loading": {
            "intercept": 1.0,
            "powder_feed": coordinates["powder_feed_g_min"],
        },
        "empirical_discrepancy_context": {
            "intercept": 1.0,
            "spray_distance": coordinates["spray_distance_mm"],
        },
    }


def _target_counts() -> dict[str, int]:
    with TARGET_CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "runs": int(payload["n_runs"]),
        "four_factor_settings": int(payload["n_unique_four_factor_settings"]),
        "joint_particles": int(payload["joint_valid_particle_count"]),
    }


def contract_payload(
    contract: EffectiveExitParameterizationContract | None = None,
) -> dict[str, object]:
    active = contract or EffectiveExitParameterizationContract()
    active.validate()
    nominal = AGroupDOEPoint(700.0, 100.0, 100.0, 20.0)
    return {
        "schema_version": "h11_effective_exit_parameterization_contract_v1",
        "status": "pass_parameterization_skeleton_calibration_locked",
        "contract": asdict(active),
        "a_group_target_counts": _target_counts(),
        "standardized_coordinates": (
            "each executed factor is centered at its A-range midpoint and "
            "scaled to [-1,1]"
        ),
        "nominal_design": pathway_design(nominal, active),
        "pathways": {
            "current_and_argon": (
                "training-fold-only effective exit temperature and velocity"
            ),
            "powder_feed": (
                "particle release/loading and justified two-way coupling only"
            ),
            "spray_distance": (
                "empirical discrepancy/context only; absent from free-jet geometry"
            ),
            "fixed_h2_over_argon_ratio": (
                "not independent; deterministic transform of argon"
            ),
        },
        "first_layer_basis": (
            "intercept plus current and argon main effects for each exit state; "
            "interactions require a separately preregistered sensitivity layer"
        ),
        "calibration_release_allowed": active.calibration_release_allowed,
        "unresolved_gates": active.unresolved_gates(),
        "jointly_calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = contract_payload()
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote effective-exit parameterization contract: {args.output}")
    print("Joint calibration remains LOCKED until particle and DPV operators exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
