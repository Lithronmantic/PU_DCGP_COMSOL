
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pu_dcgp_comsol.comsol.dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_effective_exit_joint_correction"
    / "corrected_t11160_u1090_n1023.mph"
)
SOURCE_AUDIT = (
    HERE
    / "h11_outputs"
    / "effective_exit_joint_correction"
    / "cases"
    / "corrected_t11160_u1090_n1023.json"
)
B_EXPERIMENT_SUMMARY = (
    HERE.parents[1]
    / "experiments"
    / "pu_dcgp_v26"
    / "data"
    / "b_group_auxiliary_summary.json"
)
OUTPUT_DIR = HERE / "h11_outputs" / "b_observation_planes"
CONTRACT_PATH = OUTPUT_DIR / "h11_b_observation_plane_contract.json"


@dataclass(frozen=True, slots=True)
class BObservationPlaneContract:
    observation_planes_mm: tuple[float, float] = (90.0, 110.0)
    primary_aperture_radius_mm: float = (
        DpvSamplingVolumeContract().low_speed_equivalent_radius_mm
    )
    sensitivity_aperture_radius_mm: float = (
        DpvSamplingVolumeContract().high_speed_equivalent_radius_mm
    )
    expected_diameter_nodes: int = 7
    minimum_primary_particles_per_plane: int = 70
    absolute_mean_relative_error_limit: float = 0.10
    plane_difference_interval_level: float = 0.95
    primary_weighting: str = "A_pooled_detected_diameter_weights"

    def validate(self) -> None:
        if self.observation_planes_mm != (90.0, 110.0):
            raise ValueError("B observation planes must remain 90 and 110 mm")
        if not 0 < self.primary_aperture_radius_mm < self.sensitivity_aperture_radius_mm:
            raise ValueError("DPV aperture radii are not ordered")
        if self.expected_diameter_nodes != 7:
            raise ValueError("The frozen particle support contains seven sizes")
        if self.minimum_primary_particles_per_plane < 70:
            raise ValueError("The primary plane count gate cannot be weakened")
        if self.absolute_mean_relative_error_limit != 0.10:
            raise ValueError("The absolute-scale screen is frozen at ten percent")


def build_contract() -> dict:
    contract = BObservationPlaneContract()
    contract.validate()
    for path in (SOURCE_MODEL, SOURCE_AUDIT, B_EXPERIMENT_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    b_summary = json.loads(B_EXPERIMENT_SUMMARY.read_text(encoding="utf-8"))
    empirical_intervals = {
        outcome: values["average_far_minus_near_bootstrap_95_interval"]
        for outcome, values in b_summary["position_effects"].items()
    }
    return {
        "schema_version": "h11_b_observation_plane_contract_v1",
        "status": "pass_frozen_before_two_plane_comsol_extraction",
        "contract": asdict(contract),
        "source_model": str(SOURCE_MODEL.resolve()),
        "source_model_sha256": _sha256(SOURCE_MODEL),
        "source_audit": str(SOURCE_AUDIT.resolve()),
        "source_audit_sha256": _sha256(SOURCE_AUDIT),
        "b_experiment_summary": str(B_EXPERIMENT_SUMMARY.resolve()),
        "b_experiment_summary_sha256": _sha256(B_EXPERIMENT_SUMMARY),
        "empirical_average_far_minus_near_95_intervals": empirical_intervals,
        "admission_logic": {
            "numerical": [
                "all source trajectories cross both planes",
                "all seven diameter nodes occur at both planes",
                "at least 70 primary-aperture particles occur at each plane",
                "all extracted temperature, velocity, and diameter values are finite",
            ],
            "external_consistency_per_outcome": [
                "COMSOL 110-minus-90 median difference lies inside the frozen B run-bootstrap interval",
                "absolute median relative error is at most 10 percent at each plane",
            ],
            "claim_boundary": (
                "B positions were acquired sequentially, so passing is an external-consistency "
                "screen rather than causal validation. The five nominal B distance blocks are "
                "not COMSOL free-jet inputs. No B result may tune the source model."
            ),
        },
        "calibrated_on_b": False,
        "paper_prediction_allowed": False,
    }


def main() -> None:
    payload = build_contract()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(CONTRACT_PATH)
    print(payload["source_model_sha256"])


if __name__ == "__main__":
    main()
