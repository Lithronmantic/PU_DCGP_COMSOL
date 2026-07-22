"""Freeze the first joint effective-exit and particle-count verification."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DIRECTIONAL_SUMMARY = (
    HERE
    / "h11_outputs"
    / "effective_exit_directional"
    / "h11_effective_exit_directional_summary.json"
)
SAMPLING_CONTRACT = (
    HERE
    / "h11_outputs"
    / "dpv_observation_operator"
    / "h11_dpv_sampling_volume_contract.json"
)
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "effective_exit_joint_screen"
    / "h11_effective_exit_joint_screen_contract.json"
)


@dataclass(frozen=True)
class JointScreenContract:
    directional_radius_mm: float = 0.5
    rounded_exit_temperature_k: float = 11_720.0
    rounded_exit_speed_m_s: float = 1_075.0
    particle_counts_per_size: tuple[int, ...] = (127, 255, 511)
    particle_output_step_us: float = 10.0
    adjacent_quantile_change_limit_fraction: float = 0.01
    minimum_primary_aperture_particles: int = 70

    def validate(self) -> None:
        if self.directional_radius_mm != 0.5:
            raise ValueError("The innermost common directional aperture is frozen")
        if not 10_000 <= self.rounded_exit_temperature_k <= 12_000:
            raise ValueError("Joint-screen temperature lies outside the bracket")
        if not 800 <= self.rounded_exit_speed_m_s <= 1_200:
            raise ValueError("Joint-screen speed lies outside the extension range")
        if self.particle_counts_per_size != (127, 255, 511):
            raise ValueError("Particle-count ladder cannot change after registration")
        if self.particle_output_step_us != 10.0:
            raise ValueError("Stored output interval is frozen at 10 us")
        if self.adjacent_quantile_change_limit_fraction != 0.01:
            raise ValueError("Particle convergence gate must remain one percent")
        if self.minimum_primary_aperture_particles != 70:
            raise ValueError("Primary-aperture count gate is frozen")


def _linear_inverse_candidate() -> dict[str, float]:
    summary = json.loads(DIRECTIONAL_SUMMARY.read_text(encoding="utf-8"))
    row = next(
        item
        for item in summary["centered_directional_derivatives"]
        if math.isclose(item["radius_mm"], 0.5)
    )
    nominal = next(
        case for case in summary["cases"] if case["case"] == "nominal_t10000_u600"
    )
    aperture = next(
        item for item in nominal["apertures"] if math.isclose(item["radius_mm"], 0.5)
    )
    target = nominal["target"]
    a = row["d_particle_temperature_c_per_exit_k"]
    b = row["d_particle_temperature_c_per_exit_speed_m_s"]
    c = row["d_particle_speed_per_exit_k"]
    d = row["d_particle_speed_per_exit_speed"]
    rhs_t = target["temperature_c"] - aperture["temperature_c"]
    rhs_u = target["speed_m_s"] - aperture["speed_m_s"]
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        raise ValueError("Directional response matrix is singular")
    delta_t = (rhs_t * d - b * rhs_u) / determinant
    delta_u = (a * rhs_u - rhs_t * c) / determinant
    return {
        "unrounded_exit_temperature_k": 10_000.0 + delta_t,
        "unrounded_exit_speed_m_s": 600.0 + delta_u,
        "determinant": determinant,
        "directional_radius_mm": row["radius_mm"],
    }


def contract_payload() -> dict[str, object]:
    contract = JointScreenContract()
    contract.validate()
    inverse = _linear_inverse_candidate()
    if abs(inverse["unrounded_exit_temperature_k"] - contract.rounded_exit_temperature_k) > 10:
        raise ValueError("Frozen rounded temperature no longer matches the inverse")
    if abs(inverse["unrounded_exit_speed_m_s"] - contract.rounded_exit_speed_m_s) > 5:
        raise ValueError("Frozen rounded speed no longer matches the inverse")
    sampling = json.loads(SAMPLING_CONTRACT.read_text(encoding="utf-8"))
    return {
        "schema_version": "h11_effective_exit_joint_screen_contract_v1",
        "status": "pass_frozen_joint_screen_and_particle_count_ladder",
        "contract": asdict(contract),
        "directional_linear_inverse": inverse,
        "joint_screen_case": {
            "name": "joint_t11720_u1075",
            "effective_exit_temperature_k": contract.rounded_exit_temperature_k,
            "effective_exit_speed_m_s": contract.rounded_exit_speed_m_s,
        },
        "instrument_informed_apertures_mm": sampling[
            "axisymmetric_equal_area_aperture_radius_mm"
        ],
        "frozen_physics": {
            "gas_material": "temperature-dependent pure Ar screening baseline",
            "effective_exit_radius_mm": 4.0,
            "profile_power": 2.0,
            "particle_entry_speed_m_s": 25.0,
            "particle_emissivity": 0.6,
            "diameter_support_um": [16.0, 90.0],
            "observation_plane_mm": 100.0,
        },
        "decision_rule": (
            "The joint point must pass gas conservation, all seven particle-size "
            "crossings, the primary aperture count gate, and <=1% adjacent changes "
            "for detected-weighted temperature and speed q10/q50/q90. It remains "
            "a pooled-data screening point and is never a held-out prediction."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = contract_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
