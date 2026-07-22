"""Freeze one interaction-aware correction to the first joint screen."""

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
FIRST_JOINT_AUDIT = (
    HERE
    / "h11_outputs"
    / "effective_exit_joint_screen"
    / "cases"
    / "joint_t11720_u1075_n0127.json"
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
    / "effective_exit_joint_correction"
    / "h11_effective_exit_joint_correction_contract.json"
)


@dataclass(frozen=True)
class JointCorrectionContract:
    rounded_exit_temperature_k: float = 11_160.0
    rounded_exit_speed_m_s: float = 1_090.0
    pilot_particles_per_size: int = 127
    convergence_counts_per_size: tuple[int, ...] = (255, 511)
    conditional_escalation_particles_per_size: int = 1023
    particle_output_step_us: float = 10.0
    pilot_temperature_gap_limit_fraction: float = 0.05
    pilot_speed_gap_limit_fraction: float = 0.05
    convergence_limit_fraction: float = 0.01
    minimum_primary_aperture_particles: int = 70

    def validate(self) -> None:
        if self.rounded_exit_temperature_k != 11_160.0:
            raise ValueError("The interaction-aware temperature correction is frozen")
        if self.rounded_exit_speed_m_s != 1_090.0:
            raise ValueError("The interaction-aware speed correction is frozen")
        if self.pilot_particles_per_size != 127:
            raise ValueError("The correction pilot must use 127 particles per size")
        if self.convergence_counts_per_size != (255, 511):
            raise ValueError("The convergence escalation ladder is frozen")
        if self.conditional_escalation_particles_per_size != 1023:
            raise ValueError("The conditional count-only escalation is frozen")
        if self.particle_output_step_us != 10.0:
            raise ValueError("Stored output interval is frozen")
        if (
            self.pilot_temperature_gap_limit_fraction != 0.05
            or self.pilot_speed_gap_limit_fraction != 0.05
            or self.convergence_limit_fraction != 0.01
        ):
            raise ValueError("Correction and convergence gates cannot change")


def _aperture(case: dict, radius_mm: float) -> dict:
    return next(
        item
        for item in case["apertures"]
        if math.isclose(item["radius_mm"], radius_mm, rel_tol=0, abs_tol=1e-12)
    )


def _primary_joint_observation() -> tuple[float, float, float, float]:
    sampling = json.loads(SAMPLING_CONTRACT.read_text(encoding="utf-8"))
    radius = sampling["axisymmetric_equal_area_aperture_radius_mm"][
        "low_speed_configuration"
    ]
    payload = json.loads(FIRST_JOINT_AUDIT.read_text(encoding="utf-8"))
    solve = payload["particle_solve_audit"]
    target = solve["trajectory_audit"]["unweighted_target_comparison"][
        "dpv_prt_pooled_median"
    ]
    aperture = next(
        item
        for item in solve["trajectory_audit"]["centerline_aperture_sensitivity"]
        if math.isclose(
            item["aperture_radius_mm"], radius, rel_tol=0, abs_tol=1e-12
        )
    )
    prediction = aperture["empirical_detected_diameter_weighted_median"]
    return (
        float(prediction["temperature_c"]),
        float(prediction["speed_m_s"]),
        float(target["temperature_c"]),
        float(target["speed_m_s"]),
    )


def interaction_aware_inverse() -> dict[str, float]:
    summary = json.loads(DIRECTIONAL_SUMMARY.read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in summary["cases"]}
    radius = 0.5
    y_t_low = _aperture(cases["t8000_u600"], radius)["temperature_c"]
    y_t_zero = _aperture(cases["nominal_t10000_u600"], radius)["temperature_c"]
    y_t_high = _aperture(cases["t12000_u600"], radius)["temperature_c"]
    y_u_low = _aperture(cases["t10000_u400"], radius)
    y_u_high = _aperture(cases["t10000_u800"], radius)

    h_t = 2_000.0
    h_u = 200.0
    temp_quad = (y_t_high + y_t_low - 2 * y_t_zero) / (2 * h_t**2)
    temp_t_slope = (y_t_high - y_t_low) / (2 * h_t)
    temp_u_slope = (
        y_u_high["temperature_c"] - y_u_low["temperature_c"]
    ) / (2 * h_u)
    speed_t_slope = (
        _aperture(cases["t12000_u600"], radius)["speed_m_s"]
        - _aperture(cases["t8000_u600"], radius)["speed_m_s"]
    ) / (2 * h_t)
    speed_u_slope = (
        y_u_high["speed_m_s"] - y_u_low["speed_m_s"]
    ) / (2 * h_u)
    speed_zero = _aperture(
        cases["nominal_t10000_u600"], radius
    )["speed_m_s"]

    joint_temperature, joint_speed, target_temperature, target_speed = (
        _primary_joint_observation()
    )
    x_joint = 11_720.0 - 10_000.0
    y_joint = 1_075.0 - 600.0
    temp_additive = (
        temp_quad * x_joint**2
        + temp_t_slope * x_joint
        + y_t_zero
        + temp_u_slope * y_joint
    )
    speed_additive = (
        speed_zero
        + speed_t_slope * x_joint
        + speed_u_slope * y_joint
    )
    temp_interaction = (
        joint_temperature - temp_additive
    ) / (x_joint * y_joint)
    speed_interaction = (
        joint_speed - speed_additive
    ) / (x_joint * y_joint)

    x = 1_160.0
    y = 490.0
    for _ in range(12):
        f_temp = (
            temp_quad * x**2
            + temp_t_slope * x
            + y_t_zero
            + temp_u_slope * y
            + temp_interaction * x * y
            - target_temperature
        )
        f_speed = (
            speed_zero
            + speed_t_slope * x
            + speed_u_slope * y
            + speed_interaction * x * y
            - target_speed
        )
        j11 = 2 * temp_quad * x + temp_t_slope + temp_interaction * y
        j12 = temp_u_slope + temp_interaction * x
        j21 = speed_t_slope + speed_interaction * y
        j22 = speed_u_slope + speed_interaction * x
        determinant = j11 * j22 - j12 * j21
        if abs(determinant) < 1e-12:
            raise ValueError("Interaction-aware response Jacobian is singular")
        dx = (-f_temp * j22 + j12 * f_speed) / determinant
        dy = (-j11 * f_speed + f_temp * j21) / determinant
        x += dx
        y += dy
        if math.hypot(dx, dy) < 1e-8:
            break
    return {
        "unrounded_exit_temperature_k": 10_000.0 + x,
        "unrounded_exit_speed_m_s": 600.0 + y,
        "temperature_quadratic_coefficient": temp_quad,
        "temperature_speed_interaction": temp_interaction,
        "velocity_speed_interaction": speed_interaction,
        "first_joint_temperature_c": joint_temperature,
        "first_joint_speed_m_s": joint_speed,
        "target_temperature_c": target_temperature,
        "target_speed_m_s": target_speed,
    }


def contract_payload() -> dict[str, object]:
    contract = JointCorrectionContract()
    contract.validate()
    inverse = interaction_aware_inverse()
    if abs(inverse["unrounded_exit_temperature_k"] - 11_160.0) > 10:
        raise ValueError("Rounded corrected temperature no longer matches")
    if abs(inverse["unrounded_exit_speed_m_s"] - 1_090.0) > 10:
        raise ValueError("Rounded corrected speed no longer matches")
    return {
        "schema_version": "h11_effective_exit_joint_correction_contract_v2",
        "status": "pass_frozen_interaction_aware_joint_correction",
        "contract": asdict(contract),
        "interaction_aware_inverse": inverse,
        "corrected_case": {
            "name": "corrected_t11160_u1090",
            "effective_exit_temperature_k": (
                contract.rounded_exit_temperature_k
            ),
            "effective_exit_speed_m_s": contract.rounded_exit_speed_m_s,
        },
        "decision_sequence": [
            "run the 127-per-size correction pilot",
            "require absolute pooled median temperature and speed gaps <=5%",
            "only then run 255 and 511 particles per size",
            "require final adjacent q10/q50/q90 change <=1%",
            (
                "only if 511 is converged but the primary aperture contains "
                "fewer than 70 particles, run the count-only 1023 escalation"
            ),
        ],
        "claim_boundary": (
            "One preregistered correction after observing the first joint "
            "interaction. No further pooled-target tuning is allowed; subsequent "
            "parameter estimation must occur inside grouped training folds."
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
