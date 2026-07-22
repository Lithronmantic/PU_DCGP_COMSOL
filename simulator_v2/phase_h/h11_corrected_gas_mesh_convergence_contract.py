"""Freeze the corrected free-jet gas/particle mesh-convergence ladder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "corrected_gas_mesh_convergence"
    / "h11_corrected_gas_mesh_convergence_contract.json"
)


@dataclass(frozen=True)
class CorrectedGasMeshConvergenceContract:
    effective_exit_temperature_k: float = 11_160.0
    effective_exit_speed_m_s: float = 1_090.0
    radial_domain_mm: float = 40.0
    axial_domain_mm: float = 140.0
    automatic_mesh_levels: tuple[int, ...] = (4, 3, 2)
    conditional_bridge_default_hmax_mm: float = 0.75
    conditional_bridge_default_hmin_mm: float = 0.010
    conditional_bridge_default_growth_rate: float = 1.09
    conditional_bridge_default_curvature_factor: float = 0.25
    geometric_bridge_default_hmax_mm: float = 0.62449979984
    geometric_bridge_default_hmin_mm: float = 0.00774596669241
    geometric_bridge_default_growth_rate: float = 1.08498847920
    geometric_bridge_default_curvature_factor: float = 0.25
    level2_wall_temperature_scale_k: float = 10_000.0
    level2_flow_damping: float = 0.20
    level2_turbulence_damping: float = 0.15
    level2_pseudo_time_controller: str = "interp"
    level2_pseudo_time_initial_cfl: float = 1.0
    level2_pseudo_time_target_cfl: float = 10_000.0
    level2_pseudo_time_target_error: float = 0.10
    level2_pseudo_time_pid: tuple[float, float, float] = (
        0.65,
        0.05,
        0.05,
    )
    level2_anderson_acceleration: bool = False
    level2_preconditioner_tolerance: float = 1e-4
    level2_final_relative_tolerance: float = 1e-6
    level2_fully_coupled_maximum_iterations: int = 300
    level2_fully_coupled_initial_damping: float = 1e-4
    level2_fully_coupled_minimum_damping: float = 1e-10
    level2_fully_coupled_recovery_damping: float = 0.1
    level2_fully_coupled_residual_factor: float = 1.0
    common_dpv_profile_radii_mm: tuple[float, ...] = (
        0.001,
        1.0,
        2.0,
        4.0,
        6.0,
        10.0,
        20.0,
        30.0,
    )
    gas_temperature_anomaly_l2_limit_fraction: float = 0.01
    gas_speed_l2_limit_fraction: float = 0.01
    gas_pressure_l2_over_ambient_limit_fraction: float = 1e-4
    particle_quantile_limit_fraction: float = 0.01
    particles_per_size: int = 1023
    particle_output_step_us: float = 10.0
    particle_maximum_step_us: float = 2.0
    minimum_primary_aperture_particles: int = 70
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02

    def validate(self) -> None:
        if (
            self.effective_exit_temperature_k != 11_160.0
            or self.effective_exit_speed_m_s != 1_090.0
        ):
            raise ValueError("The corrected effective-exit case is frozen")
        if (
            self.radial_domain_mm != 40.0
            or self.axial_domain_mm != 140.0
        ):
            raise ValueError("Mesh convergence must keep the domain fixed")
        if self.automatic_mesh_levels != (4, 3, 2):
            raise ValueError("The adjacent automatic mesh ladder is frozen")
        if (
            self.conditional_bridge_default_hmax_mm,
            self.conditional_bridge_default_hmin_mm,
            self.conditional_bridge_default_growth_rate,
            self.conditional_bridge_default_curvature_factor,
        ) != (0.75, 0.010, 1.09, 0.25):
            raise ValueError("The conditional initialization bridge is frozen")
        if (
            self.geometric_bridge_default_hmax_mm,
            self.geometric_bridge_default_hmin_mm,
            self.geometric_bridge_default_growth_rate,
            self.geometric_bridge_default_curvature_factor,
        ) != (
            0.62449979984,
            0.00774596669241,
            1.08498847920,
            0.25,
        ):
            raise ValueError("The geometric-mean bridge is frozen")
        if (
            self.level2_wall_temperature_scale_k,
            self.level2_flow_damping,
            self.level2_turbulence_damping,
        ) != (10_000.0, 0.20, 0.15):
            raise ValueError("The level-2 variable scaling and damping are frozen")
        if (
            self.level2_pseudo_time_controller,
            self.level2_pseudo_time_initial_cfl,
            self.level2_pseudo_time_target_cfl,
            self.level2_pseudo_time_target_error,
            self.level2_pseudo_time_pid,
            self.level2_anderson_acceleration,
            self.level2_preconditioner_tolerance,
            self.level2_final_relative_tolerance,
            self.level2_fully_coupled_maximum_iterations,
            self.level2_fully_coupled_initial_damping,
            self.level2_fully_coupled_minimum_damping,
            self.level2_fully_coupled_recovery_damping,
            self.level2_fully_coupled_residual_factor,
        ) != (
            "interp",
            1.0,
            10_000.0,
            0.10,
            (0.65, 0.05, 0.05),
            False,
            1e-4,
            1e-6,
            300,
            1e-4,
            1e-10,
            0.1,
            1.0,
        ):
            raise ValueError("The level-2 solver repair is frozen")
        if max(self.common_dpv_profile_radii_mm) >= self.radial_domain_mm:
            raise ValueError("Common DPV samples must lie inside the domain")
        frozen_limits = (
            self.gas_temperature_anomaly_l2_limit_fraction,
            self.gas_speed_l2_limit_fraction,
            self.gas_pressure_l2_over_ambient_limit_fraction,
            self.particle_quantile_limit_fraction,
            self.mass_imbalance_limit_fraction,
            self.energy_imbalance_limit_fraction,
        )
        if frozen_limits != (0.01, 0.01, 1e-4, 0.01, 0.005, 0.02):
            raise ValueError("Mesh-convergence gates are frozen")
        if (
            self.particles_per_size != 1023
            or self.particle_output_step_us != 10.0
            or self.particle_maximum_step_us != 2.0
            or self.minimum_primary_aperture_particles != 70
        ):
            raise ValueError("The converged particle observation operator is frozen")


def contract_payload() -> dict[str, object]:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    return {
        "schema_version": "h11_corrected_gas_mesh_convergence_contract_v2",
        "status": "pass_frozen_corrected_gas_mesh_convergence_contract",
        "contract": asdict(contract),
        "decision_sequence": [
            "reuse the accepted level-4 corrected gas solution",
            "project level 4 only as the initial iterate for a level-3 solve",
            "project level 3 only as the initial iterate for a level-2 solve",
            (
                "if direct level-2 projection fails, allow one fixed intermediate "
                "mesh only as an initialization bridge"
            ),
            (
                "for the unchanged final level-2 equations, restart pseudo time "
                "stepping with the COMSOL 6.3 interpolated PID controller, "
                "disable Anderson acceleration, and retain the audited fixed "
                "segregated damping"
            ),
            (
                "save the unchanged level-2 solution at 1e-4 only as a "
                "preconditioner, then switch on the same mesh to a fully "
                "coupled PARDISO Newton solve requiring both solution and "
                "residual convergence at the final 1e-6 tolerance"
            ),
            (
                "if that final solve reaches its fixed iteration ceiling, "
                "insert exactly one additional initialization-only mesh whose "
                "maximum size, minimum size, and growth rate are the geometric "
                "means of the first bridge and automatic level-2 settings"
            ),
            "resolve unchanged equations and conservation on every target mesh",
            "compare level 3 to level 2 on the fixed z=100 mm common profile",
            "propagate both refined gas fields through the frozen DPV operator",
        ],
        "claim_boundary": (
            "Mesh refinement changes neither physics nor calibrated parameters. "
            "Passing supports numerical mesh independence only."
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
