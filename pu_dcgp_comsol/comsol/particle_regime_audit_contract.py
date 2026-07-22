
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "h11_outputs" / "particle_regime_audit"
CONTRACT_PATH = OUT_DIR / "h11_particle_regime_audit_contract.json"


@dataclass(frozen=True)
class ParticleRegimeAuditContract:

    material_name: str = "7YSZ"
    diameter_min_um: float = 16.0
    diameter_max_um: float = 90.0
    lumped_biot_limit: float = 0.1
    continuum_knudsen_limit: float = 0.01
    compressibility_mach_limit: float = 0.3
    minimum_radial_enthalpy_shells: int = 6
    use_surface_temperature_for_dpv: bool = True
    use_enthalpy_phase_change: bool = True
    constant_nusselt_two_allowed: bool = False
    ranz_marshall_released_as_final_plasma_closure: bool = False
    morphology_measured: bool = False
    gas_trajectory_samples_available: bool = False
    particle_model_released: bool = False

    def validate(self) -> None:
        if self.material_name != "7YSZ":
            raise ValueError("The operator-confirmed feedstock is 7YSZ")
        if not 0 < self.diameter_min_um < self.diameter_max_um:
            raise ValueError("The confirmed particle bounds must be ordered and positive")
        if not 0 < self.lumped_biot_limit <= 0.1:
            raise ValueError("The lumped-temperature Biot limit cannot exceed 0.1")
        if not 0 < self.continuum_knudsen_limit <= 0.01:
            raise ValueError("The continuum Knudsen limit cannot exceed 0.01")
        if not 0 < self.compressibility_mach_limit <= 0.3:
            raise ValueError("The particle-slip Mach limit cannot exceed 0.3")
        if self.minimum_radial_enthalpy_shells < 4:
            raise ValueError("A radial enthalpy model needs at least four shells")
        if not self.use_surface_temperature_for_dpv:
            raise ValueError("The DPV temperature estimand is particle surface temperature")
        if not self.use_enthalpy_phase_change:
            raise ValueError("YSZ melting must be represented in enthalpy space")
        if self.constant_nusselt_two_allowed:
            raise ValueError("Nu=2 is not valid as a finite-slip default")
        if self.ranz_marshall_released_as_final_plasma_closure:
            raise ValueError("The screening correlation is not a released plasma closure")
        if self.particle_model_released:
            raise ValueError("The regime-audit contract cannot claim model release")

    def unresolved_gates(self) -> list[str]:
        gates = [
            "sample_dimensionless_numbers_on_accepted_free_jet_trajectories",
            "select_and_source_trace_plasma_heat_transfer_correction",
            "select_drag_morphology_branch",
            "source_trace_7ysz_enthalpy_conductivity_density_and_emissivity",
            "verify_radial_shell_or_lumped_temperature_convergence",
            "verify_particle_time_step_and_population_convergence",
        ]
        if not self.morphology_measured:
            gates.append("prespecify_sphericity_sensitivity_without_data_selection")
        return gates


@dataclass(frozen=True)
class ParticleRegimeSample:

    diameter_m: float
    relative_speed_m_s: float
    gas_density_kg_m3: float
    gas_dynamic_viscosity_pa_s: float
    gas_thermal_conductivity_w_m_k: float
    gas_cp_j_kg_k: float
    gas_sound_speed_m_s: float
    gas_mean_free_path_m: float
    particle_thermal_conductivity_w_m_k: float

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("All regime-sample values must be finite")
        if self.relative_speed_m_s < 0:
            raise ValueError("Relative speed cannot be negative")
        positive_names = set(values) - {"relative_speed_m_s"}
        if any(float(values[name]) <= 0 for name in positive_names):
            raise ValueError("All non-speed regime-sample values must be positive")

    @property
    def reynolds(self) -> float:
        return (
            self.gas_density_kg_m3
            * self.relative_speed_m_s
            * self.diameter_m
            / self.gas_dynamic_viscosity_pa_s
        )

    @property
    def prandtl(self) -> float:
        return (
            self.gas_cp_j_kg_k
            * self.gas_dynamic_viscosity_pa_s
            / self.gas_thermal_conductivity_w_m_k
        )

    @property
    def relative_mach(self) -> float:
        return self.relative_speed_m_s / self.gas_sound_speed_m_s

    @property
    def knudsen(self) -> float:
        return self.gas_mean_free_path_m / self.diameter_m

    @property
    def screening_nusselt(self) -> float:

        return 2.0 + 0.6 * math.sqrt(self.reynolds) * self.prandtl ** (1.0 / 3.0)

    @property
    def screening_biot(self) -> float:

        return (
            self.screening_nusselt
            * self.gas_thermal_conductivity_w_m_k
            / (6.0 * self.particle_thermal_conductivity_w_m_k)
        )


def audit_particle_regimes(
    samples: Iterable[ParticleRegimeSample],
    contract: ParticleRegimeAuditContract | None = None,
) -> dict[str, object]:

    active = contract or ParticleRegimeAuditContract()
    active.validate()
    observed = list(samples)
    if not observed:
        raise ValueError("At least one accepted trajectory sample is required")
    for sample in observed:
        sample.validate()

    metrics = {
        "reynolds": [sample.reynolds for sample in observed],
        "prandtl": [sample.prandtl for sample in observed],
        "relative_mach": [sample.relative_mach for sample in observed],
        "knudsen": [sample.knudsen for sample in observed],
        "screening_nusselt": [sample.screening_nusselt for sample in observed],
        "screening_biot": [sample.screening_biot for sample in observed],
    }
    ranges = {
        name: {"minimum": min(values), "maximum": max(values)}
        for name, values in metrics.items()
    }
    lumped_temperature_admissible = (
        ranges["screening_biot"]["maximum"] < active.lumped_biot_limit
    )
    rarefaction_correction_required = (
        ranges["knudsen"]["maximum"] >= active.continuum_knudsen_limit
    )
    compressibility_correction_required = (
        ranges["relative_mach"]["maximum"]
        >= active.compressibility_mach_limit
    )
    return {
        "sample_count": len(observed),
        "ranges": ranges,
        "decisions": {
            "particle_temperature_state": (
                "lumped_temperature_candidate"
                if lumped_temperature_admissible
                else "radial_enthalpy_shells_required"
            ),
            "lumped_temperature_admissible": lumped_temperature_admissible,
            "rarefaction_correction_required": rarefaction_correction_required,
            "compressibility_correction_required": (
                compressibility_correction_required
            ),
            "surface_temperature_required_for_dpv": True,
            "ranz_marshall_status": "screening_only_not_final_closure",
        },
    }


def contract_payload(
    contract: ParticleRegimeAuditContract | None = None,
) -> dict[str, object]:
    active = contract or ParticleRegimeAuditContract()
    active.validate()
    return {
        "schema_version": "h11_particle_regime_audit_contract_v1",
        "status": "pass_contract_trajectory_evaluation_pending",
        "contract": asdict(active),
        "definitions": {
            "Re_p": "rho_g*|u_g-u_p|*d_p/mu_g",
            "Pr_g": "Cp_g*mu_g/k_g",
            "Ma_rel": "|u_g-u_p|/a_g",
            "Kn_p": "lambda_g/d_p",
            "Nu_screen": "2+0.6*sqrt(Re_p)*Pr_g^(1/3)",
            "Bi_screen": "Nu_screen*k_g/(6*k_p)",
        },
        "decision_policy": {
            "lumped_temperature": "allowed only if max(Bi_screen)<0.1",
            "radial_enthalpy_shells": "required otherwise",
            "rarefaction_correction": "required if max(Kn_p)>=0.01",
            "compressibility_correction": "required if max(Ma_rel)>=0.3",
            "dpv_temperature": "outer-surface temperature, never gas temperature",
        },
        "unresolved_gates": active.unresolved_gates(),
        "sources": {
            "comsol_particle_temperature": (
                "https://doc.comsol.com/6.3/doc/com.comsol.help.particle/"
                "particle_ug_fluid_flow.08.52.html"
            ),
            "comsol_convective_heat_losses": (
                "https://doc.comsol.com/6.3/doc/com.comsol.help.particle/"
                "particle_ug_fluid_flow.08.23.html"
            ),
            "ysz_surface_temperature_and_melting": (
                "https://doi.org/10.1007/s11090-005-8726-3"
            ),
            "plasma_particle_knudsen_review": (
                "https://doi.org/10.1016/S0040-6090(99)00101-7"
            ),
        },
        "particle_model_released": False,
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
    print(f"Wrote particle-regime contract: {args.output}")
    print("Trajectory regime audit and particle-model release remain PENDING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
