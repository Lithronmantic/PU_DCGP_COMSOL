"""Freeze the paper-facing 7YSZ particle-physics decisions for H11.

This layer is deliberately narrower than a solved particle model.  It replaces
the old constant-property, single-temperature assumptions with a traceable
material enthalpy interval and explicit model-selection rules.  Unknown powder
PSD, injector geometry, and emissivity remain calibration or prespecified
sensitivity quantities; they are not inferred from the detected DPV diameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GAS_ACCEPTANCE_PATH = (
    HERE
    / "h11_outputs"
    / "conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_numerical_acceptance.json"
)
DPV_TARGET_PATH = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "particle_physics_contract"
    / "h11_particle_physics_contract.json"
)


@dataclass(frozen=True)
class ParticlePhysicsContract:
    material_name: str = "7YSZ"
    diameter_min_um: float = 16.0
    diameter_max_um: float = 90.0
    diameter_nodes: int = 7
    density_kg_m3: float = 5890.0
    solid_heat_capacity_j_kg_k: float = 713.0
    solid_thermal_conductivity_w_m_k: float = 2.32
    solidus_k: float = 2923.13
    liquidus_k: float = 3023.13
    latent_heat_j_kg: float = 7.07e5
    radial_enthalpy_shells: int = 8
    lumped_biot_limit: float = 0.1
    continuum_knudsen_limit: float = 0.01
    compressibility_mach_limit: float = 0.3
    observation_plane_mm: float = 100.0
    emissivity_sensitivity_min: float = 0.3
    emissivity_sensitivity_max: float = 0.9
    sphericity_sensitivity_min: float = 0.8
    sphericity_sensitivity_max: float = 1.0

    def validate(self) -> None:
        if self.material_name != "7YSZ":
            raise ValueError("The operator-confirmed powder is 7YSZ")
        if not 0 < self.diameter_min_um < self.diameter_max_um:
            raise ValueError("Particle-diameter bounds must be ordered and positive")
        if self.diameter_nodes < 5 or self.diameter_nodes % 2 == 0:
            raise ValueError("Use at least five odd log-spaced diameter nodes")
        positive = (
            self.density_kg_m3,
            self.solid_heat_capacity_j_kg_k,
            self.solid_thermal_conductivity_w_m_k,
            self.latent_heat_j_kg,
            self.observation_plane_mm,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Material values and observation distance must be positive")
        if not self.solidus_k < self.liquidus_k:
            raise ValueError("The solidus must be below the liquidus")
        if self.radial_enthalpy_shells < 6:
            raise ValueError("The default radial enthalpy model needs at least six shells")
        if not 0 < self.lumped_biot_limit <= 0.1:
            raise ValueError("The lumped-temperature Biot limit cannot exceed 0.1")
        if not 0 < self.continuum_knudsen_limit <= 0.01:
            raise ValueError("The continuum Knudsen limit cannot exceed 0.01")
        if not 0 < self.compressibility_mach_limit <= 0.3:
            raise ValueError("The relative-Mach screening limit cannot exceed 0.3")
        if not 0 < self.emissivity_sensitivity_min < self.emissivity_sensitivity_max <= 1:
            raise ValueError("Invalid emissivity sensitivity interval")
        if not 0 < self.sphericity_sensitivity_min <= self.sphericity_sensitivity_max <= 1:
            raise ValueError("Invalid sphericity sensitivity interval")

    def diameter_nodes_um(self) -> list[float]:
        """Return a fixed log grid; its weights are not a measured feedstock PSD."""

        log_lo = math.log(self.diameter_min_um)
        step = (
            math.log(self.diameter_max_um) - log_lo
        ) / (self.diameter_nodes - 1)
        return [math.exp(log_lo + index * step) for index in range(self.diameter_nodes)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_payload(
    contract: ParticlePhysicsContract | None = None,
) -> dict[str, Any]:
    active = contract or ParticlePhysicsContract()
    active.validate()
    gas = _load_json(GAS_ACCEPTANCE_PATH)
    dpv = _load_json(DPV_TARGET_PATH)

    if gas["status"] != "pass_gas_phase_numerical_acceptance":
        raise RuntimeError("The accepted conservative gas field is required")
    if dpv["status"] != "pass_prt_particle_target_contract":
        raise RuntimeError("The corrected PRT particle target is required")

    primary_audit = _load_json(Path(gas["artifacts"]["primary_audit"]))
    gas_mass_flow = abs(
        primary_audit["refined_metrics"]["mass_flux_kg_s"]["nozzle_outward"]
    )
    powder_mass_flow = {
        "minimum": 10.0 / 60_000.0,
        "maximum": 30.0 / 60_000.0,
    }
    mass_loading = {
        name: value / gas_mass_flow for name, value in powder_mass_flow.items()
    }

    return {
        "schema_version": "h11_particle_physics_contract_v2",
        "status": "pass_particle_physics_decisions_build_pending",
        "contract": asdict(active),
        "diameter_nodes_um": active.diameter_nodes_um(),
        "diameter_node_interpretation": (
            "Fixed log-space numerical support over the operator-reported 16--90 um "
            "range. Node weights are latent training-fold quantities and are never "
            "identified with the DPV-detected diameter distribution."
        ),
        "material_model": {
            "state_variable": "radial specific enthalpy",
            "default_radial_shells": active.radial_enthalpy_shells,
            "dpv_temperature_output": "outer-surface temperature",
            "phase_change": (
                "linear apparent-heat-capacity interval from solidus to liquidus "
                "with the source-traceable latent heat"
            ),
            "lumped_temperature_allowed_only_if": "max trajectory Bi < 0.1",
            "constant_cp_k_scope": (
                "source-traceable baseline followed by property sensitivity; not "
                "claimed as a temperature-resolved material database"
            ),
        },
        "momentum_model": {
            "baseline": "COMSOL standard drag correlations for spherical particles",
            "morphology_sensitivity": (
                "Haider-Levenspiel branch over the prespecified sphericity interval"
            ),
            "rarefaction_rule": "enable correction if any accepted trajectory Kn >= 0.01",
        },
        "heat_transfer_model": {
            "Nu_equals_2_allowed": False,
            "screening_only": "Nu=2+0.6*Re_p^0.5*Pr^(1/3)",
            "release_rule": (
                "select the plasma heat-transfer correction only after Re, Pr, Kn, "
                "and relative Mach are sampled along accepted nominal trajectories"
            ),
            "compressibility_rule": (
                "use a compressibility-aware branch if relative Mach >= 0.3"
            ),
            "radiation": "gray-body sensitivity, emissivity is not measured",
        },
        "injection_model": {
            "geometry": "axisymmetric effective annular release",
            "status": "training-fold calibration quantity",
            "unknown_physical_inputs": [
                "injector position and angle",
                "carrier-gas species",
                "particle injection velocity distribution",
                "feedstock PSD shape and morphology",
            ],
            "claim_boundary": (
                "The release is an effective observation-calibrated source, not an "
                "ab-initio reconstruction of the unrecorded physical injector."
            ),
        },
        "coupling_decision": {
            "powder_feed_kg_s": powder_mass_flow,
            "accepted_nominal_gas_inlet_kg_s": gas_mass_flow,
            "nominal_mass_loading_ratio": mass_loading,
            "one_way_model_role": "initial trajectory and regime screening",
            "two_way_sensitivity_required": True,
            "reason": (
                "The nominal powder/gas mass-flow ratio is not small enough to assume "
                "that momentum and heat feedback are negligible without a test."
            ),
        },
        "observation_operator": {
            "plane": "fixed gun-relative z=100 mm",
            "workpiece_present": False,
            "stages": [
                "crossing particle population",
                "DPV detection/retention",
                "1000-row PRT retained sample",
                "run distribution summaries",
            ],
            "validation_targets": dpv["required_comsol_outputs_per_run"],
        },
        "source_artifacts": {
            "gas_acceptance": str(GAS_ACCEPTANCE_PATH.resolve()),
            "gas_acceptance_sha256": _sha256(GAS_ACCEPTANCE_PATH),
            "gas_model": gas["artifacts"]["primary_model"],
            "gas_model_sha256": gas["artifacts"]["primary_model_sha256"],
            "dpv_target": str(DPV_TARGET_PATH.resolve()),
            "dpv_target_sha256": _sha256(DPV_TARGET_PATH),
        },
        "literature_sources": {
            "comsol_particle_temperature_and_biot": (
                "https://doc.comsol.com/6.3/doc/com.comsol.help.particle/"
                "particle_ug_fluid_flow.08.52.html"
            ),
            "comsol_drag_and_rarefaction": (
                "https://doc.comsol.com/6.3/doc/com.comsol.help.particle/"
                "ParticleTracingModuleUsersGuide.pdf"
            ),
            "ysz_6_to_8wtpct_property_table": (
                "https://doi.org/10.15541/jim20140212"
            ),
            "ysz_surface_temperature_distribution_model": (
                "https://doi.org/10.1007/s11090-005-8726-3"
            ),
            "plasma_noncontinuum_heat_transfer": (
                "https://doi.org/10.1016/0017-9310(90)90046-W"
            ),
        },
        "remaining_before_nominal_solve": [
            "build the accepted-gas-field particle tree",
            "implement radial enthalpy-shell auxiliary ODEs",
            "add effective release and z=100 mm crossing bookkeeping",
            "sample trajectory Re/Pr/Kn/Ma and freeze the released correlations",
        ],
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote particle-physics contract: {args.output}")
    print("Particle-tree build and nominal solve remain PENDING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
