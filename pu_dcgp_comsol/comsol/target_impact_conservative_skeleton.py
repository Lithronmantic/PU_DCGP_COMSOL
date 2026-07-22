
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.external_plume_skeleton import (
    ARGON_REFERENCE_MODEL,
    _copy_argon_material,
)


HERE = Path(__file__).resolve().parent
TARGET_GEOMETRY_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_geometry"
    / "h11_target_impact_geometry_latest.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_skeleton"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_skeleton"
MODEL_PATH = MODEL_DIR / "h11_target_impact_conservative_skeleton_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_conservative_skeleton_audit.json"


@dataclass(frozen=True)
class ConservativeTargetContract:

    ambient_temperature_k: float = 300.0
    ambient_pressure_pa: float = 101_325.0
    provisional_exit_temperature_k: float = 10_000.0
    provisional_exit_velocity_m_s: float = 600.0
    argon_molar_mass_kg_mol: float = 39.948e-3
    representative_exit_cp_j_kg_k: float = 1457.7
    turbulence_intensity: float = 0.05
    turbulence_length_mm: float = 0.4
    radial_profile_power: float = 2.0
    property_temperature_floor_k: float = 250.0
    target_temperature_c: float = 108.0
    carrier_gas_included: bool = False
    h2_mixture_included: bool = False
    effective_exit_calibrated: bool = False
    held_out_validation_passed: bool = False

    def validate(self) -> None:
        positive = (
            self.ambient_temperature_k,
            self.ambient_pressure_pa,
            self.provisional_exit_temperature_k,
            self.provisional_exit_velocity_m_s,
            self.argon_molar_mass_kg_mol,
            self.representative_exit_cp_j_kg_k,
            self.turbulence_intensity,
            self.turbulence_length_mm,
            self.radial_profile_power,
            self.property_temperature_floor_k,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Conservative target inputs must be finite and positive")
        if self.provisional_exit_temperature_k <= self.ambient_temperature_k:
            raise ValueError("Exit temperature must exceed ambient")
        if not 0 < self.turbulence_intensity < 1:
            raise ValueError("Turbulence intensity must lie in (0, 1)")
        if self.radial_profile_power < 1:
            raise ValueError("Radial profile power must be at least one")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the source-table range")
        if self.carrier_gas_included:
            raise ValueError("Unknown carrier gas must remain excluded")
        if self.h2_mixture_included:
            raise ValueError("An unverified Ar-H2 mixture model cannot be enabled")
        if self.effective_exit_calibrated or self.held_out_validation_passed:
            raise ValueError("The skeleton cannot claim calibration or validation")

    @property
    def argon_specific_gas_constant_j_kg_k(self) -> float:
        return 8.31446261815324 / self.argon_molar_mass_kg_mol

    @property
    def representative_gamma(self) -> float:
        cp = self.representative_exit_cp_j_kg_k
        rs = self.argon_specific_gas_constant_j_kg_k
        if cp <= rs:
            raise ValueError("Representative Cp must exceed the specific gas constant")
        return cp / (cp - rs)

    @property
    def provisional_centerline_mach(self) -> float:
        sound_speed = math.sqrt(
            self.representative_gamma
            * self.argon_specific_gas_constant_j_kg_k
            * self.provisional_exit_temperature_k
        )
        return self.provisional_exit_velocity_m_s / sound_speed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _selection_entities(comp: Any, tag: str) -> list[int]:
    return [int(value) for value in comp.selection(tag).entities()]


def build_model(
    client: Any,
    contract: ConservativeTargetContract,
) -> tuple[Any, Any, dict[str, Any]]:
    contract.validate()
    if not TARGET_GEOMETRY_MODEL.exists():
        raise FileNotFoundError(TARGET_GEOMETRY_MODEL)
    if not ARGON_REFERENCE_MODEL.exists():
        raise FileNotFoundError(ARGON_REFERENCE_MODEL)

    model = client.load(str(TARGET_GEOMETRY_MODEL))
    model.rename("h11_target_impact_conservative_skeleton")
    reference = client.load(str(ARGON_REFERENCE_MODEL))
    jm = model.java
    params = jm.param()
    comp = jm.component("comp1")

    _set_param(
        params,
        "T_amb",
        f"{contract.ambient_temperature_k:.12g}[K]",
        "Ambient temperature",
    )
    _set_param(
        params,
        "p_amb",
        f"{contract.ambient_pressure_pa:.12g}[Pa]",
        "Ambient absolute pressure",
    )
    _set_param(
        params,
        "T_exit_eff",
        f"{contract.provisional_exit_temperature_k:.12g}[K]",
        "Training-only effective exit centerline temperature",
    )
    _set_param(
        params,
        "u_exit_eff",
        f"{contract.provisional_exit_velocity_m_s:.12g}[m/s]",
        "Training-only effective exit centerline speed",
    )
    _set_param(
        params,
        "M_Ar",
        f"{contract.argon_molar_mass_kg_mol:.12g}[kg/mol]",
        "Argon molar mass",
    )
    _set_param(
        params,
        "R_Ar",
        "R_const/M_Ar",
        "Argon specific gas constant",
    )
    _set_param(
        params,
        "I_turb",
        f"{contract.turbulence_intensity:.12g}",
        "Inlet turbulence intensity sensitivity",
    )
    _set_param(
        params,
        "L_turb",
        f"{contract.turbulence_length_mm:.12g}[mm]",
        "Inlet turbulence length sensitivity",
    )
    _set_param(
        params,
        "profile_power",
        f"{contract.radial_profile_power:.12g}",
        "Training-only radial profile exponent",
    )

    material_audit = _copy_argon_material(reference.java, comp)
    material_audit["density_usage"] = (
        "Source 1-atm density table retained for provenance but not used by "
        "the conservative interface; density follows p/(R_Ar*T)."
    )
    material_audit["caloric_model"] = (
        "Calorically imperfect ideal argon using source Cp(T); equilibrium-plasma "
        "Cp is a pure-Ar sensitivity pending a verified Ar-H2 mixture table."
    )

    inlet_profile = comp.variable().create("var_conservative_inlet")
    inlet_profile.label("Conservative effective-exit profile")
    inlet_profile.selection().geom("geom1", 1)
    inlet_profile.selection().named("geom1_sel_nozzle_in")
    inlet_profile.set(
        "nozzle_shape",
        "(1-(r/r_exit_eff)^2)^profile_power",
        "Smooth core profile: one on axis and zero at the effective edge",
    )

    hmnf = comp.physics().create(
        "hmnf",
        "HighMachNumberFlowTurbulentkomega",
        "geom1",
    )
    hmnf.label("Conservative compressible target plume, k-omega")
    physical = hmnf.prop("PhysicalModelProperty")
    physical.set("Compressibility", "CompressibleMALT03")
    physical.set("Tref", "T_amb")
    physical.set("includeKineticEnergy", "1")
    turbulence = hmnf.prop("TurbulenceModelProperty")
    turbulence.set("ThermalTurbType", "KaysCrawford")
    turbulence.set("ThermalWallFunction", "Standard")
    hmnf.prop("AdvancedSettingProperty").set("UsePseudoTime", "1")
    hmnf.prop("AdvancedSettingProperty").set("PseudoTimeSetting", "Automatic")

    fluid = hmnf.feature("fluid1")
    fluid.set("fluidType", "idealGas")
    fluid.set("gasConstantType", "specificGC")
    fluid.set("Rs_mat", "userdef")
    fluid.set("Rs", "R_Ar")
    fluid.set("CpOrGammaOption", "Cp")
    fluid.set("Cp_mat", "from_mat")
    fluid.set("k_mat", "from_mat")
    fluid.set("mu_mat", "from_mat")
    fluid.set("PressureWorkFormulationType", "FullFormulation")

    initial = hmnf.feature("init1")
    initial.set("u_init", ["0", "0", "0"])
    initial.set("p_init", "p_amb")
    initial.set("Tinit", "T_amb")

    nozzle = hmnf.create("nozzle_in", "HighMachNumberFlowInlet", 1)
    nozzle.label("Training-calibrated subsonic effective exit")
    nozzle.selection().named("geom1_sel_nozzle_in")
    nozzle.set("FlowCondition", "Subsonic")
    nozzle.set("BoundaryCondition", "Velocity")
    nozzle.set("U0in", "u_exit_eff*nozzle_shape")
    nozzle.set("TemperatureHeatflux", "Temperature")
    nozzle.set("T0", "T_amb+(T_exit_eff-T_amb)*nozzle_shape")
    nozzle.set("RANSVarOption", "SpecifyTurbulentLengthScaleAndIntensity")
    nozzle.set("IT", "I_turb")
    nozzle.set("LT", "L_turb")
    nozzle.set("SuppressBackflow", "1")

    ambient_entities = sorted(
        {
            *_selection_entities(comp, "geom1_sel_ambient_in"),
            *_selection_entities(comp, "geom1_sel_far_r"),
        }
    )
    ambient = hmnf.create("ambient_open", "HighMachNumberFlowInlet", 1)
    ambient.label("Ambient pressure-temperature opening")
    ambient.selection().set(ambient_entities)
    ambient.set("FlowCondition", "Subsonic")
    ambient.set("BoundaryCondition", "Pressure")
    ambient.set("p0", "p_amb")
    ambient.set("TemperatureHeatflux", "Temperature")
    ambient.set("T0", "T_amb")
    ambient.set("RANSVarOption", "SpecifyTurbulentLengthScaleAndIntensity")
    ambient.set("IT", "I_turb")
    ambient.set("LT", "L_turb")
    ambient.set("SuppressBackflow", "0")

    target_temperature = hmnf.create(
        "target_temperature",
        "TemperatureBoundary",
        1,
    )
    target_temperature.label("Measured-range isothermal workpiece")
    target_temperature.selection().named("geom1_sel_target")
    target_temperature.set("T0", "T_target")

    mesh = comp.mesh().create("mesh1", "geom1")
    mesh.label("Provisional conservative-plume mesh")
    mesh.autoMeshSize(4)

    study = jm.study().create("std1")
    study.label("LOCKED conservative target-impact stationary study")
    study.create("stat", "Stationary")
    return model, jm, material_audit


def audit_model(
    jm: Any,
    contract: ConservativeTargetContract,
    material_audit: dict[str, Any],
) -> dict[str, Any]:
    comp = jm.component("comp1")
    hmnf = comp.physics("hmnf")
    physical = hmnf.prop("PhysicalModelProperty")
    fluid = hmnf.feature("fluid1")
    expected = {
        "physics_type": str(hmnf.getType()),
        "compressibility": str(physical.getString("Compressibility")),
        "include_kinetic_energy": str(physical.getString("includeKineticEnergy")),
        "fluid_type": str(fluid.getString("fluidType")),
        "specific_gas_constant": str(fluid.getString("Rs")),
        "cp_source": str(fluid.getString("Cp_mat")),
        "thermal_conductivity_source": str(fluid.getString("k_mat")),
        "dynamic_viscosity_source": str(fluid.getString("mu_mat")),
        "pressure_work": str(fluid.getString("PressureWorkFormulationType")),
    }
    required = {
        "physics_type": "HighMachNumberFlowTurbulentkomega",
        "compressibility": "CompressibleMALT03",
        "include_kinetic_energy": "1",
        "fluid_type": "idealGas",
        "specific_gas_constant": "R_Ar",
        "cp_source": "from_mat",
        "thermal_conductivity_source": "from_mat",
        "dynamic_viscosity_source": "from_mat",
        "pressure_work": "FullFormulation",
    }
    if expected != required:
        raise RuntimeError(f"Conservative physics tree drifted: {expected}")

    selections = {
        "nozzle": _selection_entities(comp, "geom1_sel_nozzle_in"),
        "ambient_in": _selection_entities(comp, "geom1_sel_ambient_in"),
        "far_r": _selection_entities(comp, "geom1_sel_far_r"),
        "target": _selection_entities(comp, "geom1_sel_target"),
    }
    if set(selections["nozzle"]) & (
        set(selections["ambient_in"]) | set(selections["far_r"])
    ):
        raise RuntimeError("Conservative nozzle and ambient selections overlap")

    return {
        "schema_version": "h11_conservative_target_skeleton_v1",
        "status": "pass_skeleton_solve_locked",
        "contract": asdict(contract),
        "provisional_centerline_mach": contract.provisional_centerline_mach,
        "model_level": (
            "calorically-imperfect pure-Ar all-Mach external plume; not an "
            "internal arc reconstruction and not yet an Ar-H2 mixture"
        ),
        "physics": expected,
        "material": material_audit,
        "selections": selections,
        "boundary_conditions": {
            "nozzle": "subsonic velocity plus static-temperature profile",
            "ambient": (
                "integrated subsonic pressure-temperature opening with "
                "backflow allowed"
            ),
            "target": "no-slip default wall plus measured-range isothermal condition",
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
        "next_gate": (
            "Run conservative continuation, require bounded fields and official "
            "mass/total-energy closure, then rebuild a systematic mesh ladder."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    contract = ConservativeTargetContract()
    contract.validate()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model, jm, material_audit = build_model(client, contract)
        audit = audit_model(jm, contract, material_audit)
        model.save(str(args.model))
    finally:
        client.clear()

    audit.update(
        {
            "comsol_version": args.version,
            "cores": args.cores,
            "runtime_sec": time.time() - started,
            "model_path": str(args.model.resolve()),
            "model_sha256": _sha256(args.model),
            "source_geometry": str(TARGET_GEOMETRY_MODEL.resolve()),
            "source_geometry_sha256": _sha256(TARGET_GEOMETRY_MODEL),
            "argon_reference_model": str(ARGON_REFERENCE_MODEL),
            "argon_reference_sha256": _sha256(ARGON_REFERENCE_MODEL),
        }
    )
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print(
        "Conservative skeleton gate: PASS; solve and experimental prediction remain LOCKED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
