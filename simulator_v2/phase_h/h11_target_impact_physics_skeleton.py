"""H11 layer 5: nonisothermal target-impact COMSOL physics skeleton.

This model is the four-condition spray-distance branch.  It starts from the
verified target geometry, adds the same external-plume flow/thermal physics
used by the no-workpiece measurement branch, and applies the operator-
reported 97--119 degC workpiece-temperature range at the target wall.

No DPV coordinate appears in this model.  The effective nozzle exit state is
still a training-only calibration object because nozzle geometry and arc
power are unavailable.  The saved stationary study therefore remains locked
until the calibration/validation runner is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_external_plume_skeleton import (
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
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_physics_skeleton"
OUT_DIR = HERE / "h11_outputs" / "target_impact_physics_skeleton"
MODEL_PATH = MODEL_DIR / "h11_target_impact_physics_skeleton_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_physics_skeleton_audit.json"


@dataclass(frozen=True)
class TargetImpactPhysicsContract:
    """Target-branch parameters and honest calibration status."""

    ambient_temperature_k: float = 300.0
    ambient_pressure_pa: float = 101325.0
    nominal_exit_temperature_k: float = 10000.0
    nominal_exit_velocity_m_s: float = 600.0
    nominal_turbulence_intensity: float = 0.05
    nominal_turbulence_length_mm: float = 0.4
    nominal_radial_profile_power: float = 2.0
    h2_setting: float = 2.5
    argon_setting_range: tuple[float, float] = (80.0, 120.0)
    target_temperature_range_c: tuple[float, float] = (97.0, 119.0)
    nominal_target_temperature_c: float = 108.0
    carrier_gas_included: bool = False
    effective_exit_calibrated: bool = False
    held_out_validation_passed: bool = False

    def validate(self) -> None:
        if self.ambient_temperature_k <= 0 or self.ambient_pressure_pa <= 0:
            raise ValueError("Ambient state must be positive")
        if self.nominal_exit_temperature_k <= self.ambient_temperature_k:
            raise ValueError("Nominal exit temperature must exceed ambient")
        if self.nominal_exit_velocity_m_s <= 0:
            raise ValueError("Nominal exit velocity must be positive")
        if not 0 < self.nominal_turbulence_intensity < 1:
            raise ValueError("Turbulence intensity must be in (0, 1)")
        if self.nominal_turbulence_length_mm <= 0:
            raise ValueError("Turbulence length must be positive")
        if self.nominal_radial_profile_power < 1:
            raise ValueError("Radial profile power must be at least one")
        if self.h2_setting <= 0:
            raise ValueError("H2 setting must be positive")
        argon_low, argon_high = self.argon_setting_range
        if not 0 < argon_low <= argon_high:
            raise ValueError("Argon setting range is invalid")
        target_low, target_high = self.target_temperature_range_c
        if not target_low <= self.nominal_target_temperature_c <= target_high:
            raise ValueError("Nominal target temperature is outside the observed range")
        if self.carrier_gas_included:
            raise ValueError("Operator instructed that unknown carrier gas be excluded")

    @property
    def solve_unlocked(self) -> bool:
        return self.effective_exit_calibrated

    @property
    def paper_use_unlocked(self) -> bool:
        return self.effective_exit_calibrated and self.held_out_validation_passed

    def h2_to_argon_setting_ratio_range(self) -> tuple[float, float]:
        low, high = self.argon_setting_range
        return self.h2_setting / high, self.h2_setting / low


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
    client: Any, contract: TargetImpactPhysicsContract
) -> tuple[Any, Any, dict[str, Any]]:
    contract.validate()
    if not TARGET_GEOMETRY_MODEL.exists():
        raise FileNotFoundError(TARGET_GEOMETRY_MODEL)
    if not ARGON_REFERENCE_MODEL.exists():
        raise FileNotFoundError(ARGON_REFERENCE_MODEL)

    model = client.load(str(TARGET_GEOMETRY_MODEL))
    model.rename("h11_target_impact_physics_skeleton")
    reference = client.load(str(ARGON_REFERENCE_MODEL))
    jm = model.java
    params = jm.param()
    comp = jm.component("comp1")

    _set_param(params, "T_amb", f"{contract.ambient_temperature_k:.9g}[K]", "Ambient temperature")
    _set_param(params, "p_amb", f"{contract.ambient_pressure_pa:.9g}[Pa]", "Ambient pressure")
    _set_param(
        params,
        "T_exit_eff",
        f"{contract.nominal_exit_temperature_k:.9g}[K]",
        "Training-only effective nozzle temperature",
    )
    _set_param(
        params,
        "u_exit_eff",
        f"{contract.nominal_exit_velocity_m_s:.9g}[m/s]",
        "Training-only effective nozzle speed",
    )
    _set_param(
        params,
        "I_turb",
        f"{contract.nominal_turbulence_intensity:.9g}",
        "Inlet turbulence-intensity sensitivity parameter",
    )
    _set_param(
        params,
        "L_turb",
        f"{contract.nominal_turbulence_length_mm:.9g}[mm]",
        "Inlet turbulence-length sensitivity parameter",
    )
    _set_param(
        params,
        "profile_power",
        f"{contract.nominal_radial_profile_power:.9g}",
        "Training-calibrated smooth radial-profile exponent",
    )
    ratio_low, ratio_high = contract.h2_to_argon_setting_ratio_range()
    _set_param(
        params,
        "h2_ar_setting_ratio_low",
        f"{ratio_low:.12g}",
        "Lower dimensionless H2/Ar setting ratio; not a mass or mole fraction",
    )
    _set_param(
        params,
        "h2_ar_setting_ratio_high",
        f"{ratio_high:.12g}",
        "Upper dimensionless H2/Ar setting ratio; not a mass or mole fraction",
    )

    material_audit = _copy_argon_material(reference.java, comp)

    inlet_profile = comp.variable().create("var_inlet_profile")
    inlet_profile.label("Training-calibrated effective inlet profile")
    inlet_profile.selection().geom("geom1", 1)
    inlet_profile.selection().named("geom1_sel_nozzle_in")
    inlet_profile.set(
        "nozzle_shape",
        "(1-(r/r_exit_eff)^2)^profile_power",
        "One on axis and zero at the effective nozzle edge",
    )

    spf = comp.physics().create("spf", "TurbulentFlowkomega", "geom1")
    spf.label("Target-branch external plume, k-omega")
    spf.prop("PhysicalModelProperty").set("Compressibility", "WeaklyCompressible")
    spf.prop("PhysicalModelProperty").set("Tref", "T_amb")
    spf.prop("PhysicalModelProperty").set("pref", "p_amb")
    inlet = spf.create("inl1", "InletBoundary", 1)
    inlet.label("Training-calibrated effective nozzle state")
    inlet.selection().named("geom1_sel_nozzle_in")
    inlet.set("BoundaryCondition", "Velocity")
    inlet.set("U0in", "u_exit_eff*nozzle_shape")
    inlet.set("RANSVarOption", "SpecifyTurbulentLengthScaleAndIntensity")
    inlet.set("IT", "I_turb")
    inlet.set("LT", "L_turb")

    open_entities = sorted(
        set(
            _selection_entities(comp, "geom1_sel_ambient_in")
            + _selection_entities(comp, "geom1_sel_far_r")
        )
    )
    flow_open = spf.create("open1", "OpenBoundary", 1)
    flow_open.label("Ambient entrainment boundary")
    flow_open.selection().set(open_entities)

    ht = comp.physics().create("ht", "HeatTransferInFluids", "geom1")
    ht.label("Target-branch external plume heat transfer")
    ht.feature("init1").set("Tinit", "T_amb")
    nozzle_temperature = ht.create("temp_nozzle", "TemperatureBoundary", 1)
    nozzle_temperature.label("Training-calibrated nozzle temperature")
    nozzle_temperature.selection().named("geom1_sel_nozzle_in")
    nozzle_temperature.set("T0", "T_amb+(T_exit_eff-T_amb)*nozzle_shape")
    thermal_open = ht.create("open1", "OpenBoundary", 1)
    thermal_open.label("Ambient thermal opening")
    thermal_open.selection().set(open_entities)
    thermal_open.set("Tustr", "T_amb")
    target_temperature = ht.create("temp_target", "TemperatureBoundary", 1)
    target_temperature.label("Measured-range isothermal workpiece")
    target_temperature.selection().named("geom1_sel_target")
    target_temperature.set("T0", "T_target")

    coupling = comp.multiphysics().create("nitf1", "NonIsothermalFlow", 2)
    coupling.label("Target-branch nonisothermal turbulent coupling")
    coupling.set("Fluid_physics", "spf")
    coupling.set("Heat_physics", "ht")
    coupling.set("includeViscousDissipation", True)
    coupling.set("includeKineticEnergy", True)

    mesh = comp.mesh().create("mesh1", "geom1")
    mesh.label("Provisional target-plume mesh - convergence required")
    mesh.autoMeshSize(4)

    study = jm.study().create("std1")
    study.label("LOCKED target-impact stationary study")
    study.create("stat", "Stationary")
    return model, jm, material_audit


def audit_model(
    jm: Any,
    contract: TargetImpactPhysicsContract,
    material_audit: dict[str, Any],
) -> dict[str, Any]:
    comp = jm.component("comp1")
    physics = {str(tag): str(comp.physics(tag).getType()) for tag in comp.physics().tags()}
    multiphysics = {
        str(tag): str(comp.multiphysics(tag).getType()) for tag in comp.multiphysics().tags()
    }
    if physics != {"spf": "TurbulentFlowkomega", "ht": "HeatTransferInFluids"}:
        raise RuntimeError(f"Unexpected target physics tree: {physics}")
    if multiphysics.get("nitf1") != "NonIsothermalFlow":
        raise RuntimeError(f"Missing nonisothermal coupling: {multiphysics}")

    target_entities = _selection_entities(comp, "geom1_sel_target")
    if len(target_entities) != 2:
        raise RuntimeError(f"Target selection must contain two split boundaries: {target_entities}")
    selection_tags = {str(tag) for tag in comp.selection().tags()}
    if any("dpv" in tag.lower() for tag in selection_tags):
        raise RuntimeError(f"DPV selection leaked into target branch: {selection_tags}")

    return {
        "schema_version": "h11_target_impact_physics_skeleton_v1",
        "status": "pass_skeleton_solve_locked",
        "contract": asdict(contract),
        "solve_unlocked": contract.solve_unlocked,
        "paper_use_unlocked": contract.paper_use_unlocked,
        "geometry_semantics": {
            "workpiece_present": True,
            "target_coordinate": "z=d_spray",
            "dpv_coordinate_present": False,
            "target_boundary_entities": target_entities,
            "target_temperature_range_c": list(contract.target_temperature_range_c),
        },
        "physics": physics,
        "multiphysics": multiphysics,
        "flow_formulation": {
            "compressibility": "WeaklyCompressible",
            "reference_temperature": "T_amb",
            "reference_pressure": "p_amb",
            "reason": (
                "Temperature-dependent argon density is incompatible with "
                "the incompressible continuity equation."
            ),
        },
        "material": material_audit,
        "h2_to_argon_setting_ratio_range": list(
            contract.h2_to_argon_setting_ratio_range()
        ),
        "h2_ratio_status": "setting sensitivity only; not mass/mole fraction",
        "carrier_gas_status": "excluded by operator instruction",
        "effective_inlet_profile": {
            "expression": "nozzle_shape=(1-(r/r_exit_eff)^2)^profile_power",
            "centerline_state": ["u_exit_eff", "T_exit_eff"],
            "edge_state": ["0 m/s", "T_amb"],
            "profile_power_status": "training-calibrated sensitivity parameter",
        },
        "next_gate": (
            "Add training-only effective-boundary calibration, then mesh/domain/"
            "mass/energy convergence before particle release."
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

    contract = TargetImpactPhysicsContract()
    contract.validate()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    print("H11 layer 5: nonisothermal target-impact physics skeleton")
    client = mph.start(cores=args.cores, version=args.version)
    print(f"COMSOL {client.version}, cores={client.cores}, standalone={client.standalone}")
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
            "source_target_geometry": str(TARGET_GEOMETRY_MODEL),
            "source_target_geometry_sha256": _sha256(TARGET_GEOMETRY_MODEL),
            "argon_reference_model": str(ARGON_REFERENCE_MODEL),
            "argon_reference_sha256": _sha256(ARGON_REFERENCE_MODEL),
        }
    )
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print("Target physics skeleton gate: PASS; solve remains LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
