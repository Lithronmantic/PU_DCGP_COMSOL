"""H11 layer 2: paper-oriented external-plume COMSOL skeleton.

The executed experiment does not contain the torch/nozzle drawing, arc
voltage, or a measured nozzle-exit state.  A full internal DC-arc model would
therefore be underidentified.  This layer builds the highest model level that
is currently defensible:

* gun-attached 2-D axisymmetric external free-jet geometry;
* a finite effective nozzle core and a separate ambient opening;
* turbulent k-omega flow, heat transfer in fluids, and nonisothermal coupling;
* the fixed DPV cross-section at ``z_dpv``;
* temperature-dependent 1-atm argon properties copied from COMSOL's official
  ``icp_torch`` application-library model.

The artifact is deliberately solve-locked.  Nominal inlet values only make
the COMSOL tree complete enough for inspection; they are not calibrated
physics inputs and are never evaluated as experimental predictions.  The
next layer may unlock the stationary solve only after the entry gates in
``PlumeEntryContract`` are satisfied.
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


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_external_plume_skeleton"
OUT_DIR = HERE / "h11_outputs" / "external_plume_skeleton"
MODEL_PATH = MODEL_DIR / "h11_external_plume_skeleton_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_external_plume_skeleton_audit.json"
ARGON_REFERENCE_MODEL = Path(
    r"E:\COMSOL63\Multiphysics\applications\Plasma_Module"
    r"\Equilibrium_Discharges\icp_torch.mph"
)


@dataclass(frozen=True)
class PlumeEntryContract:
    """Traceable inputs and explicit solve lock for the external plume."""

    z_dpv_mm: float = 100.0
    radial_domain_mm: float = 40.0
    axial_domain_mm: float = 140.0
    effective_exit_radius_mm: float = 4.0
    ambient_temperature_k: float = 300.0
    ambient_pressure_pa: float = 101325.0
    nominal_exit_temperature_k: float = 10000.0
    nominal_exit_velocity_m_s: float = 600.0
    nominal_turbulence_intensity: float = 0.05
    nominal_turbulence_length_mm: float = 0.4
    h2_unit_confirmed: bool = False
    standard_flow_reference_confirmed: bool = False
    exit_radius_traceable: bool = False
    exit_energy_state_traceable: bool = False
    workpiece_branch_resolved: bool = False

    def validate(self) -> None:
        if self.z_dpv_mm <= 0:
            raise ValueError("z_dpv_mm must be positive")
        if self.axial_domain_mm <= self.z_dpv_mm:
            raise ValueError("axial_domain_mm must extend beyond z_dpv_mm")
        if not 0 < self.effective_exit_radius_mm < self.radial_domain_mm:
            raise ValueError("effective_exit_radius_mm must be inside the radial domain")
        if self.ambient_temperature_k <= 0:
            raise ValueError("ambient_temperature_k must be positive")
        if self.ambient_pressure_pa <= 0:
            raise ValueError("ambient_pressure_pa must be positive")
        if self.nominal_exit_temperature_k <= self.ambient_temperature_k:
            raise ValueError("nominal exit temperature must exceed ambient")
        if self.nominal_exit_velocity_m_s <= 0:
            raise ValueError("nominal exit velocity must be positive")
        if not 0 < self.nominal_turbulence_intensity < 1:
            raise ValueError("turbulence intensity must be in (0, 1)")
        if self.nominal_turbulence_length_mm <= 0:
            raise ValueError("turbulence length must be positive")

    @property
    def solve_unlocked(self) -> bool:
        return all(
            (
                self.h2_unit_confirmed,
                self.standard_flow_reference_confirmed,
                self.exit_radius_traceable,
                self.exit_energy_state_traceable,
                self.workpiece_branch_resolved,
            )
        )

    def unresolved_gates(self) -> list[str]:
        gates = {
            "hydrogen_unit": self.h2_unit_confirmed,
            "standard_flow_reference": self.standard_flow_reference_confirmed,
            "nozzle_or_effective_exit_radius": self.exit_radius_traceable,
            "arc_power_or_measured_exit_state": self.exit_energy_state_traceable,
            "a_workpiece_geometry_branch": self.workpiece_branch_resolved,
        }
        return [name for name, passed in gates.items() if not passed]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _create_box_selection(
    geom: Any,
    tag: str,
    label: str,
    *,
    xmin: str,
    xmax: str,
    ymin: str,
    ymax: str,
) -> None:
    selection = geom.feature().create(tag, "BoxSelection")
    selection.label(label)
    selection.set("entitydim", "1")
    selection.set("condition", "inside")
    selection.set("xmin", xmin)
    selection.set("xmax", xmax)
    selection.set("ymin", ymin)
    selection.set("ymax", ymax)


def _java_string_matrix(matrix: Any) -> list[list[str]]:
    return [[str(cell) for cell in row] for row in matrix]


def _copy_argon_material(source_java: Any, target_comp: Any) -> dict[str, Any]:
    """Copy COMSOL's official temperature-dependent argon table into target."""

    source_comp = source_java.component("comp1")
    source_material = source_comp.material("mat4")
    if str(source_material.label()) != "Argon (1[atm])":
        raise RuntimeError("Official reference model mat4 is no longer Argon (1[atm])")

    source_group = source_material.propertyGroup("def")
    target_material = target_comp.material().create("mat_ar", "Common")
    target_material.label("Argon (1[atm]) - copied from COMSOL icp_torch")
    target_material.selection().all()
    target_group = target_material.propertyGroup("def")

    function_audit: list[dict[str, Any]] = []
    for source_tag in source_group.func().tags():
        source_function = source_group.func(source_tag)
        function_name = str(source_function.getString("funcname"))
        target_tag = f"src_{function_name}"
        target_function = target_group.func().create(target_tag, "Interpolation")
        table = _java_string_matrix(source_function.getStringMatrix("table"))
        target_function.set("funcname", function_name)
        target_function.set("source", "table")
        target_function.set("table", table)
        target_function.set("interp", str(source_function.getString("interp")))
        source_extrapolation = str(source_function.getString("extrap"))
        target_function.set("extrap", "const")
        target_function.set("argunit", [str(v) for v in source_function.getStringArray("argunit")])
        target_function.set("fununit", [str(v) for v in source_function.getStringArray("fununit")])
        temperatures = [float(row[0]) for row in table]
        function_audit.append(
            {
                "name": function_name,
                "rows": len(table),
                "temperature_min_k": min(temperatures),
                "temperature_max_k": max(temperatures),
                "interpolation": str(source_function.getString("interp")),
                "source_extrapolation": source_extrapolation,
                "applied_extrapolation": "const",
            }
        )

    for property_name in (
        "density",
        "heatcapacity",
        "thermalconductivity",
        "dynamicviscosity",
        "ratioofspecificheat",
    ):
        values = [str(v) for v in source_group.getStringArray(property_name)]
        target_group.set(property_name, values)

    return {
        "label": str(source_material.label()),
        "functions": function_audit,
        "copied_properties": [
            "density",
            "heatcapacity",
            "thermalconductivity",
            "dynamicviscosity",
            "ratioofspecificheat",
        ],
    }


def _selection_entities(comp: Any, tag: str) -> list[int]:
    return [int(value) for value in comp.selection(tag).entities()]


def build_model(client: Any, contract: PlumeEntryContract) -> tuple[Any, Any, dict[str, Any]]:
    contract.validate()
    if not ARGON_REFERENCE_MODEL.exists():
        raise FileNotFoundError(ARGON_REFERENCE_MODEL)

    reference = client.load(str(ARGON_REFERENCE_MODEL))
    model = client.create("h11_external_plume_skeleton")
    jm = model.java
    params = jm.param()

    _set_param(params, "z_dpv", f"{contract.z_dpv_mm:.9g}[mm]", "Fixed gun-frame DPV plane")
    _set_param(params, "r_domain", f"{contract.radial_domain_mm:.9g}[mm]", "Provisional radial far field")
    _set_param(params, "z_domain", f"{contract.axial_domain_mm:.9g}[mm]", "Provisional axial far field")
    _set_param(
        params,
        "r_exit_eff",
        f"{contract.effective_exit_radius_mm:.9g}[mm]",
        "Provisional effective exit radius; sensitivity parameter, not apparatus truth",
    )
    _set_param(params, "T_amb", f"{contract.ambient_temperature_k:.9g}[K]", "Ambient temperature")
    _set_param(params, "p_amb", f"{contract.ambient_pressure_pa:.9g}[Pa]", "Ambient absolute pressure")
    _set_param(
        params,
        "T_exit_eff",
        f"{contract.nominal_exit_temperature_k:.9g}[K]",
        "Nominal solve-locked exit temperature; training-only calibration parameter",
    )
    _set_param(
        params,
        "u_exit_eff",
        f"{contract.nominal_exit_velocity_m_s:.9g}[m/s]",
        "Nominal solve-locked exit speed; training-only calibration parameter",
    )
    _set_param(
        params,
        "I_turb",
        f"{contract.nominal_turbulence_intensity:.9g}",
        "Nominal inlet turbulence intensity; sensitivity parameter",
    )
    _set_param(
        params,
        "L_turb",
        f"{contract.nominal_turbulence_length_mm:.9g}[mm]",
        "Nominal inlet turbulent length scale; sensitivity parameter",
    )
    _set_param(params, "sel_tol", "0.01[mm]", "Coordinate-selection tolerance")

    comp = jm.component().create("comp1", True)
    comp.label("Gun-attached external plasma plume")
    geom = comp.geom().create("geom1", 2)
    geom.label("Axisymmetric external plume with finite exit core")
    geom.axisymmetric(True)

    rectangles = (
        ("up_core", ["r_exit_eff", "z_dpv"], ["0", "0"], "Upstream plume core"),
        (
            "up_ambient",
            ["r_domain-r_exit_eff", "z_dpv"],
            ["r_exit_eff", "0"],
            "Upstream ambient",
        ),
        (
            "down_core",
            ["r_exit_eff", "z_domain-z_dpv"],
            ["0", "z_dpv"],
            "Downstream plume core",
        ),
        (
            "down_ambient",
            ["r_domain-r_exit_eff", "z_domain-z_dpv"],
            ["r_exit_eff", "z_dpv"],
            "Downstream ambient",
        ),
    )
    for tag, size, pos, label in rectangles:
        node = geom.feature().create(tag, "Rectangle")
        node.label(label)
        node.set("size", size)
        node.set("pos", pos)

    union = geom.feature().create("uni1", "Union")
    union.label("External plume envelope with retained DPV and core interfaces")
    union.selection("input").set([item[0] for item in rectangles])
    union.set("intbnd", True)

    _create_box_selection(
        geom,
        "sel_nozzle_in",
        "Finite effective nozzle inlet",
        xmin="-sel_tol",
        xmax="r_exit_eff+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_ambient_in",
        "Upstream ambient opening outside nozzle",
        xmin="r_exit_eff-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_dpv",
        "Fixed DPV observation cross-section",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="z_dpv-sel_tol",
        ymax="z_dpv+sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_axis",
        "Axis of symmetry",
        xmin="-sel_tol",
        xmax="sel_tol",
        ymin="-sel_tol",
        ymax="z_domain+sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_far_r",
        "Radial far-field opening",
        xmin="r_domain-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="z_domain+sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_far_z",
        "Axial far-field opening",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="z_domain-sel_tol",
        ymax="z_domain+sel_tol",
    )
    geom.run()

    material_audit = _copy_argon_material(reference.java, comp)

    spf = comp.physics().create("spf", "TurbulentFlowkomega", "geom1")
    spf.label("External plume turbulent flow, k-omega")
    inlet = spf.create("inl1", "InletBoundary", 1)
    inlet.label("Effective nozzle exit state")
    inlet.selection().named("geom1_sel_nozzle_in")
    inlet.set("BoundaryCondition", "Velocity")
    inlet.set("U0in", "u_exit_eff")
    inlet.set("RANSVarOption", "SpecifyTurbulentLengthScaleAndIntensity")
    inlet.set("IT", "I_turb")
    inlet.set("LT", "L_turb")

    open_entities = sorted(
        set(
            _selection_entities(comp, "geom1_sel_ambient_in")
            + _selection_entities(comp, "geom1_sel_far_r")
            + _selection_entities(comp, "geom1_sel_far_z")
        )
    )
    flow_open = spf.create("open1", "OpenBoundary", 1)
    flow_open.label("Ambient entrainment and downstream open boundary")
    flow_open.selection().set(open_entities)

    ht = comp.physics().create("ht", "HeatTransferInFluids", "geom1")
    ht.label("External plume heat transfer")
    ht.feature("init1").set("Tinit", "T_amb")
    temperature = ht.create("temp1", "TemperatureBoundary", 1)
    temperature.label("Effective nozzle exit temperature")
    temperature.selection().named("geom1_sel_nozzle_in")
    temperature.set("T0", "T_exit_eff")
    thermal_open = ht.create("open1", "OpenBoundary", 1)
    thermal_open.label("Ambient thermal opening")
    thermal_open.selection().set(open_entities)

    coupling = comp.multiphysics().create("nitf1", "NonIsothermalFlow", 2)
    coupling.label("Nonisothermal turbulent plume coupling")
    coupling.set("Fluid_physics", "spf")
    coupling.set("Heat_physics", "ht")
    coupling.set("includeViscousDissipation", True)
    coupling.set("includeKineticEnergy", True)

    mesh = comp.mesh().create("mesh1", "geom1")
    mesh.label("Provisional plume mesh - convergence study required")
    mesh.autoMeshSize(4)

    study = jm.study().create("std1")
    study.label("LOCKED stationary external plume study")
    study.create("stat", "Stationary")

    return model, jm, material_audit


def audit_model(jm: Any, contract: PlumeEntryContract, material_audit: dict[str, Any]) -> dict[str, Any]:
    comp = jm.component("comp1")
    geometry = comp.geom("geom1")
    selections = {
        tag: _selection_entities(comp, f"geom1_{tag}")
        for tag in (
            "sel_nozzle_in",
            "sel_ambient_in",
            "sel_dpv",
            "sel_axis",
            "sel_far_r",
            "sel_far_z",
        )
    }
    if set(selections["sel_nozzle_in"]) & set(selections["sel_ambient_in"]):
        raise RuntimeError("Nozzle and ambient inlet selections overlap")
    if not selections["sel_dpv"]:
        raise RuntimeError("Fixed DPV cross-section selection is empty")

    physics_types = {
        str(tag): str(comp.physics(tag).getType()) for tag in comp.physics().tags()
    }
    multiphysics_types = {
        str(tag): str(comp.multiphysics(tag).getType())
        for tag in comp.multiphysics().tags()
    }
    expected_physics = {
        "spf": "TurbulentFlowkomega",
        "ht": "HeatTransferInFluids",
    }
    if any(physics_types.get(tag) != value for tag, value in expected_physics.items()):
        raise RuntimeError(f"Unexpected physics tree: {physics_types}")
    if multiphysics_types.get("nitf1") != "NonIsothermalFlow":
        raise RuntimeError(f"Unexpected multiphysics tree: {multiphysics_types}")

    return {
        "schema_version": "h11_external_plume_skeleton_v1",
        "status": "pass_skeleton_solve_locked",
        "contract": asdict(contract),
        "solve_unlocked": contract.solve_unlocked,
        "unresolved_gates": contract.unresolved_gates(),
        "geometry": {
            "coordinate_frame": "gun_attached_axisymmetric_rz",
            "domain_count": int(geometry.getNDomains()),
            "boundary_count": int(geometry.getNBoundaries()),
            "fixed_dpv_mm": contract.z_dpv_mm,
            "spray_distance_in_free_jet_equations": False,
            "effective_exit_radius_status": "provisional_sensitivity_parameter",
            "workpiece_boundary_included": False,
            "selections": selections,
        },
        "physics": physics_types,
        "multiphysics": multiphysics_types,
        "material": material_audit,
        "nominal_inlet_values_status": (
            "tree-completion placeholders only; prohibited from prediction until solve gates pass"
        ),
        "next_gate": (
            "Resolve apparatus inputs, replace pure-argon baseline with an "
            "Ar-H2 mixture/sensitivity bracket, then run mesh/domain/conservation studies."
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

    contract = PlumeEntryContract()
    contract.validate()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    print("H11 layer 2: external plume COMSOL skeleton")
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
            "argon_reference_model": str(ARGON_REFERENCE_MODEL),
            "argon_reference_sha256": _sha256(ARGON_REFERENCE_MODEL),
        }
    )
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print("Skeleton gate: PASS; solve remains LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
