
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pu_dcgp_comsol.comsol.external_plume_skeleton import (
    ARGON_REFERENCE_MODEL,
    _copy_argon_material,
    _create_box_selection,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_conservative_free_jet_skeleton"
OUT_DIR = HERE / "h11_outputs" / "conservative_free_jet_skeleton"
MODEL_PATH = MODEL_DIR / "h11_conservative_free_jet_skeleton_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_conservative_free_jet_skeleton_audit.json"


@dataclass(frozen=True)
class ConservativeFreeJetContract:

    dpv_plane_mm: float = 100.0
    radial_domain_mm: float = 40.0
    axial_domain_mm: float = 140.0
    effective_exit_radius_mm: float = 4.0
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
    workpiece_present: bool = False
    spray_distance_in_equations: bool = False
    upstream_annulus_is_stationary_adiabatic_wall: bool = True
    radial_far_field_is_open: bool = True
    downstream_far_field_is_open: bool = True
    carrier_gas_included: bool = False
    ar_h2_composition_basis_compatible: bool = False
    h2_mixture_included: bool = False
    effective_exit_calibrated: bool = False
    particle_population_attached: bool = False
    dpv_observation_operator_attached: bool = False
    grouped_held_out_validation_passed: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        positive = (
            self.dpv_plane_mm,
            self.radial_domain_mm,
            self.axial_domain_mm,
            self.effective_exit_radius_mm,
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
            raise ValueError("Free-jet numerical inputs must be finite and positive")
        if not math.isclose(self.dpv_plane_mm, 100.0, abs_tol=1e-12):
            raise ValueError("The A-group free-jet branch requires the fixed 100 mm DPV plane")
        if self.axial_domain_mm <= self.dpv_plane_mm:
            raise ValueError("The downstream far field must lie beyond the DPV plane")
        if self.effective_exit_radius_mm >= self.radial_domain_mm:
            raise ValueError("The effective exit must lie inside the radial domain")
        if self.provisional_exit_temperature_k <= self.ambient_temperature_k:
            raise ValueError("The provisional exit temperature must exceed ambient")
        if not 0 < self.turbulence_intensity < 1:
            raise ValueError("Turbulence intensity must lie in (0, 1)")
        if self.radial_profile_power < 1:
            raise ValueError("The radial profile power must be at least one")
        if self.property_temperature_floor_k < 250:
            raise ValueError("The temperature floor is below the source table")
        if self.representative_exit_cp_j_kg_k <= self.argon_specific_gas_constant_j_kg_k:
            raise ValueError("Representative Cp must exceed the specific gas constant")
        if self.workpiece_present or self.spray_distance_in_equations:
            raise ValueError("Workpiece spray distance is forbidden in the DPV free-jet branch")
        if not (
            self.upstream_annulus_is_stationary_adiabatic_wall
            and self.radial_far_field_is_open
            and self.downstream_far_field_is_open
        ):
            raise ValueError("The free-jet boundary policy is incomplete")
        if self.carrier_gas_included:
            raise ValueError("The unidentified carrier gas must remain excluded")
        if self.h2_mixture_included and not self.ar_h2_composition_basis_compatible:
            raise ValueError("An Ar-H2 mixture requires a compatible composition basis")
        if self.h2_mixture_included:
            raise ValueError("The solve-locked skeleton cannot enable an Ar-H2 mixture")
        if any(
            (
                self.effective_exit_calibrated,
                self.particle_population_attached,
                self.dpv_observation_operator_attached,
                self.grouped_held_out_validation_passed,
                self.paper_prediction_allowed,
            )
        ):
            raise ValueError("The gas-phase skeleton cannot claim downstream completion")

    @property
    def argon_specific_gas_constant_j_kg_k(self) -> float:
        return 8.31446261815324 / self.argon_molar_mass_kg_mol

    @property
    def representative_gamma(self) -> float:
        cp = self.representative_exit_cp_j_kg_k
        rs = self.argon_specific_gas_constant_j_kg_k
        return cp / (cp - rs)

    @property
    def provisional_centerline_mach(self) -> float:
        sound_speed = math.sqrt(
            self.representative_gamma
            * self.argon_specific_gas_constant_j_kg_k
            * self.provisional_exit_temperature_k
        )
        return self.provisional_exit_velocity_m_s / sound_speed

    def unresolved_scientific_gates(self) -> list[str]:
        return [
            "validated_ar_h2_or_declared_pure_ar_sensitivity_model",
            "jointly_calibrated_effective_exit",
            "validated_7ysz_particle_population",
            "dpv_detection_and_aggregation_operator",
            "grouped_held_out_a_group_validation",
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _selection_entities(component: Any, tag: str) -> list[int]:
    return sorted(int(value) for value in component.selection(tag).entities())


def _create_integration(
    component: Any,
    tag: str,
    label: str,
    selection: Sequence[int],
) -> None:
    operator = component.cpl().create(tag, "Integration")
    operator.label(label)
    operator.selection().geom("geom1", 1)
    operator.selection().set([int(value) for value in selection])


def validate_boundary_role_sets(
    roles: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:

    required = (
        "nozzle",
        "torch_face",
        "radial_open",
        "downstream_open",
        "axis",
        "dpv_internal",
    )
    missing = [name for name in required if name not in roles]
    if missing:
        raise ValueError(f"Missing boundary roles: {missing}")
    normalized = {
        name: sorted({int(value) for value in roles[name]}) for name in required
    }
    empty = [name for name, values in normalized.items() if not values]
    if empty:
        raise ValueError(f"Empty boundary roles: {empty}")

    exterior_names = (
        "nozzle",
        "torch_face",
        "radial_open",
        "downstream_open",
        "axis",
    )
    for index, left_name in enumerate(exterior_names):
        left = set(normalized[left_name])
        for right_name in exterior_names[index + 1 :]:
            overlap = left & set(normalized[right_name])
            if overlap:
                raise ValueError(
                    f"Exterior boundary roles overlap: {left_name}, "
                    f"{right_name}: {sorted(overlap)}"
                )
    exterior = set().union(*(set(normalized[name]) for name in exterior_names))
    dpv_overlap = exterior & set(normalized["dpv_internal"])
    if dpv_overlap:
        raise ValueError(
            f"The DPV observation section became an exterior condition: "
            f"{sorted(dpv_overlap)}"
        )
    return normalized


def build_model(
    client: Any,
    contract: ConservativeFreeJetContract,
) -> tuple[Any, Any, dict[str, Any]]:
    contract.validate()
    if not ARGON_REFERENCE_MODEL.exists():
        raise FileNotFoundError(ARGON_REFERENCE_MODEL)

    reference = client.load(str(ARGON_REFERENCE_MODEL))
    model = client.create("h11_conservative_free_jet_skeleton")
    jm = model.java
    params = jm.param()

    parameters = (
        ("z_dpv", f"{contract.dpv_plane_mm:.12g}[mm]", "Fixed A-group DPV plane"),
        (
            "r_domain",
            f"{contract.radial_domain_mm:.12g}[mm]",
            "Initial radial far field; domain audit required",
        ),
        (
            "z_domain",
            f"{contract.axial_domain_mm:.12g}[mm]",
            "Initial downstream far field; domain audit required",
        ),
        (
            "r_exit_eff",
            f"{contract.effective_exit_radius_mm:.12g}[mm]",
            "Uncalibrated effective exit radius",
        ),
        ("T_amb", f"{contract.ambient_temperature_k:.12g}[K]", "Ambient temperature"),
        ("p_amb", f"{contract.ambient_pressure_pa:.12g}[Pa]", "Ambient pressure"),
        (
            "T_exit_eff",
            f"{contract.provisional_exit_temperature_k:.12g}[K]",
            "Training-only effective exit centerline temperature",
        ),
        (
            "u_exit_eff",
            f"{contract.provisional_exit_velocity_m_s:.12g}[m/s]",
            "Training-only effective exit centerline speed",
        ),
        (
            "M_Ar",
            f"{contract.argon_molar_mass_kg_mol:.12g}[kg/mol]",
            "Argon molar mass",
        ),
        ("R_Ar", "R_const/M_Ar", "Argon specific gas constant"),
        (
            "I_turb",
            f"{contract.turbulence_intensity:.12g}",
            "Effective-exit turbulence intensity sensitivity",
        ),
        (
            "L_turb",
            f"{contract.turbulence_length_mm:.12g}[mm]",
            "Effective-exit turbulence length sensitivity",
        ),
        (
            "profile_power",
            f"{contract.radial_profile_power:.12g}",
            "Training-only radial profile exponent",
        ),
        ("sel_tol", "0.01[mm]", "Coordinate selection tolerance"),
    )
    for name, value, description in parameters:
        _set_param(params, name, value, description)

    component = jm.component().create("comp1", True)
    component.label("Gun-attached A-group free jet")
    geometry = component.geom().create("geom1", 2)
    geometry.label("Axisymmetric free jet with retained DPV observation plane")
    geometry.axisymmetric(True)

    rectangles = (
        ("up_core", ["r_exit_eff", "z_dpv"], ["0", "0"], "Upstream jet core"),
        (
            "up_outer",
            ["r_domain-r_exit_eff", "z_dpv"],
            ["r_exit_eff", "0"],
            "Upstream outer gas",
        ),
        (
            "down_core",
            ["r_exit_eff", "z_domain-z_dpv"],
            ["0", "z_dpv"],
            "Downstream jet core",
        ),
        (
            "down_outer",
            ["r_domain-r_exit_eff", "z_domain-z_dpv"],
            ["r_exit_eff", "z_dpv"],
            "Downstream outer gas",
        ),
    )
    for tag, size, position, label in rectangles:
        rectangle = geometry.feature().create(tag, "Rectangle")
        rectangle.label(label)
        rectangle.set("size", size)
        rectangle.set("pos", position)
    union = geometry.feature().create("uni1", "Union")
    union.label("Continuous gas domain with retained DPV and mesh partitions")
    union.selection("input").set([row[0] for row in rectangles])
    union.set("intbnd", True)

    _create_box_selection(
        geometry,
        "sel_nozzle_in",
        "Finite effective nozzle exit",
        xmin="-sel_tol",
        xmax="r_exit_eff+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_box_selection(
        geometry,
        "sel_torch_face",
        "Stationary adiabatic torch face outside nozzle",
        xmin="r_exit_eff-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_box_selection(
        geometry,
        "sel_dpv",
        "Fixed internal DPV observation cross-section",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="z_dpv-sel_tol",
        ymax="z_dpv+sel_tol",
    )
    _create_box_selection(
        geometry,
        "sel_axis",
        "Axis of symmetry",
        xmin="-sel_tol",
        xmax="sel_tol",
        ymin="-sel_tol",
        ymax="z_domain+sel_tol",
    )
    _create_box_selection(
        geometry,
        "sel_far_r",
        "Radial pressure-temperature far field",
        xmin="r_domain-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="z_domain+sel_tol",
    )
    _create_box_selection(
        geometry,
        "sel_far_z",
        "Downstream pressure-temperature far field",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="z_domain-sel_tol",
        ymax="z_domain+sel_tol",
    )
    geometry.run()

    material_audit = _copy_argon_material(reference.java, component)
    material_audit["density_usage"] = (
        "The copied 1-atm density table is provenance only; the conservative "
        "interface uses rho=p/(R_Ar*T)."
    )
    material_audit["composition_status"] = (
        "Pure-Ar numerical sensitivity model. The Ar DOE standard volumetric "
        "flow and reported H2 mass flow do not define a compatible mixture fraction."
    )

    profile = component.variable().create("var_free_jet_inlet")
    profile.label("Conservative effective-exit profile")
    profile.selection().geom("geom1", 1)
    profile.selection().named("geom1_sel_nozzle_in")
    profile.set(
        "nozzle_shape",
        "(1-(r/r_exit_eff)^2)^profile_power",
        "Smooth profile equal to one on axis and zero at the effective edge",
    )

    hmnf = component.physics().create(
        "hmnf",
        "HighMachNumberFlowTurbulentkomega",
        "geom1",
    )
    hmnf.label("Conservative compressible A-group free jet, k-omega")
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
    nozzle.label("Uncalibrated subsonic effective exit")
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

    radial_open = _selection_entities(component, "geom1_sel_far_r")
    downstream_open = _selection_entities(component, "geom1_sel_far_z")
    open_entities = sorted({*radial_open, *downstream_open})
    far_field = hmnf.create("far_open", "HighMachNumberFlowInlet", 1)
    far_field.label("Radial and downstream pressure-temperature openings")
    far_field.selection().set(open_entities)
    far_field.set("FlowCondition", "Subsonic")
    far_field.set("BoundaryCondition", "Pressure")
    far_field.set("p0", "p_amb")
    far_field.set("TemperatureHeatflux", "Temperature")
    far_field.set("T0", "T_amb")
    far_field.set("RANSVarOption", "SpecifyTurbulentLengthScaleAndIntensity")
    far_field.set("IT", "I_turb")
    far_field.set("LT", "L_turb")
    far_field.set("SuppressBackflow", "0")

    _create_integration(
        component,
        "int_nozzle_hmnf",
        "Axisymmetric nozzle-flux integral",
        _selection_entities(component, "geom1_sel_nozzle_in"),
    )
    _create_integration(
        component,
        "int_open_hmnf",
        "Axisymmetric radial and downstream far-field integral",
        open_entities,
    )
    _create_integration(
        component,
        "int_torch_hmnf",
        "Axisymmetric torch-face integral",
        _selection_entities(component, "geom1_sel_torch_face"),
    )
    _create_integration(
        component,
        "int_dpv_hmnf",
        "Axisymmetric fixed-DPV-plane integral",
        _selection_entities(component, "geom1_sel_dpv"),
    )

    mesh = component.mesh().create("mesh1", "geom1")
    mesh.label("Initial free-jet mesh; formal mesh and domain ladders required")
    mesh.autoMeshSize(4)

    study = jm.study().create("std1")
    study.label("LOCKED conservative free-jet stationary study")
    study.create("stat", "Stationary")
    return model, jm, material_audit


def audit_model(
    jm: Any,
    contract: ConservativeFreeJetContract,
    material_audit: dict[str, Any],
) -> dict[str, Any]:
    component = jm.component("comp1")
    geometry = component.geom("geom1")
    hmnf = component.physics("hmnf")
    physical = hmnf.prop("PhysicalModelProperty")
    fluid = hmnf.feature("fluid1")
    physics = {
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
    required_physics = {
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
    if physics != required_physics:
        raise RuntimeError(f"Conservative free-jet physics drifted: {physics}")

    roles = validate_boundary_role_sets(
        {
            "nozzle": _selection_entities(component, "geom1_sel_nozzle_in"),
            "torch_face": _selection_entities(component, "geom1_sel_torch_face"),
            "radial_open": _selection_entities(component, "geom1_sel_far_r"),
            "downstream_open": _selection_entities(component, "geom1_sel_far_z"),
            "axis": _selection_entities(component, "geom1_sel_axis"),
            "dpv_internal": _selection_entities(component, "geom1_sel_dpv"),
        }
    )
    expected_features = {
        "nozzle": roles["nozzle"],
        "far_open": sorted({*roles["radial_open"], *roles["downstream_open"]}),
        "wall": roles["torch_face"],
        "thermal_insulation": roles["torch_face"],
    }
    observed_features = {
        "nozzle": sorted(
            int(value) for value in hmnf.feature("nozzle_in").selection().entities()
        ),
        "far_open": sorted(
            int(value) for value in hmnf.feature("far_open").selection().entities()
        ),
        "wall": sorted(
            int(value) for value in hmnf.feature("wallbc1").selection().entities()
        ),
        "thermal_insulation": sorted(
            int(value) for value in hmnf.feature("ins1").selection().entities()
        ),
    }
    if observed_features != expected_features:
        raise RuntimeError(
            "Free-jet boundary features do not match the physical boundary "
            f"contract: {observed_features} != {expected_features}"
        )

    parameters = {str(value) for value in jm.param().varnames()}
    forbidden = {"spray_distance", "L_spray", "T_target"}
    present_forbidden = sorted(parameters & forbidden)
    if present_forbidden:
        raise RuntimeError(
            f"Workpiece parameters leaked into the DPV branch: {present_forbidden}"
        )

    return {
        "schema_version": "h11_conservative_free_jet_skeleton_v1",
        "status": "pass_skeleton_solve_locked",
        "contract": asdict(contract),
        "coordinate_semantics": {
            "frame": "gun_attached_axisymmetric_rz",
            "dpv_plane": "fixed internal cross-section at z=100 mm",
            "workpiece": "absent",
            "gun_to_workpiece_spray_distance": "absent from geometry and equations",
        },
        "geometry": {
            "domain_count": int(geometry.getNDomains()),
            "boundary_count": int(geometry.getNBoundaries()),
            "roles": roles,
        },
        "physics": physics,
        "material": material_audit,
        "boundary_features": observed_features,
        "provisional_centerline_mach": contract.provisional_centerline_mach,
        "gas_phase_scope": (
            "calorically imperfect pure-Ar all-Mach sensitivity model; not an "
            "internal arc reconstruction and not yet a validated Ar-H2 model"
        ),
        "integral_operators": [
            "int_nozzle_hmnf",
            "int_open_hmnf",
            "int_torch_hmnf",
            "int_dpv_hmnf",
        ],
        "unresolved_scientific_gates": contract.unresolved_scientific_gates(),
        "calibrated": False,
        "paper_prediction_allowed": False,
        "next_gate": (
            "Run a bounded continuation solve, verify mass and total-energy "
            "closure, and establish mesh plus radial/axial domain independence "
            "before attaching the particle population."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    contract = ConservativeFreeJetContract()
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
            "runtime_sec": time.time() - started,
            "comsol_version": args.version,
            "cores": args.cores,
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
    print("Conservative free-jet skeleton: PASS; solve and paper prediction remain LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
