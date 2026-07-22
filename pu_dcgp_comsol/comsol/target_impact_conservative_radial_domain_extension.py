
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.target_impact_conservative_nominal import (
    ConservativeSolveContract,
    _set_manual_scales,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_residual_localization import (
    _replace_maximum_coupling,
    _residual_localization,
    parse_detailed_residual_log,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_restart import (
    _entities,
    _replace_coupling,
    evaluate_solution,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_same_mesh_refinement import (
    _bounded,
    _last_tag,
    _sha256,
    relative_change,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_domain"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_domain"
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_numerically_accepted.mph"
)


@dataclass(frozen=True)
class RadialDomainExtensionContract:

    source_radius_mm: float = 40.0
    target_radius_mm: float = 60.0
    initialization_buffer_mm: float = 0.01
    relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 4000
    flow_subiterations: int = 1
    turbulence_subiterations: int = 3
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        radii = (self.source_radius_mm, self.target_radius_mm)
        if any(not math.isfinite(value) or value <= 0 for value in radii):
            raise ValueError("Domain radii must be finite and positive")
        if self.target_radius_mm <= self.source_radius_mm:
            raise ValueError("Target radius must exceed source radius")
        if self.target_radius_mm > 100:
            raise ValueError("Radial-domain audit is limited to 100 mm")
        if (
            not math.isfinite(self.initialization_buffer_mm)
            or self.initialization_buffer_mm <= 0
            or self.initialization_buffer_mm >= self.source_radius_mm
        ):
            raise ValueError(
                "Initialization buffer must lie inside the source radius"
            )
        if self.relative_tolerance != 1e-6:
            raise ValueError("Domain solve tolerance must remain 1e-6")
        if self.maximum_segregated_iterations != 4000:
            raise ValueError("Domain solve iteration ceiling must remain 4000")
        if self.flow_subiterations != 1:
            raise ValueError("Flow subiterations must remain 1")
        if self.turbulence_subiterations != 3:
            raise ValueError("Turbulence subiterations must remain 3")
        if self.flow_damping != 0.1 or self.turbulence_damping != 0.1:
            raise ValueError("Frozen damping is 0.1")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if (
            not math.isfinite(self.pressure_floor_pa)
            or self.pressure_floor_pa <= 0
        ):
            raise ValueError("Pressure floor must be finite and positive")
        if self.mass_imbalance_limit_fraction != 0.005:
            raise ValueError("Mass gate must remain 0.5 percent")
        if self.energy_imbalance_limit_fraction != 0.02:
            raise ValueError("Energy gate must remain 2 percent")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Domain verification cannot claim prediction")


def _radius_token(radius_mm: float) -> str:
    rounded = round(radius_mm)
    if not math.isclose(radius_mm, rounded, abs_tol=1e-9):
        raise ValueError("Radial-domain artifacts require integer millimetres")
    return f"r{rounded:03d}"


def _default_artifact_paths(
    contract: RadialDomainExtensionContract,
    build_only: bool,
) -> tuple[Path, Path, Path]:
    stem = (
        "h11_target_impact_conservative_domain_"
        f"{_radius_token(contract.source_radius_mm)}_to_"
        f"{_radius_token(contract.target_radius_mm)}"
    )
    if build_only:
        stem = f"{stem}_skeleton"
    return (
        MODEL_DIR / f"{stem}.mph",
        OUT_DIR / f"{stem}.json",
        OUT_DIR / f"{stem}.log",
    )


def _mesh_controls(mesh: Any) -> dict[str, Any]:
    size = mesh.feature("size")
    size1 = mesh.feature("size1")
    blp1 = mesh.feature("bl1").feature("blp1")
    return {
        "global": {
            "hmax_m": float(str(size.getString("hmax"))),
            "hmin_m": float(str(size.getString("hmin"))),
            "hgrad": float(str(size.getString("hgrad"))),
            "hcurve": float(str(size.getString("hcurve"))),
            "hnarrow": float(str(size.getString("hnarrow"))),
            "custom": str(size.getString("custom")),
        },
        "target_local": {
            "hmax_m": float(str(size1.getString("hmax"))),
            "hmin_m": float(str(size1.getString("hmin"))),
            "hgrad": float(str(size1.getString("hgrad"))),
            "hcurve": float(str(size1.getString("hcurve"))),
        },
        "boundary_layer": {
            "layers": int(str(blp1.getString("blnlayers"))),
            "stretch": float(str(blp1.getString("blstretch"))),
            "initial_thickness_factor": float(
                str(blp1.getString("blhminfact"))
            ),
        },
    }


def _selection_snapshot(component: Any) -> dict[str, list[int]]:
    return {
        "nozzle": sorted(_entities(component, "geom1_sel_nozzle_in")),
        "torch_face": sorted(
            _entities(component, "geom1_sel_ambient_in")
        ),
        "axis": sorted(_entities(component, "geom1_sel_axis")),
        "radial_open": sorted(_entities(component, "geom1_sel_far_r")),
        "target": sorted(_entities(component, "geom1_sel_target")),
    }


def controlled_initial_expressions(
    source_solution_tag: str,
    contract: RadialDomainExtensionContract,
) -> dict[str, Any]:

    source_limit_mm = (
        contract.source_radius_mm - contract.initialization_buffer_mm
    )
    inside = f"r<={source_limit_mm:.12g}[mm]"

    def inherited(field: str, ambient: str) -> str:
        return (
            f"if({inside},withsol('{source_solution_tag}',{field}),"
            f"{ambient})"
        )

    return {
        "source_solution": source_solution_tag,
        "source_evaluation_limit_mm": source_limit_mm,
        "radial_velocity": inherited("u", "0[m/s]"),
        "out_of_plane_velocity": "0[m/s]",
        "axial_velocity": inherited("w", "0[m/s]"),

        "absolute_pressure": inherited("p", "p_amb"),
        "temperature": inherited("T", "T_amb"),
        "turbulent_kinetic_energy": inherited("k", "hmnf.kinit"),
        "specific_dissipation_rate": inherited("om", "hmnf.omInit"),
    }


def apply_controlled_initialization(
    jm: Any,
    *,
    source_solution_tag: str,
    contract: RadialDomainExtensionContract,
) -> dict[str, Any]:

    expressions = controlled_initial_expressions(
        source_solution_tag,
        contract,
    )
    initial = jm.component("comp1").physics("hmnf").feature("init1")
    initial.set(
        "u_init",
        [
            expressions["radial_velocity"],
            expressions["out_of_plane_velocity"],
            expressions["axial_velocity"],
        ],
    )
    initial.set("p_init", expressions["absolute_pressure"])
    initial.set("Tinit", expressions["temperature"])
    initial.set(
        "k_init",
        expressions["turbulent_kinetic_energy"],
    )
    initial.set(
        "om_init",
        expressions["specific_dissipation_rate"],
    )
    return expressions


def rebuild_radial_domain(
    jm: Any,
    contract: RadialDomainExtensionContract,
) -> dict[str, Any]:

    contract.validate()
    observed_source_mm = float(jm.param().evaluate("r_domain")) * 1e3
    if not math.isclose(
        observed_source_mm,
        contract.source_radius_mm,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Source radius is {observed_source_mm} mm, expected "
            f"{contract.source_radius_mm} mm"
        )

    component = jm.component("comp1")
    geometry = component.geom("geom1")
    mesh = component.mesh("mesh1")
    controls_before = _mesh_controls(mesh)
    topology_before = {
        "domains": int(geometry.getNDomains()),
        "boundaries": int(geometry.getNBoundaries()),
        "vertices": int(geometry.getNVertices()),
        "selections": _selection_snapshot(component),
    }

    jm.param().set(
        "r_domain",
        f"{contract.target_radius_mm:.12g}[mm]",
        "Radial far field for domain-independence audit",
    )
    geometry.run()
    selections = _selection_snapshot(component)
    topology_after = {
        "domains": int(geometry.getNDomains()),
        "boundaries": int(geometry.getNBoundaries()),
        "vertices": int(geometry.getNVertices()),
        "selections": selections,
        "bounding_box_m": [
            float(value) for value in geometry.getBoundingBox()
        ],
    }
    if (
        topology_after["domains"],
        topology_after["boundaries"],
        topology_after["vertices"],
    ) != (
        topology_before["domains"],
        topology_before["boundaries"],
        topology_before["vertices"],
    ):
        raise RuntimeError("Radial extension changed geometry topology")
    if selections != topology_before["selections"]:
        raise RuntimeError("Named boundary selections changed entity IDs")


    source_target_controls = controls_before["target_local"]
    target_size = mesh.feature("size1")
    target_size.set("custom", "on")
    target_size.set(
        "hmax",
        f"{source_target_controls['hmax_m']:.16g}",
    )
    target_size.set(
        "hmin",
        f"{source_target_controls['hmin_m']:.16g}",
    )
    target_size.set(
        "hgrad",
        f"{source_target_controls['hgrad']:.16g}",
    )
    target_size.set(
        "hcurve",
        f"{source_target_controls['hcurve']:.16g}",
    )
    target_size.set("hnarrow", "1")

    hmnf = component.physics("hmnf")
    walls = sorted({*selections["torch_face"], *selections["target"]})
    hmnf.feature("nozzle_in").selection().set(selections["nozzle"])
    hmnf.feature("ambient_open").selection().set(
        selections["radial_open"]
    )
    hmnf.feature("target_temperature").selection().set(
        selections["target"]
    )
    observed_default_walls = sorted(
        int(value)
        for value in hmnf.feature("wallbc1").selection().entities()
    )
    observed_axis = sorted(
        int(value)
        for value in hmnf.feature("axi1").selection().entities()
    )
    observed_insulation = sorted(
        int(value)
        for value in hmnf.feature("ins1").selection().entities()
    )
    if observed_default_walls != walls:
        raise RuntimeError(
            "Read-only default wall selection drifted: "
            f"{observed_default_walls} != {walls}"
        )
    if observed_axis != selections["axis"]:
        raise RuntimeError(
            "Read-only axis selection drifted: "
            f"{observed_axis} != {selections['axis']}"
        )
    if observed_insulation != selections["torch_face"]:
        raise RuntimeError(
            "Read-only insulation selection drifted: "
            f"{observed_insulation} != {selections['torch_face']}"
        )
    _replace_coupling(component, "int_nozzle_hmnf", selections["nozzle"])
    _replace_coupling(
        component,
        "int_ambient_hmnf",
        selections["radial_open"],
    )
    _replace_coupling(component, "int_target_hmnf", selections["target"])
    _replace_coupling(component, "int_wall_hmnf", walls)
    _replace_maximum_coupling(component, "max_res_domain", 2)
    _replace_maximum_coupling(
        component,
        "max_res_boundary",
        1,
        walls,
    )

    mesh.feature("size1").selection().set(selections["target"])
    mesh.feature("bl1").feature("blp1").selection().set(
        selections["target"]
    )
    mesh.run()
    controls_after = _mesh_controls(mesh)
    if controls_after != controls_before:
        raise RuntimeError(
            "Absolute mesh controls changed during extension: "
            f"before={controls_before}, after={controls_after}"
        )

    return {
        "source_radius_mm": observed_source_mm,
        "target_radius_mm": contract.target_radius_mm,
        "topology_before": topology_before,
        "topology_after": topology_after,
        "boundary_semantics": {
            "nozzle": selections["nozzle"],
            "torch_face_no_slip_adiabatic": selections["torch_face"],
            "axis": selections["axis"],
            "radial_pressure_temperature_opening": selections[
                "radial_open"
            ],
            "isothermal_target": selections["target"],
            "all_walls": walls,
        },
        "mesh_controls": controls_after,
        "target_local_mesh_control": (
            "source-radius evaluated physical sizes frozen explicitly"
        ),
        "mesh": {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        },
    }


def configure_domain_study(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: RadialDomainExtensionContract,
) -> tuple[str, str, dict[str, Any]]:
    target_token = _radius_token(contract.target_radius_mm)
    study_tag = f"std_hmnf_domain_{target_token}"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label(
        f"Radial-domain extension to {contract.target_radius_mm:g} mm"
    )
    step = study.create("stat", "Stationary")
    initial_expressions = apply_controlled_initialization(
        jm,
        source_solution_tag=source_solution_tag,
        contract=contract,
    )
    step.set("useinitsol", "off")
    step.set("initmethod", "init")
    study.createAutoSequences("all")
    solution_tag = _last_tag(jm.sol())
    variables = jm.sol(solution_tag).feature("v1")
    variables.set("initmethod", "init")
    if str(variables.getString("initmethod")) != "init":
        raise RuntimeError("Target variables did not select physics initial values")

    scales = _set_manual_scales(
        jm,
        solution_tag,
        ConservativeSolveContract(automatic_mesh_level=3),
    )
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{contract.relative_tolerance:.12g}")
    advanced = stationary.feature("aDef")
    advanced.set("storeresidual", "solvingandoutput")
    advanced.set("convinfo", "detailed")
    advanced.set("checkmatherr", "on")
    segregated = stationary.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.maximum_segregated_iterations),
    )
    segregated.set("segterm", "tol")
    segregated.set("segtermonres", "off")
    for tag, damping, iterations in (
        (
            "ss1",
            contract.flow_damping,
            contract.flow_subiterations,
        ),
        (
            "ss2",
            contract.turbulence_damping,
            contract.turbulence_subiterations,
        ),
    ):
        substep = segregated.feature(tag)
        substep.set("subdtech", "const")
        substep.set("subdamp", f"{damping:.12g}")
        substep.set("subtermconst", "iter")
        substep.set("subiter", str(iterations))
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "termination_criterion": str(
            segregated.getString("segtermonres")
        ),
        "relative_tolerance": float(
            str(stationary.getString("stol"))
        ),
        "maximum_outer_iterations": int(
            str(segregated.getString("maxsegiter"))
        ),
        "store_last_residual": str(
            advanced.getString("storeresidual")
        ),
        "convergence_log_level": str(
            advanced.getString("convinfo")
        ),
        "initialization_method": str(
            variables.getString("initmethod")
        ),
        "initial_expressions": initial_expressions,
    }


def _solution_dataset(model: Any, solution_tag: str) -> Any:
    for dataset in model / "datasets":
        properties = set(dataset.properties())
        if (
            "solution" in properties
            and str(dataset.property("solution")) == solution_tag
        ):
            return dataset
    raise RuntimeError(
        f"No solution dataset refers to solver sequence {solution_tag}"
    )


def _flat_floats(values: Any) -> list[float]:
    return [float(value) for value in values.reshape(-1)]


def _value_range(values: Any) -> dict[str, float]:
    flat = _flat_floats(values)
    return {"minimum": min(flat), "maximum": max(flat)}


def _point_values(
    model: Any,
    dataset: Any,
    expressions: list[str],
    units: list[str],
) -> dict[str, float]:
    values = model.evaluate(
        expressions,
        unit=units,
        dataset=dataset,
    )
    return {
        expression: _flat_floats(value)[0]
        for expression, value in zip(expressions, values)
    }


def probe_controlled_initialization(
    model: Any,
    *,
    solution_tag: str,
    source_solution_tag: str,
    contract: RadialDomainExtensionContract,
) -> dict[str, Any]:

    jm = model.java
    jm.sol(solution_tag).runFromTo("st1", "v1")
    dataset = _solution_dataset(model, solution_tag)
    temperature, pressure, speed, turbulent_energy, dissipation = (
        model.evaluate(
            ["T", "p", "hmnf.U", "k", "om"],
            unit=["K", "Pa", "m/s", "m^2/s^2", "1/s"],
            dataset=dataset,
        )
    )
    field_ranges = {
        "temperature_k": _value_range(temperature),
        "absolute_pressure_pa": _value_range(pressure),
        "speed_m_s": _value_range(speed),
        "turbulent_kinetic_energy_m2_s2": _value_range(
            turbulent_energy
        ),
        "specific_dissipation_rate_1_s": _value_range(dissipation),
    }

    inner_r_mm = 1e-3
    axial_mm = 50.0
    outer_r_mm = (
        contract.source_radius_mm + contract.target_radius_mm
    ) / 2
    fields = ("T", "p", "u", "w", "k", "om")
    units = ("K", "Pa", "m/s", "m/s", "m^2/s^2", "1/s")
    current_inner_expressions = [
        f"at2({inner_r_mm:.12g}[mm],{axial_mm:g}[mm],{field})"
        for field in fields
    ]
    source_inner_expressions = [
        (
            f"at2({inner_r_mm:.12g}[mm],{axial_mm:g}[mm],"
            f"withsol('{source_solution_tag}',{field}))"
        )
        for field in fields
    ]
    outer_expressions = [
        f"at2({outer_r_mm:.12g}[mm],{axial_mm:g}[mm],{field})"
        for field in fields
    ]
    current_inner = _point_values(
        model,
        dataset,
        current_inner_expressions,
        list(units),
    )
    source_inner = _point_values(
        model,
        dataset,
        source_inner_expressions,
        list(units),
    )
    outer = _point_values(
        model,
        dataset,
        outer_expressions,
        list(units),
    )
    inner_relative_errors = {}
    for field, current_expression, source_expression in zip(
        fields,
        current_inner_expressions,
        source_inner_expressions,
    ):
        current_value = current_inner[current_expression]
        source_value = source_inner[source_expression]
        scale = max(abs(source_value), 1e-12)
        inner_relative_errors[field] = abs(
            current_value - source_value
        ) / scale

    outer_by_field = {
        field: outer[expression]
        for field, expression in zip(fields, outer_expressions)
    }
    finite = all(
        math.isfinite(value)
        for field_range in field_ranges.values()
        for value in field_range.values()
    ) and all(math.isfinite(value) for value in outer_by_field.values())
    inherited = all(
        value <= 1e-8 for value in inner_relative_errors.values()
    )
    ambient_outer = bool(
        abs(outer_by_field["T"] - 300.0) <= 1e-6
        and abs(outer_by_field["p"] - 101_325.0) <= 1e-6
        and abs(outer_by_field["u"]) <= 1e-12
        and abs(outer_by_field["w"]) <= 1e-12
        and outer_by_field["k"] > 0
        and outer_by_field["om"] > 0
    )
    physically_admissible = bool(
        field_ranges["temperature_k"]["minimum"]
        >= contract.property_temperature_floor_k
        and field_ranges["absolute_pressure_pa"]["minimum"]
        >= contract.pressure_floor_pa
        and field_ranges["turbulent_kinetic_energy_m2_s2"]["minimum"]
        >= 0
        and field_ranges["specific_dissipation_rate_1_s"]["minimum"]
        >= 0
    )
    gates = {
        "all_initial_values_finite": finite,
        "source_domain_inherited_to_1e_minus_8": inherited,
        "new_annulus_is_ambient": ambient_outer,
        "initial_state_physically_admissible": physically_admissible,
    }
    return {
        "status": (
            "pass_controlled_cross_geometry_initialization"
            if all(gates.values())
            else "fail_controlled_cross_geometry_initialization"
        ),
        "field_ranges": field_ranges,
        "inner_probe": {
            "r_mm": inner_r_mm,
            "z_mm": axial_mm,
            "current": {
                field: current_inner[expression]
                for field, expression in zip(
                    fields,
                    current_inner_expressions,
                )
            },
            "source": {
                field: source_inner[expression]
                for field, expression in zip(
                    fields,
                    source_inner_expressions,
                )
            },
            "relative_errors": inner_relative_errors,
        },
        "outer_probe": {
            "r_mm": outer_r_mm,
            "z_mm": axial_mm,
            "values": outer_by_field,
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--source-radius-mm", type=float, default=40.0)
    parser.add_argument("--target-radius-mm", type=float, default=60.0)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.source_model.exists():
        raise FileNotFoundError(args.source_model)
    contract = RadialDomainExtensionContract(
        source_radius_mm=args.source_radius_mm,
        target_radius_mm=args.target_radius_mm,
    )
    contract.validate()
    default_model, default_audit, default_log = _default_artifact_paths(
        contract,
        args.build_only,
    )
    model_out = args.model_out or default_model
    audit_path = args.audit or default_audit
    log_path = args.log or default_log
    for path in (model_out, audit_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        client.java.showProgress(str(log_path.resolve()))
        model = client.load(str(args.source_model))
        jm = model.java
        source_study_tag = _last_tag(jm.study())
        source_solution_tag = _last_tag(jm.sol())
        source_metrics = evaluate_solution(model)
        source_mesh = {
            "elements": int(
                jm.component("comp1").mesh("mesh1").getNumElem()
            ),
            "vertices": int(
                jm.component("comp1").mesh("mesh1").getNumVertex()
            ),
        }
        extension = rebuild_radial_domain(jm, contract)
        study_tag, solution_tag, solver = configure_domain_study(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )

        solve_error = None
        initialization_probe = None
        if args.build_only:
            initialization_probe = probe_controlled_initialization(
                model,
                solution_tag=solution_tag,
                source_solution_tag=source_solution_tag,
                contract=contract,
            )
            if initialization_probe["status"].startswith("pass_"):
                status = (
                    "pass_radial_domain_skeleton_"
                    "controlled_initialization_verified"
                )
            else:
                status = "failed_radial_domain_skeleton_initialization"
            metrics: dict[str, Any] = {}
            residuals: dict[str, Any] = {}
            changes: dict[str, float] = {}
            gates = {
                "solver_returned_converged_solution": False,
                "physically_bounded": False,
                "mass_imbalance_below_0_5_percent": False,
                "energy_imbalance_below_2_percent": False,
            }
        else:
            try:
                jm.study(study_tag).run()
            except Exception as exc:
                solve_error = str(exc)
                status = "failed_radial_domain_equation_solve"
            else:
                status = "radial_domain_solve_completed"
            if solve_error is None:
                metrics = evaluate_solution(model)
                residuals = _residual_localization(model)
                source_near = source_metrics["one_mm_upstream_of_target"]
                target_near = metrics["one_mm_upstream_of_target"]
                changes = {
                    "near_target_temperature_fraction": relative_change(
                        source_near["temperature_k"],
                        target_near["temperature_k"],
                    ),
                    "near_target_speed_fraction": relative_change(
                        source_near["speed_m_s"],
                        target_near["speed_m_s"],
                    ),
                }
                bounded = _bounded(
                    metrics,
                    ConservativeSolveContract(automatic_mesh_level=3),
                )
                mass_pass = (
                    metrics["mass_flux_kg_s"]["imbalance_fraction"]
                    < contract.mass_imbalance_limit_fraction
                )
                energy_pass = (
                    metrics["energy_balance_w"][
                        "imbalance_fraction_of_inlet"
                    ]
                    < contract.energy_imbalance_limit_fraction
                )
                gates = {
                    "solver_returned_converged_solution": True,
                    "physically_bounded": bounded,
                    "mass_imbalance_below_0_5_percent": mass_pass,
                    "energy_imbalance_below_2_percent": energy_pass,
                }
                if all(gates.values()):
                    status = (
                        "pass_radial_domain_numerical_gates_"
                        "independence_comparison_pending"
                    )
            else:
                metrics = {}
                residuals = {}
                changes = {}
                gates = {
                    "solver_returned_converged_solution": False,
                    "physically_bounded": False,
                    "mass_imbalance_below_0_5_percent": False,
                    "energy_imbalance_below_2_percent": False,
                }

        model.rename(model_out.stem)
        model.save(str(model_out))
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        result = {
            "schema_version": "h11_radial_domain_extension_v1",
            "status": status,
            "contract": asdict(contract),
            "strategy": {
                "only_radial_domain_changed": True,
                "absolute_mesh_controls_changed": False,
                "physics_changed": False,
                "material_changed": False,
                "boundary_semantics_changed": False,
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "target_study": study_tag,
                "target_solution": solution_tag,
            },
            "source_mesh": source_mesh,
            "extension": extension,
            "solver": solver,
            "initialization_probe": initialization_probe,
            "source_metrics": source_metrics,
            "target_metrics": metrics,
            "relative_changes_from_source_radius": changes,
            "target_raw_residuals": residuals,
            "detailed_convergence": parse_detailed_residual_log(log_text),
            "solver_error": solve_error,
            "gates": gates,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.java.showProgress(False)
        client.clear()

    result.update(
        {
            "runtime_sec": time.time() - started,
            "comsol_version": args.version,
            "cores": args.cores,
            "source_model": str(args.source_model.resolve()),
            "source_sha256": _sha256(args.source_model),
            "model_path": str(model_out.resolve()),
            "model_sha256": _sha256(model_out),
            "log_path": str(log_path.resolve()),
        }
    )
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_out}")
    print(f"Wrote audit: {audit_path}")
    print(f"Wrote solver progress log: {log_path}")
    print(f"Radial-domain extension: {result['status']}")
    return int(
        (
            args.build_only
            and not result["status"].startswith("pass_")
        )
        or (
            not args.build_only
            and result["status"]
            != (
                "pass_radial_domain_numerical_gates_"
                "independence_comparison_pending"
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
