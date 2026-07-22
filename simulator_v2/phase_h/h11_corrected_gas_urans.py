"""Build and audit the fine-mesh URANS pilot without running production."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_convergence import (
    _mesh_audit,
    _sha256,
    _solution_for_study,
    _write_json,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_geometric_bridge import (
    artifact_paths as geometric_bridge_artifact_paths,
)
from simulator_v2.phase_h.h11_corrected_gas_urans_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    CorrectedGasUransContract,
)
from simulator_v2.phase_h.h11_effective_exit_directional import _gas_gates
from simulator_v2.phase_h.h11_target_impact_conservative_nominal import (
    _set_manual_scales,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_corrected_gas_urans"
OUTPUT_DIR = HERE / "h11_outputs" / "corrected_gas_urans"
PILOT_SKELETON_MODEL = MODEL_DIR / "h11_corrected_gas_urans_pilot_skeleton.mph"
PILOT_SKELETON_AUDIT = OUTPUT_DIR / "h11_corrected_gas_urans_pilot_skeleton.json"
PILOT_MODEL = MODEL_DIR / "h11_corrected_gas_urans_pilot.mph"
PILOT_AUDIT = OUTPUT_DIR / "h11_corrected_gas_urans_pilot.json"
PILOT_LOG = OUTPUT_DIR / "h11_corrected_gas_urans_pilot.log"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _physics_activation(jm: Any) -> list[str]:
    tags = [
        str(tag) for tag in jm.component("comp1").physics().tags()
    ]
    activation: list[str] = []
    for tag in tags:
        activation.extend([tag, "on" if tag == "hmnf" else "off"])
    return activation


def _ensure_domain_mass_operator(jm: Any) -> None:
    component = jm.component("comp1")
    couplings = component.cpl()
    tag = "int_domain_urans"
    if tag in {str(value) for value in couplings.tags()}:
        couplings.remove(tag)
    operator = couplings.create(tag, "Integration")
    operator.label("Axisymmetric URANS domain mass integral")
    operator.selection().geom("geom1", 2)
    operator.selection().all()


def configure_urans_pilot(
    jm: Any,
    *,
    source_study: str,
    source_solution: str,
) -> tuple[str, str, dict[str, Any]]:
    contract = CorrectedGasUransContract()
    contract.validate()
    study_tag = "std_urans_pilot"
    if study_tag in {str(tag) for tag in jm.study().tags()}:
        jm.study().remove(study_tag)
    study = jm.study().create(study_tag)
    study.label("Fine-mesh URANS feasibility pilot")
    step = study.create("time", "Transient")
    step.set("activate", _physics_activation(jm))
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", source_study)
    step.set("initstudystep", "stat")
    step.set("initsol", source_solution)
    step.set("solnum", "last")
    step.set(
        "tlist",
        "range("
        f"0,{contract.pilot_output_step_s:.12g},"
        f"{contract.pilot_end_time_s:.12g})",
    )
    study.createAutoSequences("all")
    solution_tag = _solution_for_study(jm, study_tag)
    solution = jm.sol(solution_tag)
    time_solver = solution.feature("t1")
    time_solver.set("rtol", f"{contract.time_relative_tolerance:.12g}")
    time_solver.set("timemethod", contract.time_method)
    time_solver.set("tstepsbdf", "free")
    time_solver.set("minorder", str(contract.minimum_bdf_order))
    time_solver.set("maxorder", str(contract.maximum_bdf_order))
    time_solver.set("initialstepbdfactive", "on")
    time_solver.set(
        "initialstepbdf",
        f"{contract.initial_time_step_s:.12g}",
    )
    time_solver.set("maxstepconstraintbdf", "const")
    time_solver.set(
        "maxstepbdf",
        f"{contract.maximum_internal_time_step_s:.12g}",
    )

    scales = _set_manual_scales(
        jm,
        solution_tag,
        FreeJetSolveContract(),
    )
    variables = solution.feature("v1")
    wall_scales: dict[str, dict[str, str]] = {}
    for field_tag in ("comp1_hmnf_TWall_d", "comp1_hmnf_TWall_u"):
        field = variables.feature(field_tag)
        field.set("scalemethod", "manual")
        field.set("scaleval", "10000")
        wall_scales[field_tag] = {
            "scalemethod": str(field.getString("scalemethod")),
            "scaleval": str(field.getString("scaleval")),
        }

    segregated = time_solver.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.maximum_segregated_iterations_per_step),
    )
    lower = segregated.feature("ll1")
    lower.set(
        "lowerlimit",
        "comp1.k 0 comp1.om 0 comp1.T 250[K] comp1.p 1000[Pa] ",
    )
    advanced = time_solver.feature("aDef")
    advanced.set("storeresidual", "solvingandoutput")
    advanced.set("convinfo", "detailed")
    advanced.set("checkmatherr", "on")

    audit = {
        "method": str(time_solver.getString("timemethod")),
        "minimum_order": int(str(time_solver.getString("minorder"))),
        "maximum_order": int(str(time_solver.getString("maxorder"))),
        "relative_tolerance": float(str(time_solver.getString("rtol"))),
        "initial_step_s": float(
            str(time_solver.getString("initialstepbdf"))
        ),
        "maximum_step_s": float(str(time_solver.getString("maxstepbdf"))),
        "maximum_segregated_iterations_per_step": int(
            str(segregated.getString("maxsegiter"))
        ),
        "manual_scales": scales,
        "wall_temperature_field_scales": wall_scales,
        "lower_limits": str(lower.getString("lowerlimit")),
        "store_last_residual": str(advanced.getString("storeresidual")),
        "convergence_log_level": str(advanced.getString("convinfo")),
    }
    return study_tag, solution_tag, audit


def build_pilot_skeleton(client: Any) -> dict[str, Any]:
    contract = CorrectedGasUransContract()
    contract.validate()
    source_paths = geometric_bridge_artifact_paths(True)
    source_audit = _load_json(source_paths["audit"])
    if source_audit["status"] != "pass_geometric_mesh_urans_preconditioner":
        raise RuntimeError("The physical URANS preconditioner has not passed")
    if source_audit["relative_tolerance"] != (
        contract.preconditioner_relative_tolerance
    ):
        raise RuntimeError("The URANS preconditioner tolerance changed")
    model = client.load(str(source_paths["model"]))
    model.rename(PILOT_SKELETON_MODEL.stem)
    jm = model.java
    try:
        source_study = "std_mesh_geometric_bridge"
        source_solution = _solution_for_study(jm, source_study)
        source_mesh = _mesh_audit(jm, "geometric_bridge")
        _ensure_domain_mass_operator(jm)
        study_tag, solution_tag, solver = configure_urans_pilot(
            jm,
            source_study=source_study,
            source_solution=source_solution,
        )
        if source_mesh["elements"] != 39_811:
            raise RuntimeError("The URANS pilot mesh changed")
        parameter_names = {str(name) for name in jm.param().varnames()}
        forbidden = sorted(
            parameter_names & {"spray_distance", "d_spray", "T_target"}
        )
        if forbidden:
            raise RuntimeError(
                f"Workpiece parameters leaked into URANS: {forbidden}"
            )
        PILOT_SKELETON_MODEL.parent.mkdir(parents=True, exist_ok=True)
        PILOT_SKELETON_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(PILOT_SKELETON_MODEL))
        payload = {
            "schema_version": "h11_corrected_gas_urans_pilot_skeleton_v1",
            "status": "pass_corrected_gas_urans_pilot_skeleton",
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_model": str(source_paths["model"].resolve()),
            "source_model_sha256": _sha256(source_paths["model"]),
            "source_audit": str(source_paths["audit"].resolve()),
            "source_audit_sha256": _sha256(source_paths["audit"]),
            "source_study": source_study,
            "source_solution": source_solution,
            "pilot_study": study_tag,
            "pilot_solution": solution_tag,
            "mesh": source_mesh,
            "solver": solver,
            "geometry_changed": False,
            "physics_changed": False,
            "workpiece_parameters": forbidden,
            "model_path": str(PILOT_SKELETON_MODEL.resolve()),
            "model_sha256": _sha256(PILOT_SKELETON_MODEL),
            "role": "build_only_no_time_step_solved",
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(PILOT_SKELETON_AUDIT, payload)
        return payload
    finally:
        client.remove(model)


def _dataset_for_solution(model: Any, solution_tag: str) -> Any:
    matches = []
    for dataset in model / "datasets":
        properties = dataset.properties()
        reference = None
        if "solution" in properties:
            reference = dataset.property("solution")
        elif "data" in properties:
            reference = dataset.property("data")
        if reference == solution_tag:
            matches.append(dataset)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one dataset for {solution_tag}: {matches}"
        )
    return matches[0]


def _array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("URANS evaluation contains undefined values")
    return array


def _time_series_matrix(
    value: Any,
    *,
    expression_count: int,
    time_count: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if not np.isfinite(array).all():
        raise ValueError("URANS time series contains undefined values")
    if expression_count == 1:
        result = array.reshape(1, -1)
    elif array.shape == (time_count, expression_count):
        result = array.T
    elif array.shape == (expression_count, time_count):
        result = array
    else:
        raise ValueError(
            "Unexpected global-evaluation shape: "
            f"{array.shape}, expected ({time_count},{expression_count}) "
            f"or ({expression_count},{time_count})"
        )
    if result.shape != (expression_count, time_count):
        raise ValueError("URANS time-series matrix shape is inconsistent")
    return result


def _pilot_time_audit(model: Any, solution_tag: str) -> dict[str, Any]:
    contract = CorrectedGasUransContract()
    dataset = _dataset_for_solution(model, solution_tag)
    indices, values = model.inner(dataset)
    times = _array(values)
    inner = [int(value) for value in np.asarray(indices).reshape(-1)]
    if times.size != len(inner):
        raise RuntimeError("URANS time indices and values do not align")

    mass = _array(
        model.evaluate(
            "int_domain_urans(2*pi*r*hmnf.rho)",
            unit="kg",
            dataset=dataset,
            inner=inner,
        )
    )
    mass_expressions = [
        f"{operator}(2*pi*r*hmnf.rho*(u*nr+w*nz))"
        for operator in (
            "int_nozzle_hmnf",
            "int_open_hmnf",
            "int_torch_hmnf",
        )
    ]
    mass_expressions.append(
        "int_torch_hmnf(2*pi*r*hmnf.contCoeffFace*hmnf.unJump)"
    )
    mass_terms = _time_series_matrix(
        model.evaluate(
            mass_expressions,
            unit=["kg/s"] * len(mass_expressions),
            dataset=dataset,
            inner=inner,
        ),
        expression_count=len(mass_expressions),
        time_count=times.size,
    )
    nozzle, open_boundary, torch, torch_weak = mass_terms
    outward_flux = nozzle + open_boundary + torch - torch_weak
    storage_rate = np.gradient(mass, times, edge_order=2)
    mass_fraction = np.abs(storage_rate + outward_flux) / np.maximum(
        np.abs(nozzle),
        np.finfo(float).tiny,
    )

    energy_balance = _array(
        model.evaluate(
            "hmnf.energyBalance",
            unit="W",
            dataset=dataset,
            inner=inner,
        )
    )
    inlet_energy = np.abs(
        _array(
            model.evaluate(
                "int_nozzle_hmnf(2*pi*r*hmnf.nteflux)",
                unit="W",
                dataset=dataset,
                inner=inner,
            )
        )
    )
    energy_fraction = np.abs(energy_balance) / np.maximum(
        inlet_energy,
        np.finfo(float).tiny,
    )

    radii = FreeJetSolveContract().dpv_profile_radii_mm
    profiles: dict[str, np.ndarray] = {}
    for name, field, unit in (
        ("temperature_k", "T", "K"),
        ("speed_m_s", "hmnf.U", "m/s"),
        ("absolute_pressure_pa", "p", "Pa"),
    ):
        expressions = [
            f"at2({radius:.12g}[mm],100[mm],{field})"
            for radius in radii
        ]
        profiles[name] = _time_series_matrix(
            model.evaluate(
                expressions,
                unit=[unit] * len(expressions),
                dataset=dataset,
                inner=inner,
            ),
            expression_count=len(expressions),
            time_count=times.size,
        )

    final_temperature, final_speed, final_pressure = model.evaluate(
        ["T", "hmnf.U", "p"],
        unit=["K", "m/s", "Pa"],
        dataset=dataset,
        inner="last",
    )
    final_temperature = _array(final_temperature)
    final_speed = _array(final_speed)
    final_pressure = _array(final_pressure)
    expected_count = (
        round(contract.pilot_end_time_s / contract.pilot_output_step_s)
        + 1
    )
    gates = {
        "all_requested_output_times_stored": times.size == expected_count,
        "pilot_reached_frozen_end_time": math.isclose(
            float(times[-1]),
            contract.pilot_end_time_s,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "temperature_within_property_range_at_final_time": (
            float(final_temperature.min()) >= 250.0
            and float(final_temperature.max()) <= 25_000.0
        ),
        "absolute_pressure_positive_at_final_time": (
            float(final_pressure.min()) > 0
        ),
        "transient_mass_balance_95_percentile_below_0_5_percent": (
            float(np.quantile(mass_fraction[1:], 0.95))
            <= contract.mass_imbalance_limit_fraction
        ),
        "transient_energy_balance_95_percentile_below_2_percent": (
            float(np.quantile(energy_fraction[1:], 0.95))
            <= contract.energy_imbalance_limit_fraction
        ),
    }
    return {
        "dataset": str(dataset),
        "time_s": times.tolist(),
        "n_output_times": int(times.size),
        "final_field_ranges": {
            "temperature_k": {
                "minimum": float(final_temperature.min()),
                "maximum": float(final_temperature.max()),
            },
            "speed_m_s": {
                "minimum": float(final_speed.min()),
                "maximum": float(final_speed.max()),
            },
            "absolute_pressure_pa": {
                "minimum": float(final_pressure.min()),
                "maximum": float(final_pressure.max()),
            },
        },
        "transient_mass_balance": {
            "fraction": mass_fraction.tolist(),
            "median_fraction": float(np.median(mass_fraction[1:])),
            "p95_fraction": float(np.quantile(mass_fraction[1:], 0.95)),
            "maximum_fraction": float(mass_fraction[1:].max()),
            "definition": (
                "abs(d/dt integral(rho dV)+corrected outward mass flux)"
                "/abs(nozzle mass inflow)"
            ),
        },
        "transient_energy_balance": {
            "fraction": energy_fraction.tolist(),
            "median_fraction": float(np.median(energy_fraction[1:])),
            "p95_fraction": float(np.quantile(energy_fraction[1:], 0.95)),
            "maximum_fraction": float(energy_fraction[1:].max()),
            "definition": (
                "abs(hmnf.energyBalance)/abs(nozzle energy inflow)"
            ),
        },
        "dpv_profile_time_series": {
            "radii_mm": list(radii),
            **{
                name: array.tolist()
                for name, array in profiles.items()
            },
        },
        "gates": gates,
    }


def solve_pilot(client: Any) -> dict[str, Any]:
    skeleton = _load_json(PILOT_SKELETON_AUDIT)
    if skeleton["status"] != "pass_corrected_gas_urans_pilot_skeleton":
        raise RuntimeError("The URANS pilot skeleton has not passed")
    model = client.load(str(PILOT_SKELETON_MODEL))
    model.rename(PILOT_MODEL.stem)
    jm = model.java
    client.java.showProgress(str(PILOT_LOG.resolve()))
    started = time.time()
    try:
        jm.study("std_urans_pilot").run()
        PILOT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(PILOT_MODEL))
        time_audit = _pilot_time_audit(model, skeleton["pilot_solution"])
        final_metrics = evaluate_solution(model, FreeJetSolveContract())
        final_gates = _gas_gates(final_metrics)
        final_gates.pop("mass_imbalance_below_0_5_percent")
        gates = {
            **time_audit["gates"],
            **{
                f"final_{name}": value
                for name, value in final_gates.items()
            },
        }
        payload = {
            "schema_version": "h11_corrected_gas_urans_pilot_v1",
            "status": (
                "pass_corrected_gas_urans_pilot"
                if all(gates.values())
                else "fail_corrected_gas_urans_pilot_gates"
            ),
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "skeleton_audit": str(PILOT_SKELETON_AUDIT.resolve()),
            "skeleton_audit_sha256": _sha256(PILOT_SKELETON_AUDIT),
            "mesh": skeleton["mesh"],
            "solver": skeleton["solver"],
            "time_audit": time_audit,
            "final_metrics": final_metrics,
            "gates": gates,
            "model_path": str(PILOT_MODEL.resolve()),
            "model_sha256": _sha256(PILOT_MODEL),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(PILOT_AUDIT, payload)
        return payload
    finally:
        client.java.showProgress(False)
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--solve-pilot", action="store_true")
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        result = (
            solve_pilot(client)
            if args.solve_pilot
            else build_pilot_skeleton(client)
        )
    finally:
        client.clear()
    print(f"{result['status']}: {result['model_path']}")
    return 0 if result["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
