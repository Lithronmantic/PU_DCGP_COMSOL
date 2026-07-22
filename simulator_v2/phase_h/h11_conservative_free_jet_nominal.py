"""H11: solve and audit the uncalibrated conservative A-group free jet.

The source tree contains no workpiece and retains an internal cross-section
at the fixed A-group DPV coordinate, z=100 mm.  This runner first freezes a
0-to-1 effective-exit continuation and a full-load refinement.  The same
entry point can then run those studies and audit the unchanged pure-Ar gas
equations.

The DPV-plane quantities in this file are gas-phase numerical diagnostics.
They are not particle temperature, particle velocity, or an instrument
prediction.  Particle and observation operators remain locked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from simulator_v2.phase_h.h11_conservative_free_jet_skeleton import (
    AUDIT_PATH as SOURCE_AUDIT_PATH,
    MODEL_PATH as SOURCE_MODEL,
    _sha256,
)
from simulator_v2.phase_h.h11_target_impact_conservative_nominal import (
    _last_tag,
    _set_manual_scales,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_conservative_free_jet_nominal"
OUT_DIR = HERE / "h11_outputs" / "conservative_free_jet_nominal"
MODEL_PATH = MODEL_DIR / "h11_conservative_free_jet_nominal_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_conservative_free_jet_nominal_audit.json"
LOG_PATH = OUT_DIR / "h11_conservative_free_jet_nominal.log"


@dataclass(frozen=True)
class FreeJetSolveContract:
    """Frozen numerical contract; no value is an experimental calibration."""

    load_fractions: tuple[float, ...] = (
        0.0,
        0.002,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    )
    seed_velocity_m_s: float = 1.0
    continuation_relative_tolerance: float = 5e-4
    refinement_relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 2000
    refinement_maximum_segregated_iterations: int = 4000
    automatic_mesh_level: int = 4
    pressure_scale_pa: float = 1e5
    temperature_scale_k: float = 1e4
    velocity_scale_m_s: float = 600.0
    turbulent_kinetic_energy_scale_m2_s2: float = 1e3
    specific_dissipation_scale_s_inv: float = 1e6
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    dpv_plane_mm: float = 100.0
    dpv_profile_radii_mm: tuple[float, ...] = (
        0.001,
        1.0,
        2.0,
        4.0,
        6.0,
        10.0,
        20.0,
        30.0,
        39.0,
    )
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02
    workpiece_present: bool = False
    particle_population_attached: bool = False
    dpv_observation_operator_attached: bool = False
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        values = self.load_fractions
        if len(values) < 3 or values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("Continuation must span 0 to 1")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in values
        ):
            raise ValueError("Continuation values must be finite in [0, 1]")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Continuation values must strictly increase")
        if self.seed_velocity_m_s != 1.0:
            raise ValueError("Frozen ambient seed velocity is 1 m/s")
        if self.continuation_relative_tolerance != 5e-4:
            raise ValueError("Continuation tolerance must remain 5e-4")
        if self.refinement_relative_tolerance != 1e-6:
            raise ValueError("Refinement tolerance must remain 1e-6")
        if self.maximum_segregated_iterations != 2000:
            raise ValueError("Continuation iteration ceiling must remain 2000")
        if self.refinement_maximum_segregated_iterations != 4000:
            raise ValueError("Refinement iteration ceiling must remain 4000")
        if self.automatic_mesh_level != 4:
            raise ValueError("Initial free-jet mesh level must remain 4")
        scales = (
            self.pressure_scale_pa,
            self.temperature_scale_k,
            self.velocity_scale_m_s,
            self.turbulent_kinetic_energy_scale_m2_s2,
            self.specific_dissipation_scale_s_inv,
        )
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Manual dependent-variable scales must be positive")
        if self.property_temperature_floor_k != 250.0:
            raise ValueError("Temperature floor must remain 250 K")
        if self.pressure_floor_pa != 1_000.0:
            raise ValueError("Absolute-pressure floor must remain 1000 Pa")
        if self.dpv_plane_mm != 100.0:
            raise ValueError("A-group DPV plane must remain 100 mm")
        if (
            not self.dpv_profile_radii_mm
            or min(self.dpv_profile_radii_mm) <= 0
            or max(self.dpv_profile_radii_mm) >= 40
        ):
            raise ValueError("DPV profile radii must lie inside the 40 mm domain")
        if self.mass_imbalance_limit_fraction != 0.005:
            raise ValueError("Mass gate must remain 0.5 percent")
        if self.energy_imbalance_limit_fraction != 0.02:
            raise ValueError("Energy gate must remain 2 percent")
        if self.workpiece_present:
            raise ValueError("A workpiece is forbidden in the A-group free jet")
        if (
            self.particle_population_attached
            or self.dpv_observation_operator_attached
            or self.calibrated
            or self.paper_prediction_allowed
        ):
            raise ValueError("Nominal gas solve cannot claim downstream completion")

    def continuation_list(self) -> str:
        return " ".join(f"{value:.12g}" for value in self.load_fractions)


def _default_paths(build_only: bool) -> tuple[Path, Path, Path]:
    if not build_only:
        return MODEL_PATH, AUDIT_PATH, LOG_PATH
    return (
        MODEL_PATH.with_name(f"{MODEL_PATH.stem}_skeleton.mph"),
        AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_skeleton.json"),
        LOG_PATH.with_name(f"{LOG_PATH.stem}_skeleton.log"),
    )


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError("Expected one finite COMSOL scalar")
    return float(array.reshape(-1)[0])


def _range(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("COMSOL field contains undefined values")
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def dpv_profile_expressions(
    contract: FreeJetSolveContract,
    field: str,
) -> list[str]:
    contract.validate()
    return [
        f"at2({radius:.12g}[mm],100[mm],{field})"
        for radius in contract.dpv_profile_radii_mm
    ]


def corrected_mass_imbalance_fraction(
    *,
    nozzle_outward: float,
    open_outward: float,
    torch_outward: float,
    torch_weak_flux: float,
) -> float:
    values = (
        nozzle_outward,
        open_outward,
        torch_outward,
        torch_weak_flux,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Mass fluxes must be finite")
    inlet = abs(nozzle_outward)
    if inlet <= 0:
        raise ValueError("Nozzle mass inflow must be nonzero")
    residual = abs(
        nozzle_outward
        + open_outward
        + torch_outward
        - torch_weak_flux
    )
    return residual / inlet


def _source_tree_audit(jm: Any) -> dict[str, object]:
    component = jm.component("comp1")
    hmnf = component.physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Source free jet is not all-Mach k-omega")
    if str(
        hmnf.prop("PhysicalModelProperty").getString(
            "includeKineticEnergy"
        )
    ) != "1":
        raise RuntimeError("Conservative kinetic energy is disabled")
    parameter_names = {str(value) for value in jm.param().varnames()}
    forbidden = sorted(
        parameter_names & {"spray_distance", "d_spray", "T_target"}
    )
    if forbidden:
        raise RuntimeError(f"Workpiece parameters leaked into source: {forbidden}")
    selections = {
        name: sorted(
            int(value)
            for value in component.selection(tag).entities()
        )
        for name, tag in (
            ("nozzle", "geom1_sel_nozzle_in"),
            ("torch_face", "geom1_sel_torch_face"),
            ("radial_open", "geom1_sel_far_r"),
            ("downstream_open", "geom1_sel_far_z"),
            ("dpv_internal", "geom1_sel_dpv"),
        )
    }
    exterior = set().union(
        *(
            set(selections[name])
            for name in (
                "nozzle",
                "torch_face",
                "radial_open",
                "downstream_open",
            )
        )
    )
    if exterior & set(selections["dpv_internal"]):
        raise RuntimeError("DPV section became an exterior boundary")
    return {
        "physics_type": str(hmnf.getType()),
        "workpiece_parameters": forbidden,
        "boundary_selections": selections,
        "dpv_is_internal_only": True,
    }


def _configure_segregated_solver(
    jm: Any,
    solution_tag: str,
    *,
    tolerance: float,
    maximum_iterations: int,
    contract: FreeJetSolveContract,
) -> dict[str, object]:
    scales = _set_manual_scales(jm, solution_tag, contract)
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{tolerance:.12g}")
    advanced = stationary.feature("aDef")
    advanced.set("storeresidual", "solvingandoutput")
    advanced.set("convinfo", "detailed")
    advanced.set("checkmatherr", "on")
    features = {str(value) for value in stationary.feature().tags()}
    if "se1" not in features:
        raise RuntimeError("Expected the all-Mach segregated solver tree")
    segregated = stationary.feature("se1")
    segregated.set("maxsegiter", str(maximum_iterations))
    lower = segregated.feature("ll1")
    lower.set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    return {
        "manual_scales": scales,
        "relative_tolerance": float(str(stationary.getString("stol"))),
        "maximum_segregated_iterations": int(
            str(segregated.getString("maxsegiter"))
        ),
        "store_last_residual": str(
            advanced.getString("storeresidual")
        ),
        "convergence_log_level": str(
            advanced.getString("convinfo")
        ),
        "lower_limits": str(lower.getString("lowerlimit")),
    }


def configure_studies(
    jm: Any,
    contract: FreeJetSolveContract,
) -> dict[str, object]:
    contract.validate()
    source_audit = _source_tree_audit(jm)
    hmnf = jm.component("comp1").physics("hmnf")
    jm.param().set("load_s", "0", "Numerical continuation fraction")
    nozzle = hmnf.feature("nozzle_in")
    nozzle.set(
        "U0in",
        f"({contract.seed_velocity_m_s:.12g}[m/s]"
        f"+load_s*(u_exit_eff-{contract.seed_velocity_m_s:.12g}[m/s]))"
        "*nozzle_shape",
    )
    nozzle.set(
        "T0",
        "T_amb+load_s*(T_exit_eff-T_amb)*nozzle_shape",
    )

    mesh = jm.component("comp1").mesh("mesh1")
    mesh.autoMeshSize(contract.automatic_mesh_level)
    mesh.run()
    mesh_audit = {
        "automatic_level": contract.automatic_mesh_level,
        "elements": int(mesh.getNumElem()),
        "vertices": int(mesh.getNumVertex()),
        "minimum_quality": float(mesh.getMinQuality()),
        "mean_quality": float(mesh.getMeanQuality()),
    }

    study = jm.study("std1")
    features = study.feature()
    if "param" in {str(value) for value in features.tags()}:
        features.remove("param")
    parametric = features.create("param", "Parametric")
    parametric.label("Uncalibrated effective-exit continuation")
    parametric.set("pname", ["load_s"])
    parametric.set("plistarr", [contract.continuation_list()])
    parametric.set("punit", [""])
    parametric.set("sweeptype", "filled")
    parametric.set("reusesol", "on")
    parametric.set("keepsol", "all")
    study.createAutoSequences("all")
    continuation_solution = _last_tag(jm.sol())
    continuation_solver = _configure_segregated_solver(
        jm,
        continuation_solution,
        tolerance=contract.continuation_relative_tolerance,
        maximum_iterations=contract.maximum_segregated_iterations,
        contract=contract,
    )

    studies = jm.study()
    if "std_refine" in {str(value) for value in studies.tags()}:
        studies.remove("std_refine")
    refinement = studies.create("std_refine")
    refinement.label("Full-load free-jet numerical refinement")
    step = refinement.create("stat", "Stationary")
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", "std1")
    step.set("initstudystep", "stat")
    step.set("initsol", continuation_solution)
    step.set("solnum", "last")
    refinement.createAutoSequences("all")
    refinement_solution = _last_tag(jm.sol())
    refinement_solver = _configure_segregated_solver(
        jm,
        refinement_solution,
        tolerance=contract.refinement_relative_tolerance,
        maximum_iterations=(
            contract.refinement_maximum_segregated_iterations
        ),
        contract=contract,
    )
    jm.param().set("load_s", "1")
    return {
        "source_tree": source_audit,
        "mesh": mesh_audit,
        "continuation_study": "std1",
        "continuation_solution": continuation_solution,
        "continuation_solver": continuation_solver,
        "refinement_study": "std_refine",
        "refinement_solution": refinement_solution,
        "refinement_solver": refinement_solver,
        "geometry_changed": False,
        "physics_changed": False,
        "material_changed": False,
        "boundary_conditions_changed": False,
    }


def _last_dataset(model: Any) -> tuple[Any, dict[str, object]]:
    dataset = list(model / "datasets")[-1]
    inner_indices, _ = model.inner(dataset)
    selector: dict[str, object] = {}
    if len(inner_indices):
        selector["inner"] = "last"
    return dataset, selector


def _continuation_dataset(
    model: Any,
    contract: FreeJetSolveContract,
) -> tuple[Any, Any, Any]:
    """Find the frozen load ladder without relying on dataset position."""

    expected = np.asarray(contract.load_fractions, dtype=float)
    for dataset in model / "datasets":
        try:
            indices, values = model.inner(dataset)
        except Exception:
            continue
        observed = np.asarray(values, dtype=float).reshape(-1)
        if (
            len(indices) == expected.size
            and observed.size == expected.size
            and np.allclose(observed, expected, rtol=0.0, atol=1e-12)
        ):
            return dataset, indices, values
    raise RuntimeError("Frozen continuation dataset was not found")


def _evaluate_many(
    model: Any,
    expressions: Iterable[str],
    units: Iterable[str],
    dataset: Any,
    selector: dict[str, object],
) -> list[float]:
    values = model.evaluate(
        list(expressions),
        unit=list(units),
        dataset=dataset,
        **selector,
    )
    return [_scalar(value) for value in values]


def evaluate_solution(
    model: Any,
    contract: FreeJetSolveContract,
    *,
    expected_load_fraction: float | None = None,
) -> dict[str, object]:
    contract.validate()
    datasets = list(model / "datasets")
    dataset, selector = _last_dataset(model)
    if expected_load_fraction is None:
        if len(datasets) < 2:
            raise RuntimeError("Expected continuation and refinement datasets")
        _, indices, parameter_values = _continuation_dataset(model, contract)
        if (
            len(indices) != len(contract.load_fractions)
            or not math.isclose(float(parameter_values[-1]), 1.0)
        ):
            raise RuntimeError(
                "Continuation did not store the frozen load ladder"
            )
    else:
        if not 0.0 < expected_load_fraction <= 1.0:
            raise ValueError("Expected load fraction must lie in (0, 1]")
        evaluated_load = _scalar(
            model.evaluate(
                "load_s",
                unit="1",
                dataset=dataset,
                **selector,
            )
        )
        if not math.isclose(
            evaluated_load,
            expected_load_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Selected solution does not have the expected load fraction"
            )
        indices = np.asarray([1], dtype=int)
        parameter_values = np.asarray([evaluated_load], dtype=float)
    temperature, speed, pressure = model.evaluate(
        ["T", "hmnf.U", "p"],
        unit=["K", "m/s", "Pa"],
        dataset=dataset,
        **selector,
    )

    boundary_names = (
        ("nozzle_outward", "int_nozzle_hmnf"),
        ("open_outward", "int_open_hmnf"),
        ("torch_outward", "int_torch_hmnf"),
    )
    mass_fluxes: dict[str, float] = {}
    energy_fluxes: dict[str, float] = {}
    for name, operator in boundary_names:
        mass_fluxes[name] = _scalar(
            model.evaluate(
                f"{operator}(2*pi*r*hmnf.rho*(u*nr+w*nz))",
                unit="kg/s",
                dataset=dataset,
                **selector,
            )
        )
        energy_fluxes[name] = _scalar(
            model.evaluate(
                f"{operator}(2*pi*r*hmnf.nteflux)",
                unit="W",
                dataset=dataset,
                **selector,
            )
        )
    torch_weak_flux = _scalar(
        model.evaluate(
            "int_torch_hmnf("
            "2*pi*r*hmnf.contCoeffFace*hmnf.unJump)",
            unit="kg/s",
            dataset=dataset,
            **selector,
        )
    )
    mass_fraction = corrected_mass_imbalance_fraction(
        nozzle_outward=mass_fluxes["nozzle_outward"],
        open_outward=mass_fluxes["open_outward"],
        torch_outward=mass_fluxes["torch_outward"],
        torch_weak_flux=torch_weak_flux,
    )
    energy_balance = _scalar(
        model.evaluate(
            "hmnf.energyBalance",
            unit="W",
            dataset=dataset,
            **selector,
        )
    )
    inlet_energy = abs(energy_fluxes["nozzle_outward"])

    zero_flux = "0[kg/(m^2*s)]"
    axial_mass_density = "hmnf.rho*w"
    dpv_expressions = (
        f"int_dpv_hmnf(2*pi*r*({axial_mass_density}))",
        (
            "int_dpv_hmnf(2*pi*r*"
            f"max({axial_mass_density},{zero_flux}))"
        ),
        (
            "-int_dpv_hmnf(2*pi*r*"
            f"min({axial_mass_density},{zero_flux}))"
        ),
        (
            "int_dpv_hmnf(2*pi*r*"
            f"max({axial_mass_density},{zero_flux})*T)"
        ),
        (
            "int_dpv_hmnf(2*pi*r*"
            f"max({axial_mass_density},{zero_flux})*hmnf.U)"
        ),
    )
    (
        dpv_net_mass,
        dpv_forward_mass,
        dpv_reverse_mass,
        dpv_temperature_moment,
        dpv_speed_moment,
    ) = _evaluate_many(
        model,
        dpv_expressions,
        ("kg/s", "kg/s", "kg/s", "kg*K/s", "kg*m/s^2"),
        dataset,
        selector,
    )
    if dpv_forward_mass <= 0:
        raise RuntimeError("No forward gas mass crosses the fixed DPV plane")

    profile = {}
    for name, field, unit in (
        ("temperature_k", "T", "K"),
        ("speed_m_s", "hmnf.U", "m/s"),
        ("axial_velocity_m_s", "w", "m/s"),
        ("absolute_pressure_pa", "p", "Pa"),
    ):
        expressions = dpv_profile_expressions(contract, field)
        profile[name] = _evaluate_many(
            model,
            expressions,
            [unit] * len(expressions),
            dataset,
            selector,
        )
    return {
        "n_continuation_solutions": int(len(indices)),
        "full_load_fraction": float(parameter_values[-1]),
        "temperature_k": _range(temperature),
        "speed_m_s": _range(speed),
        "absolute_pressure_pa": _range(pressure),
        "mass_flux_kg_s": {
            **mass_fluxes,
            "torch_upwind_weak_flux": torch_weak_flux,
            "discrete_corrected_absolute_residual": (
                mass_fraction * abs(mass_fluxes["nozzle_outward"])
            ),
            "imbalance_fraction": mass_fraction,
            "definition": (
                "abs(nozzle+open+torch-torch_DG_weak_flux)"
                "/abs(nozzle)"
            ),
        },
        "energy_balance_w": {
            **energy_fluxes,
            "comsol_energy_balance": energy_balance,
            "imbalance_fraction_of_inlet": (
                abs(energy_balance) / inlet_energy
            ),
            "definition": "hmnf.energyBalance=dEi0Int+ntefluxInt-WInt-QInt",
        },
        "fixed_dpv_plane_gas_diagnostics": {
            "z_mm": contract.dpv_plane_mm,
            "net_axial_mass_flux_kg_s": dpv_net_mass,
            "forward_axial_mass_flux_kg_s": dpv_forward_mass,
            "reverse_axial_mass_flux_kg_s": dpv_reverse_mass,
            "forward_mass_weighted_temperature_k": (
                dpv_temperature_moment / dpv_forward_mass
            ),
            "forward_mass_weighted_speed_m_s": (
                dpv_speed_moment / dpv_forward_mass
            ),
            "profile_radii_mm": list(contract.dpv_profile_radii_mm),
            "gas_profile": profile,
            "interpretation": (
                "Gas-phase state only; not a DPV particle or instrument "
                "observation."
            ),
        },
    }


def _partial_failure_payload(
    model: Any,
    contract: FreeJetSolveContract,
    *,
    phase: str,
    error: Exception,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "h11_free_jet_nominal_failure_v1",
        "status": f"failed_{phase}",
        "contract": asdict(contract),
        "error": str(error),
    }
    datasets = list(model / "datasets")
    if datasets:
        try:
            indices, values = model.inner(datasets[0])
            payload["stored_parameter_values"] = [
                float(value) for value in values
            ]
            payload["n_stored_solutions"] = int(len(indices))
            payload["last_stored_load"] = (
                float(values[-1]) if len(values) else None
            )
        except Exception as audit_error:
            payload["partial_solution_audit_error"] = str(audit_error)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-in", type=Path, default=SOURCE_MODEL)
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
    if not args.model_in.exists():
        raise FileNotFoundError(args.model_in)
    if not SOURCE_AUDIT_PATH.exists():
        raise FileNotFoundError(SOURCE_AUDIT_PATH)
    contract = FreeJetSolveContract()
    contract.validate()
    default_model, default_audit, default_log = _default_paths(
        args.build_only
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
        model = client.load(str(args.model_in))
        model.rename(model_out.stem)
        jm = model.java
        solver_audit = configure_studies(jm, contract)
        if args.build_only:
            status = "pass_free_jet_nominal_solver_skeleton"
            metrics: dict[str, object] = {}
            gates = {
                "solver_run": False,
                "physically_bounded": False,
                "mass_imbalance_below_0_5_percent": False,
                "energy_imbalance_below_2_percent": False,
                "forward_gas_crosses_fixed_dpv_plane": False,
            }
        else:
            try:
                jm.study("std1").run()
            except Exception as exc:
                partial_model = model_out.with_name(
                    f"{model_out.stem}_continuation_partial.mph"
                )
                model.save(str(partial_model))
                failure = _partial_failure_payload(
                    model,
                    contract,
                    phase="free_jet_continuation",
                    error=exc,
                )
                failure.update(
                    {
                        "partial_model": str(partial_model.resolve()),
                        "partial_model_sha256": _sha256(partial_model),
                    }
                )
                failure_path = audit_path.with_name(
                    f"{audit_path.stem}_continuation_failure.json"
                )
                with failure_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        failure,
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"Wrote failure audit: {failure_path}", flush=True)
                raise
            try:
                jm.study("std_refine").run()
            except Exception as exc:
                partial_model = model_out.with_name(
                    f"{model_out.stem}_refinement_partial.mph"
                )
                model.save(str(partial_model))
                failure = _partial_failure_payload(
                    model,
                    contract,
                    phase="free_jet_refinement",
                    error=exc,
                )
                failure.update(
                    {
                        "partial_model": str(partial_model.resolve()),
                        "partial_model_sha256": _sha256(partial_model),
                    }
                )
                failure_path = audit_path.with_name(
                    f"{audit_path.stem}_refinement_failure.json"
                )
                with failure_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        failure,
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"Wrote failure audit: {failure_path}", flush=True)
                raise
            metrics = evaluate_solution(model, contract)
            bounded = bool(
                float(metrics["temperature_k"]["minimum"])  # type: ignore[index]
                >= contract.property_temperature_floor_k
                and float(  # type: ignore[index]
                    metrics["absolute_pressure_pa"]["minimum"]
                )
                >= contract.pressure_floor_pa
            )
            mass_pass = bool(
                float(  # type: ignore[index]
                    metrics["mass_flux_kg_s"]["imbalance_fraction"]
                )
                < contract.mass_imbalance_limit_fraction
            )
            energy_pass = bool(
                float(  # type: ignore[index]
                    metrics["energy_balance_w"][
                        "imbalance_fraction_of_inlet"
                    ]
                )
                < contract.energy_imbalance_limit_fraction
            )
            dpv_forward = bool(
                float(  # type: ignore[index]
                    metrics["fixed_dpv_plane_gas_diagnostics"][
                        "forward_axial_mass_flux_kg_s"
                    ]
                )
                > 0
            )
            gates = {
                "solver_run": True,
                "physically_bounded": bounded,
                "mass_imbalance_below_0_5_percent": mass_pass,
                "energy_imbalance_below_2_percent": energy_pass,
                "forward_gas_crosses_fixed_dpv_plane": dpv_forward,
            }
            status = (
                "pass_uncalibrated_free_jet_numerical_gates"
                if all(gates.values())
                else "solve_completed_but_free_jet_gate_failed"
            )
        model.save(str(model_out))
    finally:
        client.java.showProgress(False)
        client.clear()

    audit = {
        "schema_version": "h11_conservative_free_jet_nominal_v1",
        "status": status,
        "contract": asdict(contract),
        "solver_and_mesh": solver_audit,
        "metrics": metrics,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
        "runtime_sec": time.time() - started,
        "comsol_version": args.version,
        "cores": args.cores,
        "source_model": str(args.model_in.resolve()),
        "source_sha256": _sha256(args.model_in),
        "source_audit": str(SOURCE_AUDIT_PATH.resolve()),
        "source_audit_sha256": _sha256(SOURCE_AUDIT_PATH),
        "model_path": str(model_out.resolve()),
        "model_sha256": _sha256(model_out),
        "log_path": str(log_path.resolve()),
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_out}")
    print(f"Wrote audit: {audit_path}")
    print(f"Free-jet nominal: {status}")
    return int(
        not args.build_only
        and status != "pass_uncalibrated_free_jet_numerical_gates"
    )


if __name__ == "__main__":
    raise SystemExit(main())
