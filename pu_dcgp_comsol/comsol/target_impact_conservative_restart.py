
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.target_impact_conservative_nominal import (
    ConservativeSolveContract,
    _range,
    _scalar,
    _set_manual_scales,
)


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_mesh_convergence"
    / "h11_target_impact_mesh_level_4.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_restart"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_restart"
MODEL_PATH = MODEL_DIR / "h11_target_impact_conservative_restart_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_conservative_restart_audit.json"


@dataclass(frozen=True)
class ConservativeRestartContract:

    source_solution_tag: str = "sol2"
    source_study_tag: str = "std_refine"
    source_pressure_is_gauge: bool = True
    source_physics_tags: tuple[str, ...] = ("spf", "ht")
    continuation_tolerance: float = 5e-4
    refinement_tolerance: float = 1e-6
    maximum_iterations: int = 300
    maximum_segregated_iterations: int = 4000
    pressure_scale_pa: float = 1e5
    temperature_scale_k: float = 1e4
    velocity_scale_m_s: float = 600.0
    turbulent_kinetic_energy_scale_m2_s2: float = 1e3
    specific_dissipation_scale_s_inv: float = 1e6
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not self.source_solution_tag or not self.source_study_tag:
            raise ValueError("Source solution and study tags are required")
        if not self.source_pressure_is_gauge:
            raise ValueError("The audited source pressure convention is gauge")
        if set(self.source_physics_tags) != {"spf", "ht"}:
            raise ValueError("Unexpected source physics contract")
        if not 0 < self.continuation_tolerance <= 1e-3:
            raise ValueError("Initial solve tolerance must lie in (0, 1e-3]")
        if not 0 < self.refinement_tolerance <= 1e-5:
            raise ValueError("Refinement tolerance must lie in (0, 1e-5]")
        if self.maximum_iterations < 100:
            raise ValueError("Fully coupled iteration limit is too small")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Segregated iteration limit is too small")
        scales = (
            self.pressure_scale_pa,
            self.temperature_scale_k,
            self.velocity_scale_m_s,
            self.turbulent_kinetic_energy_scale_m2_s2,
            self.specific_dissipation_scale_s_inv,
            self.pressure_floor_pa,
        )
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Variable scales and pressure floor must be positive")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Restart verification cannot claim prediction")

    def solve_contract(self) -> ConservativeSolveContract:
        return ConservativeSolveContract(
            continuation_relative_tolerance=self.continuation_tolerance,
            refinement_relative_tolerance=self.refinement_tolerance,
            maximum_segregated_iterations=self.maximum_segregated_iterations,
            refinement_maximum_segregated_iterations=(
                self.maximum_segregated_iterations
            ),
            pressure_scale_pa=self.pressure_scale_pa,
            temperature_scale_k=self.temperature_scale_k,
            velocity_scale_m_s=self.velocity_scale_m_s,
            turbulent_kinetic_energy_scale_m2_s2=(
                self.turbulent_kinetic_energy_scale_m2_s2
            ),
            specific_dissipation_scale_s_inv=(
                self.specific_dissipation_scale_s_inv
            ),
            property_temperature_floor_k=self.property_temperature_floor_k,
            pressure_floor_pa=self.pressure_floor_pa,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _entities(component: Any, selection_tag: str) -> list[int]:
    return [int(value) for value in component.selection(selection_tag).entities()]


def _replace_coupling(component: Any, tag: str, selection: list[int]) -> None:
    couplings = component.cpl()
    if tag in {str(value) for value in couplings.tags()}:
        couplings.remove(tag)
    operator = couplings.create(tag, "Integration")
    operator.selection().geom("geom1", 1)
    operator.selection().set(selection)


def validate_source_model(jm: Any, contract: ConservativeRestartContract) -> dict[str, Any]:
    contract.validate()
    comp = jm.component("comp1")
    physics = {str(value) for value in comp.physics().tags()}
    missing_physics = set(contract.source_physics_tags) - physics
    if missing_physics:
        raise RuntimeError(f"Missing source physics: {sorted(missing_physics)}")
    solutions = {str(value) for value in jm.sol().tags()}
    if contract.source_solution_tag not in solutions:
        raise RuntimeError("Audited source solution is missing")
    studies = {str(value) for value in jm.study().tags()}
    if contract.source_study_tag not in studies:
        raise RuntimeError("Audited source study is missing")
    required_selections = {
        "geom1_sel_nozzle_in",
        "geom1_sel_ambient_in",
        "geom1_sel_far_r",
        "geom1_sel_target",
    }
    present_selections = {str(value) for value in comp.selection().tags()}
    missing_selections = required_selections - present_selections
    if missing_selections:
        raise RuntimeError(f"Missing selections: {sorted(missing_selections)}")
    if "R_Ar" not in {str(value) for value in jm.param().varnames()}:
        jm.param().set("R_Ar", "8.314462618[J/(mol*K)]/0.039948[kg/mol]")
    return {
        "source_physics": sorted(physics),
        "source_solutions": sorted(solutions),
        "source_studies": sorted(studies),
        "mesh_elements": int(comp.mesh("mesh1").getNumElem()),
        "mesh_vertices": int(comp.mesh("mesh1").getNumVertex()),
    }


def replace_with_conservative_rans(
    jm: Any,
    contract: ConservativeRestartContract,
) -> None:

    comp = jm.component("comp1")
    physics = comp.physics()
    for tag in contract.source_physics_tags:
        if tag in {str(value) for value in physics.tags()}:
            physics.remove(tag)
    if "hmnf" in {str(value) for value in physics.tags()}:
        physics.remove("hmnf")

    hmnf = physics.create(
        "hmnf",
        "HighMachNumberFlowTurbulentkomega",
        "geom1",
    )
    hmnf.label("Conservative compressible target plume, k-omega restart")
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

    source = contract.source_solution_tag
    initial = hmnf.feature("init1")
    initial.set(
        "u_init",
        [
            f"withsol('{source}',u)",
            "0",
            f"withsol('{source}',w)",
        ],
    )
    initial.set("p_init", f"withsol('{source}',p+p_amb)")
    initial.set("Tinit", f"max(withsol('{source}',T),250[K])")
    initial.set("k_init", f"max(withsol('{source}',k),1e-8[m^2/s^2])")
    initial.set("om_init", f"max(withsol('{source}',om),1[1/s])")

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

    ambient_entities = sorted(
        {
            *_entities(comp, "geom1_sel_ambient_in"),
            *_entities(comp, "geom1_sel_far_r"),
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

    target = hmnf.create("target_temperature", "TemperatureBoundary", 1)
    target.label("Measured-range isothermal workpiece")
    target.selection().named("geom1_sel_target")
    target.set("T0", "T_target")

    _replace_coupling(
        comp,
        "int_nozzle_hmnf",
        _entities(comp, "geom1_sel_nozzle_in"),
    )
    _replace_coupling(comp, "int_ambient_hmnf", ambient_entities)
    _replace_coupling(
        comp,
        "int_target_hmnf",
        _entities(comp, "geom1_sel_target"),
    )


def _last_tag(collection: Any) -> str:
    tags = [str(value) for value in collection.tags()]
    if not tags:
        raise RuntimeError("Expected nonempty COMSOL collection")
    return tags[-1]


def _configure_nonlinear_solver(
    jm: Any,
    solution_tag: str,
    contract: ConservativeRestartContract,
    tolerance: float,
) -> dict[str, Any]:
    scales = _set_manual_scales(jm, solution_tag, contract.solve_contract())
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{tolerance:.12g}")
    features = {str(value) for value in stationary.feature().tags()}
    strategy: dict[str, Any] = {"manual_scales": scales}
    if "fc1" in features:
        fully_coupled = stationary.feature("fc1")
        fully_coupled.set("dtech", "hnlin")
        fully_coupled.set("initsteph", "1e-4")
        fully_coupled.set("minsteph", "1e-10")
        fully_coupled.set("useminsteprecovery", "on")
        fully_coupled.set("minsteprecovery", "0.1")
        fully_coupled.set("maxiter", str(contract.maximum_iterations))
        strategy["nonlinear_method"] = "fully_coupled_highly_nonlinear_newton"
    elif "se1" in features:
        segregated = stationary.feature("se1")
        segregated.set(
            "maxsegiter",
            str(contract.maximum_segregated_iterations),
        )
        segregated.feature("ll1").set(
            "lowerlimit",
            f"comp1.k 0 comp1.om 0 comp1.T "
            f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
            f"{contract.pressure_floor_pa:.12g}[Pa] ",
        )
        strategy["nonlinear_method"] = "segregated_with_physical_lower_limits"
    else:
        raise RuntimeError("Unsupported nonlinear solver tree")
    return strategy


def configure_restart_studies(
    jm: Any,
    contract: ConservativeRestartContract,
) -> tuple[str, str, dict[str, Any]]:
    studies = jm.study()
    if "std_hmnf" in {str(value) for value in studies.tags()}:
        studies.remove("std_hmnf")
    first = studies.create("std_hmnf")
    first.label("Conservative all-Mach RANS restart")
    first.create("stat", "Stationary")
    first.createAutoSequences("all")
    first_solution = _last_tag(jm.sol())
    first_strategy = _configure_nonlinear_solver(
        jm,
        first_solution,
        contract,
        contract.continuation_tolerance,
    )

    if "std_hmnf_refine" in {str(value) for value in studies.tags()}:
        studies.remove("std_hmnf_refine")
    refine = studies.create("std_hmnf_refine")
    refine.label("Conservative all-Mach RANS residual refinement")
    step = refine.create("stat", "Stationary")
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", "std_hmnf")
    step.set("initstudystep", "stat")
    step.set("initsol", first_solution)
    step.set("solnum", "last")
    refine.createAutoSequences("all")
    refine_solution = _last_tag(jm.sol())
    refine_strategy = _configure_nonlinear_solver(
        jm,
        refine_solution,
        contract,
        contract.refinement_tolerance,
    )
    return first_solution, refine_solution, {
        "initial": first_strategy,
        "refinement": refine_strategy,
    }


def evaluate_solution(model: Any) -> dict[str, Any]:
    dataset = list(model / "datasets")[-1]
    selector: dict[str, Any] = {}
    inner_indices, _ = model.inner(dataset)
    if len(inner_indices):
        selector["inner"] = "last"
    temperature, speed, pressure = model.evaluate(
        ["T", "hmnf.U", "hmnf.pA"],
        unit=["K", "m/s", "Pa"],
        dataset=dataset,
        **selector,
    )
    mass_fluxes: dict[str, float] = {}
    energy_fluxes: dict[str, float] = {}
    for name, operator in (
        ("nozzle_outward", "int_nozzle_hmnf"),
        ("ambient_outward", "int_ambient_hmnf"),
        ("target_outward", "int_target_hmnf"),
    ):
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
    component = model.java.component("comp1")
    coupling_tags = {
        str(value) for value in component.cpl().tags()
    }
    wall_operator = (
        "int_wall_hmnf"
        if "int_wall_hmnf" in coupling_tags
        else "int_target_hmnf"
    )
    all_walls_outward = _scalar(
        model.evaluate(
            f"{wall_operator}(2*pi*r*hmnf.rho*(u*nr+w*nz))",
            unit="kg/s",
            dataset=dataset,
            **selector,
        )
    )
    physical_boundary_sum = (
        mass_fluxes["nozzle_outward"]
        + mass_fluxes["ambient_outward"]
        + all_walls_outward
    )
    wall_upwind_weak_flux = _scalar(
        model.evaluate(
            f"{wall_operator}("
            "2*pi*r*hmnf.contCoeffFace*hmnf.unJump)",
            unit="kg/s",
            dataset=dataset,
            **selector,
        )
    )
    discrete_corrected_residual = abs(
        physical_boundary_sum - wall_upwind_weak_flux
    )
    inlet_mass = abs(mass_fluxes["nozzle_outward"])
    energy_balance = _scalar(
        model.evaluate(
            "hmnf.energyBalance",
            unit="W",
            dataset=dataset,
            **selector,
        )
    )
    inlet_energy = abs(energy_fluxes["nozzle_outward"])
    return {
        "temperature_k": _range(temperature),
        "speed_m_s": _range(speed),
        "absolute_pressure_pa": _range(pressure),
        "one_mm_upstream_of_target": {
            "temperature_k": _scalar(
                model.evaluate(
                    "at2(1e-6[m],d_spray-1[mm],T)",
                    unit="K",
                    dataset=dataset,
                    **selector,
                )
            ),
            "speed_m_s": _scalar(
                model.evaluate(
                    "at2(1e-6[m],d_spray-1[mm],hmnf.U)",
                    unit="m/s",
                    dataset=dataset,
                    **selector,
                )
            ),
        },
        "mass_flux_kg_s": {
            **mass_fluxes,
            "all_walls_outward": all_walls_outward,
            "wall_operator": wall_operator,
            "physical_boundary_sum": physical_boundary_sum,
            "physical_boundary_absolute_residual": abs(
                physical_boundary_sum
            ),
            "wall_upwind_weak_flux": wall_upwind_weak_flux,
            "discrete_corrected_absolute_residual": (
                discrete_corrected_residual
            ),
            "imbalance_fraction": (
                discrete_corrected_residual / inlet_mass
            ),
            "definition": (
                "abs(sum(rho*u.n)-integral_wall("
                "contCoeffFace*unJump))/abs(nozzle_mass_flow)"
            ),
        },
        "energy_balance_w": {
            **energy_fluxes,
            "comsol_energy_balance": energy_balance,
            "imbalance_fraction_of_inlet": abs(energy_balance) / inlet_energy,
            "definition": "hmnf.energyBalance=dEi0Int+ntefluxInt-WInt-QInt",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model-in", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.model_in.exists():
        raise FileNotFoundError(args.model_in)
    base_contract = ConservativeRestartContract()
    base_contract.validate()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.model_in))
        model.rename("h11_target_impact_conservative_restart")
        jm = model.java
        contract = replace(
            base_contract,
            source_solution_tag=_last_tag(jm.sol()),
            source_study_tag=_last_tag(jm.study()),
        )
        source_audit = validate_source_model(jm, contract)
        replace_with_conservative_rans(jm, contract)
        first_solution, refine_solution, solver_strategy = configure_restart_studies(
            jm,
            contract,
        )
        try:
            jm.study("std_hmnf").run()
            jm.study("std_hmnf_refine").run()
        except Exception as exc:
            partial = args.model_out.with_name(f"{args.model_out.stem}_partial.mph")
            model.save(str(partial))
            failure = {
                "schema_version": "h11_conservative_restart_failure_v1",
                "status": "failed_final_equation_restart",
                "contract": asdict(contract),
                "source_audit": source_audit,
                "solver_strategy": solver_strategy,
                "error": str(exc),
                "partial_model": str(partial.resolve()),
                "partial_model_sha256": _sha256(partial),
            }
            failure_path = args.audit.with_name(f"{args.audit.stem}_failure.json")
            with failure_path.open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
            print(f"Wrote failure audit: {failure_path}", flush=True)
            raise
        model.save(str(args.model_out))
        metrics = evaluate_solution(model)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_audit = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
    finally:
        client.clear()

    bounded = bool(
        metrics["temperature_k"]["minimum"]
        > contract.property_temperature_floor_k
        and metrics["temperature_k"]["maximum"] <= 10_000.01
        and metrics["speed_m_s"]["minimum"] >= -1e-9
        and metrics["speed_m_s"]["maximum"] <= 600.01
        and metrics["absolute_pressure_pa"]["minimum"]
        > contract.pressure_floor_pa
    )
    mass_pass = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    energy_pass = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    audit = {
        "schema_version": "h11_conservative_restart_v1",
        "status": (
            "pass_final_equations_numerical_gates_not_calibrated"
            if bounded and mass_pass and energy_pass
            else "pass_solve_fail_one_or_more_numerical_gates"
        ),
        "contract": asdict(contract),
        "source_role": (
            "Nonlinear initial guess only. The reported fields and balances "
            "are recomputed by the final all-Mach k-omega equations."
        ),
        "source_audit": source_audit,
        "solver_strategy": solver_strategy,
        "initial_solution": first_solution,
        "refinement_solution": refine_solution,
        "mesh": mesh_audit,
        "metrics": metrics,
        "gates": {
            "physically_bounded": bounded,
            "mass_imbalance_below_0_5_percent": mass_pass,
            "energy_imbalance_below_2_percent": energy_pass,
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
        "runtime_sec": time.time() - started,
        "comsol_version": args.version,
        "cores": args.cores,
        "source_model": str(args.model_in.resolve()),
        "source_sha256": _sha256(args.model_in),
        "model_path": str(args.model_out.resolve()),
        "model_sha256": _sha256(args.model_out),
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {args.model_out}")
    print(f"Wrote audit: {args.audit}")
    print(
        f"Conservative restart: {audit['status']}; "
        f"mass={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
