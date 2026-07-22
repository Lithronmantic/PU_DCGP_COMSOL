"""Resume the true fine-mesh gas solve from the audited load-0.35 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    _configure_segregated_solver,
    _evaluate_many,
    _last_dataset,
    _range,
    _scalar,
    corrected_mass_imbalance_fraction,
    dpv_profile_expressions,
)
from simulator_v2.phase_h.h11_corrected_gas_fine_resume_contract import (
    CONTRACT_PATH,
    FineResumeContract,
    MODEL_DIR,
    OUTPUT_DIR,
    SOURCE_PARTIAL_MODEL,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_convergence import (
    _configure_projected_study,
    _domain_parameters,
    _mesh_audit,
    _solution_for_study,
)
from simulator_v2.phase_h.h11_effective_exit_directional import _gas_gates
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


MODEL_PATH = MODEL_DIR / "corrected_t11160_u1090_mesh_level2_resumed.mph"
AUDIT_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_resume_audit.json"
LOG_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_resume.log"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _partial_parameter_values(model: Any) -> tuple[Any, list[float]]:
    candidates: list[tuple[Any, list[float]]] = []
    for dataset in model / "datasets":
        try:
            _, raw_values = model.inner(dataset)
        except Exception:
            continue
        values = [float(value) for value in np.asarray(raw_values).reshape(-1)]
        if values and all(math.isfinite(value) for value in values):
            candidates.append((dataset, values))
    if not candidates:
        raise RuntimeError("The partial model has no parameterized dataset")
    return max(candidates, key=lambda item: (item[1][-1], len(item[1])))


def _configure_resume_study(
    jm: Any,
    source_solution: str,
    contract: FineResumeContract,
) -> tuple[str, str, dict[str, Any]]:
    study_tag = "std_fine_resume"
    if study_tag in {str(tag) for tag in jm.study().tags()}:
        jm.study().remove(study_tag)
    study = jm.study().create(study_tag)
    study.label("True fine-mesh deterministic load resume")
    step = study.create("stat", "Stationary")
    activation: list[str] = []
    for tag in (str(value) for value in jm.component("comp1").physics().tags()):
        activation.extend([tag, "on" if tag == "hmnf" else "off"])
    step.set("activate", activation)
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", "std1")
    step.set("initstudystep", "stat")
    step.set("initsol", source_solution)
    step.set("solnum", "last")
    parametric = study.create("param", "Parametric")
    parametric.set("pname", ["load_s"])
    parametric.set(
        "plistarr",
        [" ".join(f"{value:.12g}" for value in contract.resume_fractions)],
    )
    parametric.set("punit", [""])
    parametric.set("sweeptype", "filled")
    parametric.set("reusesol", "on")
    parametric.set("keepsol", "all")
    study.createAutoSequences("all")
    solution_tag = _solution_for_study(jm, study_tag)
    base = FreeJetSolveContract()
    solver = _configure_segregated_solver(
        jm,
        solution_tag,
        tolerance=contract.continuation_relative_tolerance,
        maximum_iterations=base.maximum_segregated_iterations,
        contract=base,
    )
    return study_tag, solution_tag, solver


def _evaluate_latest(
    model: Any,
    base: FreeJetSolveContract,
    n_continuation_solutions: int,
) -> dict[str, Any]:
    dataset, selector = _last_dataset(model)
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
            "int_torch_hmnf(2*pi*r*hmnf.contCoeffFace*hmnf.unJump)",
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
        model.evaluate("hmnf.energyBalance", unit="W", dataset=dataset, **selector)
    )
    inlet_energy = abs(energy_fluxes["nozzle_outward"])
    zero_flux = "0[kg/(m^2*s)]"
    axial_mass_density = "hmnf.rho*w"
    dpv_expressions = (
        f"int_dpv_hmnf(2*pi*r*({axial_mass_density}))",
        f"int_dpv_hmnf(2*pi*r*max({axial_mass_density},{zero_flux}))",
        f"-int_dpv_hmnf(2*pi*r*min({axial_mass_density},{zero_flux}))",
        f"int_dpv_hmnf(2*pi*r*max({axial_mass_density},{zero_flux})*T)",
        f"int_dpv_hmnf(2*pi*r*max({axial_mass_density},{zero_flux})*hmnf.U)",
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
    profile: dict[str, list[float]] = {}
    for name, field, unit in (
        ("temperature_k", "T", "K"),
        ("speed_m_s", "hmnf.U", "m/s"),
        ("axial_velocity_m_s", "w", "m/s"),
        ("absolute_pressure_pa", "p", "Pa"),
    ):
        expressions = dpv_profile_expressions(base, field)
        profile[name] = _evaluate_many(
            model,
            expressions,
            [unit] * len(expressions),
            dataset,
            selector,
        )
    return {
        "n_continuation_solutions": n_continuation_solutions,
        "full_load_fraction": 1.0,
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
            "definition": "abs(nozzle+open+torch-torch_DG_weak_flux)/abs(nozzle)",
        },
        "energy_balance_w": {
            **energy_fluxes,
            "comsol_energy_balance": energy_balance,
            "imbalance_fraction_of_inlet": abs(energy_balance) / inlet_energy,
            "definition": "hmnf.energyBalance=dEi0Int+ntefluxInt-WInt-QInt",
        },
        "fixed_dpv_plane_gas_diagnostics": {
            "z_mm": base.dpv_plane_mm,
            "net_axial_mass_flux_kg_s": dpv_net_mass,
            "forward_axial_mass_flux_kg_s": dpv_forward_mass,
            "reverse_axial_mass_flux_kg_s": dpv_reverse_mass,
            "forward_mass_weighted_temperature_k": (
                dpv_temperature_moment / dpv_forward_mass
            ),
            "forward_mass_weighted_speed_m_s": dpv_speed_moment / dpv_forward_mass,
            "profile_radii_mm": list(base.dpv_profile_radii_mm),
            "gas_profile": profile,
            "interpretation": "Gas-phase state only; not a DPV observation.",
        },
    }


def solve(client: Any) -> dict[str, Any]:
    contract = FineResumeContract()
    contract.validate()
    base = FreeJetSolveContract()
    base.validate()
    for path in (MODEL_PATH, AUDIT_PATH, LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    model = client.load(str(SOURCE_PARTIAL_MODEL))
    model.rename(MODEL_PATH.stem)
    jm = model.java
    client.java.showProgress(str(LOG_PATH.resolve()))
    started = time.time()
    try:
        _, prefix_values = _partial_parameter_values(model)
        if not math.isclose(
            prefix_values[-1],
            contract.source_last_successful_load,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"Partial model ends at load {prefix_values[-1]}, not 0.35"
            )
        mesh = _mesh_audit(jm, contract.automatic_mesh_level)
        if mesh["elements"] != contract.expected_mesh_elements:
            raise RuntimeError("The partial checkpoint is not the frozen fine mesh")
        source_solution = _solution_for_study(jm, "std1")
        study_tag, solution_tag, resume_solver = _configure_resume_study(
            jm, source_solution, contract
        )
        print(
            f"resuming {mesh['elements']}-element solve from load 0.35 "
            f"through {len(contract.resume_fractions)} frozen steps",
            flush=True,
        )
        resume_started = time.time()
        jm.study(study_tag).run()
        resume_runtime = time.time() - resume_started
        print("fine-mesh resume reached full load", flush=True)
        refine_study, refine_solution, refine_solver = _configure_projected_study(
            jm,
            source_study=study_tag,
            source_solution=solution_tag,
            study_tag="std_fine_resume_refine",
            study_label="True fine-mesh resumed full-load 1e-6 refinement",
            solver_mode="ordinary_strict",
            tolerance=contract.final_relative_tolerance,
        )
        refine_started = time.time()
        jm.study(refine_study).run()
        refine_runtime = time.time() - refine_started
        print("fine-mesh resumed 1e-6 refinement complete", flush=True)
        metrics = _evaluate_latest(
            model,
            base,
            len(prefix_values) + len(contract.resume_fractions),
        )
        gates = _gas_gates(metrics)
        domain = _domain_parameters(jm)
        gates.update(
            {
                "true_level2_mesh_unchanged": (
                    _mesh_audit(jm, contract.automatic_mesh_level)["elements"]
                    == contract.expected_mesh_elements
                ),
                "resume_reached_full_load": contract.resume_fractions[-1] == 1.0,
                "strict_refinement_completed": True,
                "fixed_40_by_140_mm_domain": domain
                == {"r_domain": 40.0, "z_domain": 140.0},
            }
        )
        model.save(str(MODEL_PATH))
        return {
            "schema_version": "h11_corrected_gas_fine_resume_audit_v1",
            "status": (
                "pass_corrected_gas_fine_resume"
                if all(gates.values())
                else "fail_corrected_gas_fine_resume_gates"
            ),
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_partial_model": str(SOURCE_PARTIAL_MODEL.resolve()),
            "source_partial_model_sha256": _sha256(SOURCE_PARTIAL_MODEL),
            "stored_prefix_fractions": prefix_values,
            "resume_fractions": list(contract.resume_fractions),
            "mesh": mesh,
            "domain_mm": domain,
            "resume_solver": resume_solver,
            "refinement_study": refine_study,
            "refinement_solution": refine_solution,
            "refinement_solver": refine_solver,
            "resume_runtime_sec": resume_runtime,
            "refinement_runtime_sec": refine_runtime,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(MODEL_PATH.resolve()),
            "model_sha256": _sha256(MODEL_PATH),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    except Exception as exc:
        partial = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_partial.mph")
        model.save(str(partial))
        failure = {
            "schema_version": "h11_corrected_gas_fine_resume_failure_v1",
            "status": "failed_corrected_gas_fine_resume_solve",
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_partial_model": str(SOURCE_PARTIAL_MODEL.resolve()),
            "source_partial_model_sha256": _sha256(SOURCE_PARTIAL_MODEL),
            "error": str(exc),
            "partial_model": str(partial.resolve()),
            "partial_model_sha256": _sha256(partial),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_failure.json"), failure)
        raise
    finally:
        client.java.showProgress(False)
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        payload = solve(client)
    finally:
        client.clear()
    _write_json(AUDIT_PATH, payload)
    print(AUDIT_PATH)
    print(payload["status"])
    return 0 if payload["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
