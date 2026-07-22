
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.conservative_free_jet_nominal import (
    AUDIT_PATH as SOURCE_AUDIT,
    MODEL_PATH as SOURCE_MODEL,
    FreeJetSolveContract,
    evaluate_solution,
)
from pu_dcgp_comsol.comsol.conservative_free_jet_residual_audit import (
    fixed_dpv_relative_changes,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_fully_coupled_refinement import (
    FullyCoupledRefinementContract,
    configure_fully_coupled_refinement,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_residual_localization import (
    _replace_maximum_coupling,
    _residual_localization,
    parse_detailed_residual_log,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_same_mesh_refinement import (
    _last_tag,
    _sha256,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_conservative_free_jet_fully_coupled"
OUT_DIR = HERE / "h11_outputs" / "conservative_free_jet_fully_coupled"
MODEL_PATH = MODEL_DIR / "h11_conservative_free_jet_fully_coupled_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_conservative_free_jet_fully_coupled_audit.json"
LOG_PATH = OUT_DIR / "h11_conservative_free_jet_fully_coupled.log"


@dataclass(frozen=True)
class FreeJetFullyCoupledContract:

    relative_tolerance: float = 1e-6


    maximum_iterations: int = 300
    initial_damping: float = 0.1
    minimum_damping: float = 1e-8
    recovery_damping: float = 0.1
    residual_factor: float = 1.0
    observable_relative_change_limit: float = 5e-6
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 < self.relative_tolerance <= 1e-5:
            raise ValueError("Relative tolerance must lie in (0, 1e-5]")
        if self.maximum_iterations < 20:
            raise ValueError("Fully coupled iteration window is too short")
        damping = (
            self.initial_damping,
            self.minimum_damping,
            self.recovery_damping,
        )
        if any(not math.isfinite(value) or value <= 0 for value in damping):
            raise ValueError("Newton damping values must be finite and positive")
        if self.minimum_damping >= self.initial_damping:
            raise ValueError("Minimum damping must be below initial damping")
        if not self.initial_damping <= self.recovery_damping <= 1:
            raise ValueError(
                "Recovery damping must lie between initial damping and one"
            )
        if not 0 < self.residual_factor <= 1:
            raise ValueError("Residual factor must lie in (0, 1]")
        if not 0 < self.observable_relative_change_limit <= 1e-3:
            raise ValueError("Observable-stability limit is not credible")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError(
                "Numerical refinement cannot claim calibration or prediction"
            )

    def solver_contract(self) -> FullyCoupledRefinementContract:
        self.validate()
        return FullyCoupledRefinementContract(
            relative_tolerance=self.relative_tolerance,
            maximum_iterations=self.maximum_iterations,
            initial_damping=self.initial_damping,
            minimum_damping=self.minimum_damping,
            recovery_damping=self.recovery_damping,
            residual_factor=self.residual_factor,
        )


def _default_paths(build_only: bool) -> tuple[Path, Path, Path]:
    if not build_only:
        return MODEL_PATH, AUDIT_PATH, LOG_PATH
    return (
        MODEL_PATH.with_name(f"{MODEL_PATH.stem}_skeleton.mph"),
        AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_skeleton.json"),
        LOG_PATH.with_name(f"{LOG_PATH.stem}_skeleton.log"),
    )


def assess_numerical_result(
    *,
    strict_solver_converged: bool,
    source_metrics: dict[str, Any],
    refined_metrics: dict[str, Any],
    observable_changes: dict[str, float],
    free_jet_contract: FreeJetSolveContract,
    refinement_contract: FreeJetFullyCoupledContract,
) -> dict[str, Any]:

    finite_changes = all(
        math.isfinite(value) for value in observable_changes.values()
    )
    observable_stable = finite_changes and all(
        value <= refinement_contract.observable_relative_change_limit
        for value in observable_changes.values()
    )
    mass_ok = (
        float(refined_metrics["mass_flux_kg_s"]["imbalance_fraction"])
        <= free_jet_contract.mass_imbalance_limit_fraction
    )
    energy_ok = (
        float(refined_metrics["energy_balance_w"]["imbalance_fraction_of_inlet"])
        <= free_jet_contract.energy_imbalance_limit_fraction
    )
    source_mass = float(
        source_metrics["mass_flux_kg_s"]["imbalance_fraction"]
    )
    source_energy = float(
        source_metrics["energy_balance_w"]["imbalance_fraction_of_inlet"]
    )
    return {
        "strict_equation_convergence": bool(strict_solver_converged),
        "fixed_dpv_gas_observables_stable": bool(observable_stable),
        "mass_conservation_pass": bool(mass_ok),
        "energy_conservation_pass": bool(energy_ok),
        "source_mass_imbalance_fraction": source_mass,
        "source_energy_imbalance_fraction": source_energy,
        "refined_mass_imbalance_fraction": float(
            refined_metrics["mass_flux_kg_s"]["imbalance_fraction"]
        ),
        "refined_energy_imbalance_fraction": float(
            refined_metrics["energy_balance_w"][
                "imbalance_fraction_of_inlet"
            ]
        ),
        "numerically_accepted": bool(
            strict_solver_converged and observable_stable and mass_ok and energy_ok
        ),
        "interpretation": (
            "Strict equation convergence is mandatory for this branch. "
            "Observable stability alone is reported but cannot relabel a "
            "failed residual-controlled solve."
        ),
    }


def _safe_metrics(
    model: Any,
    contract: FreeJetSolveContract,
) -> tuple[dict[str, Any], str | None]:
    try:
        return evaluate_solution(model, contract), None
    except Exception as exc:
        return {}, str(exc)


def refinement_study_tag(source_study_tag: str) -> str:

    primary = "std_hmnf_fully_coupled_refine"
    if source_study_tag != primary:
        return primary
    return "std_hmnf_fully_coupled_verify"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
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
    if not args.source_audit.exists():
        raise FileNotFoundError(args.source_audit)
    model_default, audit_default, log_default = _default_paths(args.build_only)
    model_out = args.model_out or model_default
    audit_path = args.audit or audit_default
    log_path = args.log or log_default
    for path in (model_out, audit_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    free_jet_contract = FreeJetSolveContract()
    refinement_contract = FreeJetFullyCoupledContract()
    free_jet_contract.validate()
    refinement_contract.validate()

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        client.java.showProgress(str(log_path.resolve()))
        model = client.load(str(args.source_model))
        jm = model.java
        source_study_tag = _last_tag(jm.study())
        source_solution_tag = _last_tag(jm.sol())
        source_metrics = evaluate_solution(model, free_jet_contract)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver = configure_fully_coupled_refinement(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=refinement_contract.solver_contract(),
            target_study_tag=refinement_study_tag(source_study_tag),
        )
        stationary = jm.sol(solution_tag).feature("s1")
        advanced = stationary.feature("aDef")
        advanced.set("storeresidual", "solvingandoutput")
        advanced.set("convinfo", "detailed")
        advanced.set("checkmatherr", "on")
        component = jm.component("comp1")
        _replace_maximum_coupling(component, "max_res_domain", 2)
        wall_entities = [
            int(value)
            for value in component.physics("hmnf")
            .feature("wallbc1")
            .selection()
            .entities()
        ]
        _replace_maximum_coupling(
            component,
            "max_res_boundary",
            1,
            wall_entities,
        )

        strict_solver_converged = False
        solver_error = None
        refined_metrics: dict[str, Any] = {}
        metrics_error = None
        raw_residuals: dict[str, Any] = {}
        observable_changes: dict[str, float] = {}
        decision: dict[str, Any] = {}
        if args.build_only:
            status = "pass_free_jet_fully_coupled_skeleton"
        else:
            try:
                jm.study(study_tag).run()
                strict_solver_converged = True
            except Exception as exc:
                solver_error = str(exc)
            refined_metrics, metrics_error = _safe_metrics(
                model,
                free_jet_contract,
            )
            if refined_metrics:
                observable_changes = fixed_dpv_relative_changes(
                    source_metrics,
                    refined_metrics,
                )
                raw_residuals = _residual_localization(model)
                decision = assess_numerical_result(
                    strict_solver_converged=strict_solver_converged,
                    source_metrics=source_metrics,
                    refined_metrics=refined_metrics,
                    observable_changes=observable_changes,
                    free_jet_contract=free_jet_contract,
                    refinement_contract=refinement_contract,
                )
            if strict_solver_converged and decision.get(
                "numerically_accepted",
                False,
            ):
                status = "pass_strict_fully_coupled_refinement"
            elif strict_solver_converged:
                status = "fail_postsolve_numerical_gate"
            else:
                status = "fail_strict_fully_coupled_refinement"

        model.rename(model_out.stem)
        model.save(str(model_out))
        mesh_after_feature = jm.component("comp1").mesh("mesh1")
        mesh_after = {
            "elements": int(mesh_after_feature.getNumElem()),
            "vertices": int(mesh_after_feature.getNumVertex()),
            "minimum_quality": float(mesh_after_feature.getMinQuality()),
            "mean_quality": float(mesh_after_feature.getMeanQuality()),
        }
        if mesh_after != mesh_before:
            raise RuntimeError("Fully coupled refinement changed the mesh")
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        payload = {
            "schema_version": "h11_free_jet_fully_coupled_v1",
            "status": status,
            "free_jet_contract": asdict(free_jet_contract),
            "refinement_contract": asdict(refinement_contract),
            "strategy": {
                "geometry_changed": False,
                "mesh_changed": False,
                "physics_changed": False,
                "material_changed": False,
                "boundary_conditions_changed": False,
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "target_study": study_tag,
                "target_solution": solution_tag,
            },
            "solver": solver,
            "mesh": mesh_after,
            "source_metrics": source_metrics,
            "refined_metrics": refined_metrics,
            "metrics_error": metrics_error,
            "fixed_dpv_relative_changes": observable_changes,
            "raw_residual_localization": raw_residuals,
            "detailed_convergence": parse_detailed_residual_log(log_text),
            "numerical_decision": decision,
            "solver_error": solver_error,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.java.showProgress(False)
        client.clear()

    payload.update(
        {
            "runtime_sec": time.time() - started,
            "comsol_version": args.version,
            "cores": args.cores,
            "source_model": str(args.source_model.resolve()),
            "source_sha256": _sha256(args.source_model),
            "source_audit": str(args.source_audit.resolve()),
            "source_audit_sha256": _sha256(args.source_audit),
            "model_path": str(model_out.resolve()),
            "model_sha256": _sha256(model_out),
            "log_path": str(log_path.resolve()),
        }
    )
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_out}")
    print(f"Wrote audit: {audit_path}")
    print(f"Free-jet fully coupled refinement: {status}")
    return int(status.startswith("fail_"))


if __name__ == "__main__":
    raise SystemExit(main())
