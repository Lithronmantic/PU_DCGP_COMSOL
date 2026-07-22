"""H11: localize strict residuals of the accepted nominal free-jet field.

The nominal free jet reaches the frozen 1e-6 solution-increment tolerance,
but COMSOL's detailed log reports a much larger relative pressure residual.
This diagnostic preserves geometry, mesh, physics, material, and boundary
conditions.  It performs exactly five outer updates using the existing H11
solution-and-residual localization contract, stores assembled raw residual
fields, and checks whether fixed-DPV gas diagnostics move.

Failure to meet the relative residual criterion is retained as a failure; it
is not relabeled convergence.  Raw residuals and field stability determine
the next numerical action, not experimental calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    AUDIT_PATH as SOURCE_AUDIT,
    MODEL_PATH as SOURCE_MODEL,
    FreeJetSolveContract,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_target_impact_conservative_residual_localization import (
    ResidualLocalizationContract,
    _log_tail,
    _residual_localization,
    configure_residual_localization,
    parse_detailed_residual_log,
)
from simulator_v2.phase_h.h11_target_impact_conservative_same_mesh_refinement import (
    _last_tag,
    _sha256,
    relative_change,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_conservative_free_jet_residual"
OUT_DIR = HERE / "h11_outputs" / "conservative_free_jet_residual"
MODEL_PATH = MODEL_DIR / "h11_conservative_free_jet_residual_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_conservative_free_jet_residual_audit.json"
LOG_PATH = OUT_DIR / "h11_conservative_free_jet_residual.log"


def _default_paths(build_only: bool) -> tuple[Path, Path, Path]:
    if not build_only:
        return MODEL_PATH, AUDIT_PATH, LOG_PATH
    return (
        MODEL_PATH.with_name(f"{MODEL_PATH.stem}_skeleton.mph"),
        AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_skeleton.json"),
        LOG_PATH.with_name(f"{LOG_PATH.stem}_skeleton.log"),
    )


def fixed_dpv_relative_changes(
    source: dict[str, object],
    diagnostic: dict[str, object],
) -> dict[str, float]:
    source_dpv = source["fixed_dpv_plane_gas_diagnostics"]
    diagnostic_dpv = diagnostic["fixed_dpv_plane_gas_diagnostics"]
    names = (
        "net_axial_mass_flux_kg_s",
        "forward_mass_weighted_temperature_k",
        "forward_mass_weighted_speed_m_s",
    )
    changes = {
        name: relative_change(
            float(source_dpv[name]),  # type: ignore[index]
            float(diagnostic_dpv[name]),  # type: ignore[index]
        )
        for name in names
    }
    if not all(math.isfinite(value) for value in changes.values()):
        raise ValueError("DPV diagnostic changes must be finite")
    return changes


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
    model_default, audit_default, log_default = _default_paths(
        args.build_only
    )
    model_out = args.model_out or model_default
    audit_path = args.audit or audit_default
    log_path = args.log or log_default
    for path in (model_out, audit_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    solve_contract = FreeJetSolveContract()
    solve_contract.validate()
    residual_contract = ResidualLocalizationContract()
    residual_contract.validate()

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        client.java.showProgress(str(log_path.resolve()))
        model = client.load(str(args.source_model))
        model.rename(model_out.stem)
        jm = model.java
        source_study_tag = _last_tag(jm.study())
        source_solution_tag = _last_tag(jm.sol())
        source_metrics = evaluate_solution(model, solve_contract)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver = configure_residual_localization(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=residual_contract,
        )
        if args.build_only:
            status = "pass_free_jet_residual_audit_skeleton"
            diagnostic_metrics: dict[str, Any] = {}
            raw_residuals: dict[str, Any] = {}
            changes: dict[str, float] = {}
            solver_error = None
        else:
            solver_error = None
            try:
                jm.study(study_tag).run()
                status = "diagnostic_converged_within_five_updates"
            except Exception as exc:
                solver_error = str(exc)
                if "最大分离式迭代次数" in solver_error:
                    status = (
                        "diagnostic_iteration_cap_reached_as_expected"
                    )
                elif (
                    "maximum number of segregated iterations"
                    in solver_error.lower()
                    or "鏈€澶у垎绂诲紡杩唬娆℃暟"
                    in solver_error
                ):
                    status = (
                        "diagnostic_iteration_cap_reached_as_expected"
                    )
                else:
                    status = "diagnostic_unexpected_solver_failure"
            diagnostic_metrics = evaluate_solution(model, solve_contract)
            raw_residuals = _residual_localization(model)
            changes = fixed_dpv_relative_changes(
                source_metrics,
                diagnostic_metrics,
            )
        model.save(str(model_out))
        mesh_after_feature = jm.component("comp1").mesh("mesh1")
        mesh_after = {
            "elements": int(mesh_after_feature.getNumElem()),
            "vertices": int(mesh_after_feature.getNumVertex()),
            "minimum_quality": float(
                mesh_after_feature.getMinQuality()
            ),
            "mean_quality": float(mesh_after_feature.getMeanQuality()),
        }
        if mesh_after != mesh_before:
            raise RuntimeError("Residual audit changed the mesh")
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        payload = {
            "schema_version": "h11_free_jet_residual_audit_v1",
            "status": status,
            "solve_contract": asdict(solve_contract),
            "residual_contract": asdict(residual_contract),
            "strategy": {
                "geometry_changed": False,
                "mesh_changed": False,
                "physics_changed": False,
                "material_changed": False,
                "boundary_conditions_changed": False,
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "diagnostic_study": study_tag,
                "diagnostic_solution": solution_tag,
            },
            "solver": solver,
            "mesh": mesh_after,
            "source_metrics": source_metrics,
            "diagnostic_metrics": diagnostic_metrics,
            "fixed_dpv_relative_changes": changes,
            "raw_residual_localization": raw_residuals,
            "raw_residual_interpretation": (
                "Compare a residual only within its own equation; equation "
                "units differ. COMSOL scaled estimates remain separate."
            ),
            "detailed_convergence": parse_detailed_residual_log(log_text),
            "solver_error": solver_error,
            "log_tail": _log_tail(log_path),
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
    print(f"Free-jet residual audit: {status}")
    return int(status == "diagnostic_unexpected_solver_failure")


if __name__ == "__main__":
    raise SystemExit(main())
