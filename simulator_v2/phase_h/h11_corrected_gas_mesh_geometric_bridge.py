"""Solve the single geometric-mean bridge before automatic mesh level 2.

The first initialization bridge and the automatic level-2 mesh differ by
almost a factor of two in element count.  This layer inserts one fixed mesh
whose global size parameters are the geometric means of those two endpoints.
It changes neither geometry nor physics and is never an evaluation mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_convergence import (
    MODEL_DIR,
    OUTPUT_DIR,
    _configure_projected_study,
    _domain_parameters,
    _mesh_audit,
    _sha256,
    _solution_for_study,
    _write_json,
    bridge_paths,
    gas_paths,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_convergence_contract import (
    CorrectedGasMeshConvergenceContract,
)
from simulator_v2.phase_h.h11_effective_exit_directional import _gas_gates


STEM = "corrected_t11160_u1090_mesh_geometric_bridge_gas"
MODEL_PATH = MODEL_DIR / f"{STEM}.mph"
AUDIT_PATH = OUTPUT_DIR / "cases" / f"{STEM}.json"
LOG_PATH = OUTPUT_DIR / "logs" / f"{STEM}.log"


def artifact_paths(urans_preconditioner: bool) -> dict[str, Path]:
    if not urans_preconditioner:
        return {
            "model": MODEL_PATH,
            "audit": AUDIT_PATH,
            "log": LOG_PATH,
        }
    stem = "corrected_t11160_u1090_mesh_geometric_urans_preconditioner"
    return {
        "model": MODEL_DIR / f"{stem}.mph",
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
        "log": OUTPUT_DIR / "logs" / f"{stem}.log",
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_geometric_bridge_mesh(jm: Any) -> dict[str, Any]:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    mesh = jm.component("comp1").mesh("mesh1")
    mesh.autoMeshSize(3)
    size = mesh.feature("size")
    size.set("custom", "on")
    size.set(
        "hmax",
        f"{contract.geometric_bridge_default_hmax_mm:.12g}[mm]",
    )
    size.set(
        "hmin",
        f"{contract.geometric_bridge_default_hmin_mm:.12g}[mm]",
    )
    size.set(
        "hgrad",
        f"{contract.geometric_bridge_default_growth_rate:.12g}",
    )
    size.set(
        "hcurve",
        f"{contract.geometric_bridge_default_curvature_factor:.12g}",
    )
    mesh.run()
    return {
        **_mesh_audit(jm, "geometric_bridge"),
        "role": "initialization_only_not_a_convergence_evaluation_level",
        "default_size_settings": {
            key: str(size.getString(key))
            for key in (
                "custom",
                "hmax",
                "hmin",
                "hgrad",
                "hcurve",
                "hnarrow",
            )
        },
        "construction": (
            "geometric means of the first bridge and automatic level-2 "
            "hmax, hmin, and growth rate"
        ),
    }


def solve(
    client: Any,
    *,
    urans_preconditioner: bool = False,
) -> dict[str, Any]:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    paths = artifact_paths(urans_preconditioner)
    relative_tolerance = (
        contract.level2_preconditioner_tolerance
        if urans_preconditioner
        else 1e-6
    )
    source_audit = _load_json(bridge_paths()["audit"])
    if source_audit["status"] != "pass_conditional_mesh_bridge":
        raise RuntimeError("The first conditional bridge has not passed")
    level2_failure = gas_paths(2)["audit"].with_name(
        f"{gas_paths(2)['audit'].stem}_failure.json"
    )
    if not level2_failure.exists():
        raise RuntimeError("The failed final level-2 audit is missing")

    source_model = bridge_paths()["model"]
    source_study = "std_mesh_bridge"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    model = client.load(str(source_model))
    model.rename(paths["model"].stem)
    jm = model.java
    client.java.showProgress(str(paths["log"].resolve()))
    started = time.time()
    try:
        source_solution = _solution_for_study(jm, source_study)
        source_mesh = _mesh_audit(jm, "conditional_bridge")
        source_domain = _domain_parameters(jm)
        target_mesh = configure_geometric_bridge_mesh(jm)
        study_tag, solution_tag, solver = _configure_projected_study(
            jm,
            source_study=source_study,
            source_solution=source_solution,
            study_tag="std_mesh_geometric_bridge",
            study_label=(
                "Corrected free-jet geometric-mean "
                + (
                    "URANS preconditioner"
                    if urans_preconditioner
                    else "initialization bridge"
                )
            ),
            solver_mode="ordinary_strict",
            tolerance=relative_tolerance,
        )
        print(
            "  gas bridge-1->geometric-bridge: "
            f"{source_mesh['elements']}->{target_mesh['elements']} elements",
            flush=True,
        )
        try:
            jm.study(study_tag).run()
        except Exception as exc:
            partial_model = paths["model"].with_name(
                f"{paths['model'].stem}_partial.mph"
            )
            model.save(str(partial_model))
            failure = {
                "schema_version": (
                    "h11_corrected_gas_geometric_bridge_failure_v1"
                ),
                "status": "failed_geometric_bridge_equation_solve",
                "source_model": str(source_model.resolve()),
                "source_model_sha256": _sha256(source_model),
                "source_study": source_study,
                "source_solution": source_solution,
                "target_study": study_tag,
                "target_solution": solution_tag,
                "source_mesh": source_mesh,
                "target_mesh": target_mesh,
                "domain_mm": _domain_parameters(jm),
                "solver": solver,
                "error": str(exc),
                "partial_model": str(partial_model.resolve()),
                "partial_model_sha256": _sha256(partial_model),
                "runtime_sec": time.time() - started,
                "calibrated": False,
                "paper_prediction_allowed": False,
            }
            failure_path = paths["audit"].with_name(
                f"{paths['audit'].stem}_failure.json"
            )
            _write_json(failure_path, failure)
            raise

        metrics = evaluate_solution(model, FreeJetSolveContract())
        gates = _gas_gates(metrics)
        domain = _domain_parameters(jm)
        gates["fixed_40_by_140_mm_domain"] = (
            domain == source_domain
            and math.isclose(domain["r_domain"], 40.0)
            and math.isclose(domain["z_domain"], 140.0)
        )
        gates["elements_strictly_between_bridges"] = (
            source_mesh["elements"]
            < target_mesh["elements"]
            < 55_291
        )
        gates["positive_mesh_quality"] = target_mesh["minimum_quality"] > 0
        model.save(str(paths["model"]))
        success_status = (
            "pass_geometric_mesh_urans_preconditioner"
            if urans_preconditioner
            else "pass_geometric_mesh_bridge"
        )
        payload = {
            "schema_version": (
                "h11_corrected_gas_geometric_bridge_case_v1"
            ),
            "status": (
                success_status
                if all(gates.values())
                else "fail_geometric_mesh_bridge_gates"
            ),
            "role": (
                "urans_initialization_only_not_a_convergence_evaluation_level"
                if urans_preconditioner
                else "initialization_only_not_a_convergence_evaluation_level"
            ),
            "relative_tolerance": relative_tolerance,
            "triggering_level2_failure": str(level2_failure.resolve()),
            "triggering_level2_failure_sha256": _sha256(level2_failure),
            "source_model": str(source_model.resolve()),
            "source_model_sha256": _sha256(source_model),
            "source_study": source_study,
            "source_solution": source_solution,
            "target_study": study_tag,
            "target_solution": solution_tag,
            "source_mesh": source_mesh,
            "target_mesh": target_mesh,
            "domain_mm": domain,
            "solver": solver,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["model"].resolve()),
            "model_sha256": _sha256(paths["model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(paths["audit"], payload)
        return payload
    finally:
        client.java.showProgress(False)
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--urans-preconditioner", action="store_true")
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        result = solve(
            client,
            urans_preconditioner=args.urans_preconditioner,
        )
    finally:
        client.clear()
    print(
        f"{result['status']}; "
        f"elements={result['target_mesh']['elements']}; "
        f"runtime={result['runtime_sec']:.1f}s"
    )
    accepted = {
        "pass_geometric_mesh_bridge",
        "pass_geometric_mesh_urans_preconditioner",
    }
    return 0 if result["status"] in accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
