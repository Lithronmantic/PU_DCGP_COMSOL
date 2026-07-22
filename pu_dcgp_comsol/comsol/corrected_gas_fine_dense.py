
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from pu_dcgp_comsol.comsol.conservative_free_jet_nominal import (
    FreeJetSolveContract,
    configure_studies,
    evaluate_solution,
)
from pu_dcgp_comsol.comsol.corrected_gas_fine_continuation_contract import (
    GAS_SKELETON_MODEL,
    MODEL_DIR,
    OUTPUT_DIR,
)
from pu_dcgp_comsol.comsol.corrected_gas_fine_dense_contract import (
    CONTRACT_PATH,
    FineDenseContract,
)
from pu_dcgp_comsol.comsol.corrected_gas_mesh_convergence import (
    _domain_parameters,
    _mesh_audit,
)
from pu_dcgp_comsol.comsol.effective_exit_directional import _gas_gates
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import _sha256


MODEL_PATH = MODEL_DIR / "corrected_t11160_u1090_mesh_level2_dense.mph"
AUDIT_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_dense_audit.json"
LOG_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_dense.log"


def _write_json(path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def solve(client: Any) -> dict[str, Any]:
    contract = FineDenseContract()
    contract.validate()
    numerical = FreeJetSolveContract(load_fractions=contract.load_fractions)
    numerical.validate()
    for path in (MODEL_PATH, AUDIT_PATH, LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    model = client.load(str(GAS_SKELETON_MODEL))
    model.rename(MODEL_PATH.stem)
    jm = model.java
    client.java.showProgress(str(LOG_PATH.resolve()))
    started = time.time()
    try:
        jm.param().set(
            "T_exit_eff", f"{contract.effective_exit_temperature_k:.12g}[K]"
        )
        jm.param().set(
            "u_exit_eff", f"{contract.effective_exit_speed_m_s:.12g}[m/s]"
        )
        solver = configure_studies(jm, numerical)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh.autoMeshSize(contract.automatic_mesh_level)
        mesh.feature("size").set("hauto", str(contract.automatic_mesh_level))
        mesh.feature("size").set("custom", "off")
        mesh.run()
        mesh_audit = _mesh_audit(jm, contract.automatic_mesh_level)
        if mesh_audit["elements"] != contract.expected_mesh_elements:
            raise RuntimeError("The generated mesh is not the frozen 55,291-element mesh")
        print(
            f"dense fine-mesh continuation: {mesh_audit['elements']} elements, "
            f"{len(contract.load_fractions)} loads",
            flush=True,
        )
        continuation_started = time.time()
        jm.study("std1").run()
        continuation_runtime = time.time() - continuation_started
        print("dense fine-mesh continuation reached full load", flush=True)
        refinement_started = time.time()
        jm.study("std_refine").run()
        refinement_runtime = time.time() - refinement_started
        print("dense fine-mesh 1e-6 refinement complete", flush=True)
        metrics = evaluate_solution(model, numerical)
        gates = _gas_gates(metrics)
        domain = _domain_parameters(jm)
        gates.update(
            {
                "true_level2_mesh": mesh_audit["elements"]
                == contract.expected_mesh_elements,
                "positive_mesh_quality": mesh_audit["minimum_quality"] > 0.0,
                "full_dense_ladder_stored": metrics["n_continuation_solutions"]
                == len(contract.load_fractions),
                "fixed_40_by_140_mm_domain": domain
                == {"r_domain": 40.0, "z_domain": 140.0},
            }
        )
        model.save(str(MODEL_PATH))
        return {
            "schema_version": "h11_corrected_gas_fine_dense_audit_v1",
            "status": (
                "pass_corrected_gas_fine_dense"
                if all(gates.values())
                else "fail_corrected_gas_fine_dense_gates"
            ),
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_model": str(GAS_SKELETON_MODEL.resolve()),
            "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
            "mesh": mesh_audit,
            "domain_mm": domain,
            "solver": solver,
            "continuation_runtime_sec": continuation_runtime,
            "refinement_runtime_sec": refinement_runtime,
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
            "schema_version": "h11_corrected_gas_fine_dense_failure_v1",
            "status": "failed_corrected_gas_fine_dense_solve",
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_model": str(GAS_SKELETON_MODEL.resolve()),
            "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
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
