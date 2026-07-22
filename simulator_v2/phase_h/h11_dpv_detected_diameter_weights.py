"""Build detected-particle diameter weights for the H11 DPV operator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulator_v2.phase_h.h11_dpv_target_contract import (
    PRT_DIR,
    load_prt_particles,
)
from simulator_v2.phase_h.h11_particle_physics_contract import (
    ParticlePhysicsContract,
)


HERE = Path(__file__).resolve().parent
TARGET_CONTRACT = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "dpv_observation_operator"
    / "h11_dpv_detected_diameter_weights.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def log_midpoint_edges(nodes_um: Sequence[float]) -> np.ndarray:
    nodes = np.asarray(nodes_um, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("At least two diameter nodes are required")
    if not np.all(np.isfinite(nodes)) or not np.all(np.diff(nodes) > 0):
        raise ValueError("Diameter nodes must be finite and strictly increasing")
    return np.sqrt(nodes[:-1] * nodes[1:])


def detected_node_counts(
    detected_diameter_um: np.ndarray,
    nodes_um: Sequence[float],
) -> np.ndarray:
    values = np.asarray(detected_diameter_um, dtype=float).ravel()
    if not values.size or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Detected diameters must be positive and finite")
    bins = np.digitize(values, log_midpoint_edges(nodes_um), right=False)
    return np.bincount(bins, minlength=len(nodes_um)).astype(int)


def _weight_quantiles(run_weights: np.ndarray) -> list[dict[str, float]]:
    quantiles = np.quantile(run_weights, [0.10, 0.50, 0.90], axis=0)
    return [
        {
            "q10": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q90": float(quantiles[2, index]),
        }
        for index in range(run_weights.shape[1])
    ]


def build_detected_diameter_weight_contract() -> dict[str, Any]:
    particle = ParticlePhysicsContract()
    particle.validate()
    nodes_um = np.asarray(particle.diameter_nodes_um(), dtype=float)
    paths = [PRT_DIR / f"{run:03d}.prt" for run in range(1, 151)]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("The standard 150 A-group PRT files are required")

    run_diameters = [
        load_prt_particles(path)["particle_diameter_um"] for path in paths
    ]
    counts_per_run = np.asarray(
        [detected_node_counts(values, nodes_um) for values in run_diameters],
        dtype=int,
    )
    pooled_counts = counts_per_run.sum(axis=0)
    pooled_weights = pooled_counts / pooled_counts.sum()
    run_weights = counts_per_run / counts_per_run.sum(axis=1, keepdims=True)
    pooled = np.concatenate(run_diameters)
    below_support = float(np.mean(pooled < nodes_um[0]))
    above_support = float(np.mean(pooled > nodes_um[-1]))

    if pooled.size != 150_000:
        raise RuntimeError(f"Unexpected retained-particle count: {pooled.size}")
    if not np.isclose(pooled_weights.sum(), 1.0):
        raise RuntimeError("Detected diameter weights do not sum to one")
    if (pooled_counts == 0).any():
        raise RuntimeError("At least one numerical diameter node is unsupported")

    target = json.loads(TARGET_CONTRACT.read_text(encoding="utf-8"))
    return {
        "schema_version": "h11_dpv_detected_diameter_weights_v1",
        "status": "pass_detected_diameter_weight_contract",
        "role": (
            "A_group_pooled_DPV_detection_operator_sensitivity; "
            "not_feedstock_PSD_and_not_heldout_validation"
        ),
        "source_level": "150000_row_aligned_individually_detected_PRT_particles",
        "run_count": 150,
        "detected_particle_count": int(pooled.size),
        "diameter_nodes_um": nodes_um.tolist(),
        "log_midpoint_bin_edges_um": log_midpoint_edges(nodes_um).tolist(),
        "pooled_detected_counts": pooled_counts.tolist(),
        "pooled_detected_weights": pooled_weights.tolist(),
        "run_weight_quantiles": _weight_quantiles(run_weights),
        "outside_numerical_support": {
            "below_16um_fraction": below_support,
            "above_90um_fraction": above_support,
            "combined_fraction": below_support + above_support,
            "handling": "clipped_to_the_nearest_endpoint_node_for_sensitivity",
        },
        "source_provenance": {
            "prt_directory": str(PRT_DIR.resolve()),
            "standard_prt_file_set_sha256": target["source_artifacts"][
                "standard_prt_file_set_sha256"
            ],
            "dpv_target_contract": str(TARGET_CONTRACT.resolve()),
            "dpv_target_contract_sha256": _sha256(TARGET_CONTRACT),
        },
        "paper_use_allowed": False,
        "next_gate": (
            "apply these detected-particle weights to each centerline-aperture "
            "sensitivity, then freeze calibration/heldout run splits"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    contract = build_detected_diameter_weight_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)
    print(f"Wrote: {args.output}")
    print(
        "Detected weights: "
        + ", ".join(
            f"{value:.4f}" for value in contract["pooled_detected_weights"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
