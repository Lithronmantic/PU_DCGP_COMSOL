"""H11: empirical A-group PRT targets for COMSOL-DPV validation.

CSV rows are instrument summary frames and must not be treated as particles.
This v2 contract reads the 150 standard PRT exports instead.  Each contains a
configurable retained sample of 1000 individually detected particles with
joint temperature, velocity, diameter, position, and signal fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulator_v2.phase_h.h11_dpv_measurement_level_audit import (
    DpvMeasurementAuditContract,
    read_design,
    spearman,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA_ROOT = WORKSPACE / "data-DPV"
PRT_DIR = DATA_ROOT / "0709"
DESIGN_WORKBOOK = (
    DATA_ROOT / "APS_YSZ_150组总试验文件_A方案_固定观测距离.xlsx"
)
MEASUREMENT_AUDIT = (
    HERE
    / "h11_outputs"
    / "dpv_measurement_level"
    / "h11_dpv_measurement_level_audit.json"
)
DEFAULT_OUTPUT = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)
QUANTILES = np.asarray([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
VALIDATION_QUANTILES = np.asarray([0.10, 0.50, 0.90])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _quantile_dict(
    values: np.ndarray,
    quantiles: np.ndarray = QUANTILES,
) -> dict[str, float]:
    estimates = np.quantile(np.asarray(values, dtype=float), quantiles)
    return {
        f"q{int(round(100 * quantile)):02d}": float(value)
        for quantile, value in zip(quantiles, estimates)
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or len(right) != len(left):
        return None
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(spearman(left.tolist(), right.tolist()))


def load_prt_particles(path: Path) -> dict[str, np.ndarray]:
    """Read the seven numeric particle fields while preserving row alignment."""

    matrix = np.loadtxt(
        path,
        skiprows=1,
        usecols=(2, 3, 4, 5, 6, 7, 8),
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[1] != 7:
        raise ValueError(f"Unexpected PRT numeric matrix: {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite PRT particle value: {path}")
    if (matrix[:, 2:5] <= 0).any():
        raise ValueError(f"Nonpositive speed/temperature/diameter: {path}")
    return {
        "x_position": matrix[:, 0],
        "y_position": matrix[:, 1],
        "velocity_m_s": matrix[:, 2],
        "temperature_c": matrix[:, 3],
        "particle_diameter_um": matrix[:, 4],
        "energy_a": matrix[:, 5],
        "energy_b": matrix[:, 6],
    }


def _file_set_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest().upper()


def build_target_contract() -> dict[str, Any]:
    measurement = json.loads(MEASUREMENT_AUDIT.read_text(encoding="utf-8"))
    if measurement["status"] != "pass_A_group_measurement_level_audit":
        raise ValueError("The DPV measurement-level audit has not passed")
    design_contract = DpvMeasurementAuditContract()
    design_contract.validate()
    design = read_design(DESIGN_WORKBOOK, design_contract)
    if len(design) != 150:
        raise ValueError("The A-group design must contain 150 runs")

    paths = [PRT_DIR / f"{run:03d}.prt" for run in range(1, 151)]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("One or more standard A-group PRT files are missing")
    particle_runs = [load_prt_particles(path) for path in paths]
    counts = np.asarray(
        [len(run["temperature_c"]) for run in particle_runs],
        dtype=int,
    )
    if not np.all(counts == 1000):
        raise ValueError("Every standard A-group PRT export must contain 1000 rows")
    outcome_names = (
        "temperature_c",
        "velocity_m_s",
        "particle_diameter_um",
    )
    if not all(
        len(run[outcome]) == counts[index]
        for index, run in enumerate(particle_runs)
        for outcome in outcome_names
    ):
        raise ValueError("Joint PRT particle alignment was lost")

    outcomes: dict[str, Any] = {}
    for outcome in outcome_names:
        samples = [run[outcome] for run in particle_runs]
        pooled = np.concatenate(samples)
        run_quantiles = np.asarray(
            [
                np.quantile(values, VALIDATION_QUANTILES)
                for values in samples
            ],
            dtype=float,
        )
        run_means = np.asarray(
            [np.mean(values) for values in samples],
            dtype=float,
        )
        outcomes[outcome] = {
            "pooled_detected_particle_sample_count": int(len(pooled)),
            "pooled_min": float(np.min(pooled)),
            "pooled_max": float(np.max(pooled)),
            "pooled_quantiles": _quantile_dict(pooled),
            "run_mean_quantiles": _quantile_dict(run_means),
            "run_level_validation_quantiles": {
                "q10": _quantile_dict(run_quantiles[:, 0]),
                "q50": _quantile_dict(run_quantiles[:, 1]),
                "q90": _quantile_dict(run_quantiles[:, 2]),
            },
        }

    pair_names = (
        ("temperature_c", "velocity_m_s"),
        ("temperature_c", "particle_diameter_um"),
        ("velocity_m_s", "particle_diameter_um"),
    )
    within_run_rank_associations: dict[str, Any] = {}
    for left_name, right_name in pair_names:
        correlations = [
            _safe_spearman(run[left_name], run[right_name])
            for run in particle_runs
        ]
        finite = np.asarray(
            [value for value in correlations if value is not None],
            dtype=float,
        )
        within_run_rank_associations[f"{left_name}__{right_name}"] = {
            "n_runs": int(len(finite)),
            "quantiles": _quantile_dict(finite),
            "fraction_positive": float(np.mean(finite > 0)),
        }

    detection_fields: dict[str, Any] = {}
    for name in ("x_position", "y_position", "energy_a", "energy_b"):
        samples = [run[name] for run in particle_runs]
        pooled = np.concatenate(samples)
        detection_fields[name] = {
            "pooled_quantiles": _quantile_dict(pooled),
            "run_mean_quantiles": _quantile_dict(
                np.asarray([np.mean(values) for values in samples])
            ),
        }

    treatment_values = np.asarray(
        [
            (
                row["current_a"],
                row["argon_scfh"],
                row["powder_feed_g_min"],
                row["spray_distance_mm"],
            )
            for row in design
        ],
        dtype=float,
    )
    unique_settings = np.unique(treatment_values, axis=0)
    return {
        "schema_version": "h11_dpv_target_contract_v2_prt_particles",
        "status": "pass_prt_particle_target_contract",
        "n_runs": len(particle_runs),
        "n_unique_four_factor_settings": int(len(unique_settings)),
        "joint_valid_particle_count": int(np.sum(counts)),
        "particle_count_interpretation": (
            "150 x 1000 retained individually detected particles. Run 063 "
            "also has 10 and 10000 row exports, proving the export count is "
            "configurable and is not a total particle count or count rate."
        ),
        "particles_per_run": {
            "minimum": int(np.min(counts)),
            "median": float(np.median(counts)),
            "maximum": int(np.max(counts)),
            "quantiles": _quantile_dict(counts.astype(float)),
        },
        "outcomes": outcomes,
        "detection_fields": detection_fields,
        "within_run_rank_associations": within_run_rank_associations,
        "estimand_contract": {
            "temperature_c": (
                "PRT-reported temperature distribution for the retained "
                "sample of individually detected particles at the fixed "
                "gun-relative 100 mm plane; exact radiometric definition "
                "still requires the instrument manual."
            ),
            "velocity_m_s": (
                "PRT-reported velocity distribution for the same row-aligned "
                "retained detected particles."
            ),
            "particle_diameter_um": (
                "PRT-reported detected-particle diameter distribution; a "
                "validation target only and never a substitute for the feedstock PSD."
            ),
            "x_y_and_energy": (
                "Detection/selection covariates for learning the observation "
                "operator, not primary causal outcomes."
            ),
        },
        "required_comsol_outputs_per_run": [
            "crossing_particle_temperature_q10_q50_q90",
            "crossing_particle_velocity_q10_q50_q90",
            "crossing_particle_diameter_q10_q50_q90",
            "crossing_position_distribution",
            "detection_or_retention_probability",
            "detected_particle_temperature_velocity_diameter_joint_sample",
        ],
        "validation_rule": (
            "Fit effective exit, injection, discrepancy, and detection "
            "parameters on training settings only. Validate complete held-out "
            "four-factor settings using quantile loss, one-dimensional "
            "Wasserstein distances, and joint rank-association errors."
        ),
        "source_artifacts": {
            "measurement_level_audit": str(MEASUREMENT_AUDIT.resolve()),
            "measurement_level_audit_sha256": _sha256(MEASUREMENT_AUDIT),
            "design_workbook": str(DESIGN_WORKBOOK.resolve()),
            "design_workbook_sha256": _sha256(DESIGN_WORKBOOK),
            "prt_directory": str(PRT_DIR.resolve()),
            "standard_prt_file_set_sha256": _file_set_hash(paths),
        },
        "csv_frames_used_as_particles": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    contract = build_target_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)
    print(f"Wrote: {args.output}")
    print(
        f"A runs={contract['n_runs']}, "
        f"settings={contract['n_unique_four_factor_settings']}, "
        f"retained PRT particles={contract['joint_valid_particle_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
