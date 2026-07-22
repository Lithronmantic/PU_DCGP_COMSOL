
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "pu_dcgp" / "data" / "run_manifest.json"
)
DEFAULT_OUTPUT = Path(__file__).with_name("data")
REQUIRED_COLUMNS = ("Temperature", "Speed", "Diameter")
MINIMUM_PARTICLES = 20
LOW_VALID_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class BGroupQCAudit:
    raw_run_count: int
    factorial_cell_count: int
    cell_replicates: tuple[int, ...]
    raw_particle_row_count: int
    jointly_valid_particle_row_count: int
    invalid_particle_row_count: int
    primary_included_run_count: int
    primary_excluded_run_count: int
    below_20_particle_run_count: int
    low_valid_fraction_run_count: int
    duplicate_file_hash_group_count: int
    valid_particle_count_minimum: int
    valid_particle_count_median: float
    valid_particle_count_maximum: int
    outcome_blind: bool
    raw_files_modified: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def audit_b_group_data(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[BGroupQCAudit, list[dict[str, Any]]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = (manifest_path.parent / manifest["data_root"]).resolve()
    runs = [run for run in manifest["runs"] if run["group"] == "B"]
    if len(runs) != 30:
        raise AssertionError("The frozen B-group dataset contains 30 runs")

    run_ids = [run["run_id"] for run in runs]
    orders = [int(run["execution_order"]) for run in runs]
    references = [run["dpv_csv"] for run in runs]
    duplicate_ids = {value for value in run_ids if run_ids.count(value) > 1}
    duplicate_orders = {value for value in orders if orders.count(value) > 1}
    duplicate_refs = {value for value in references if references.count(value) > 1}

    hashes: dict[str, list[int]] = {}
    flags: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        path = data_root / run["dpv_csv"]
        reasons: list[str] = []
        warnings: list[str] = []
        raw_rows = 0
        valid_rows = 0
        file_hash = ""
        if not path.is_file():
            reasons.append("missing_dpv_file")
        else:
            file_hash = _sha256(path)
            hashes.setdefault(file_hash, []).append(index)
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if not set(REQUIRED_COLUMNS).issubset(set(reader.fieldnames or ())):
                        reasons.append("missing_required_dpv_column")
                    else:
                        for row in reader:
                            raw_rows += 1
                            if all(_finite_positive(row[column]) for column in REQUIRED_COLUMNS):
                                valid_rows += 1
            except (OSError, csv.Error, UnicodeError):
                reasons.append("unparseable_dpv_file")
        if valid_rows == 0 and not reasons:
            reasons.append("no_jointly_valid_particle")
        if run["run_id"] in duplicate_ids:
            reasons.append("duplicate_run_id")
        if int(run["execution_order"]) in duplicate_orders:
            reasons.append("duplicate_execution_order")
        if run["dpv_csv"] in duplicate_refs:
            reasons.append("duplicate_dpv_file_reference")
        valid_fraction = valid_rows / raw_rows if raw_rows else 0.0
        if 0 < valid_rows < MINIMUM_PARTICLES:
            warnings.append("particle_count_below_20")
        if raw_rows and valid_fraction < LOW_VALID_FRACTION:
            warnings.append("joint_valid_fraction_below_0_90")
        flags.append(
            {
                "run_id": run["run_id"],
                "execution_order": int(run["execution_order"]),
                "nominal_spray_distance_mm": float(run["spray_distance_mm"]),
                "measurement_position_mm": float(run["measurement_position_mm"]),
                "dpv_csv": run["dpv_csv"],
                "process_export": run["process_export"],
                "file_sha256": file_hash,
                "raw_particle_rows": raw_rows,
                "jointly_valid_particle_rows": valid_rows,
                "joint_valid_fraction": valid_fraction,
                "primary_include": not reasons,
                "primary_exclusion_reasons": ";".join(reasons),
                "qc_warnings": ";".join(warnings),
                "count_sensitivity_include": not reasons and valid_rows >= MINIMUM_PARTICLES,
            }
        )

    duplicate_hash_groups = [indices for indices in hashes.values() if len(indices) > 1]
    for indices in duplicate_hash_groups:
        for index in indices:
            flags[index]["qc_warnings"] = ";".join(
                item
                for item in (
                    flags[index]["qc_warnings"],
                    "duplicate_file_content_hash",
                )
                if item
            )

    cells: dict[tuple[float, float], int] = {}
    for run in runs:
        key = (float(run["spray_distance_mm"]), float(run["measurement_position_mm"]))
        cells[key] = cells.get(key, 0) + 1
    counts = sorted(row["jointly_valid_particle_rows"] for row in flags)
    median = float((counts[14] + counts[15]) / 2)
    raw_count = sum(row["raw_particle_rows"] for row in flags)
    valid_count = sum(row["jointly_valid_particle_rows"] for row in flags)
    audit = BGroupQCAudit(
        raw_run_count=len(runs),
        factorial_cell_count=len(cells),
        cell_replicates=tuple(sorted(cells.values())),
        raw_particle_row_count=raw_count,
        jointly_valid_particle_row_count=valid_count,
        invalid_particle_row_count=raw_count - valid_count,
        primary_included_run_count=sum(row["primary_include"] for row in flags),
        primary_excluded_run_count=sum(not row["primary_include"] for row in flags),
        below_20_particle_run_count=sum(
            row["jointly_valid_particle_rows"] < MINIMUM_PARTICLES
            for row in flags
            if row["primary_include"]
        ),
        low_valid_fraction_run_count=sum(
            row["joint_valid_fraction"] < LOW_VALID_FRACTION
            for row in flags
            if row["raw_particle_rows"]
        ),
        duplicate_file_hash_group_count=len(duplicate_hash_groups),
        valid_particle_count_minimum=counts[0],
        valid_particle_count_median=median,
        valid_particle_count_maximum=counts[-1],
        outcome_blind=True,
        raw_files_modified=False,
    )
    return audit, flags


def write_b_group_qc(
    audit: BGroupQCAudit,
    flags: list[dict[str, Any]],
    output_directory: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "b_group_qc_summary.json"
    flags_path = output_directory / "b_group_qc_run_flags.csv"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "pu_dcgp_v26_b_group_outcome_blind_qc_v1",
                "audit": asdict(audit),
                "claim_boundary": (
                    "QC is independent of B trends and COMSOL agreement; all technically usable runs remain primary."
                ),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with flags_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(flags[0]))
        writer.writeheader()
        writer.writerows(flags)
    return summary_path, flags_path
