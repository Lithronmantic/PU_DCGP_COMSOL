"""Correct A-group mapping to design row == execution order == file number."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE / "experiments" / "pu_dcgp" / "data" / "run_manifest.json"
OUTPUT = Path(__file__).with_name("data") / "a_group_manifest_mapping_correction.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _particle_file(sequence: int, data_root: Path) -> str:
    padded = f"stips-0709/{sequence:03d}.csv"
    unpadded = f"stips-0709/{sequence}.csv"
    if (data_root / padded).is_file():
        return padded
    if (data_root / unpadded).is_file():
        return unpadded
    raise FileNotFoundError(f"No A-group DPV file for sequence {sequence}")


def correct_manifest() -> dict[str, object]:
    """Apply the user-confirmed one-to-one A-group execution mapping."""

    old_bytes = MANIFEST.read_bytes()
    manifest = json.loads(old_bytes.decode("utf-8"))
    data_root = (MANIFEST.parent / manifest["data_root"]).resolve()
    a_runs = sorted(
        (run for run in manifest["runs"] if run["group"] == "A"),
        key=lambda run: int(run["design_sequence"]),
    )
    b_runs = [run for run in manifest["runs"] if run["group"] != "A"]
    if [int(run["design_sequence"]) for run in a_runs] != list(range(1, 151)):
        raise ValueError("A-group design sequence must be exactly 1 through 150")

    changed_run_ids: list[str] = []
    for sequence, run in enumerate(a_runs, start=1):
        expected_csv = _particle_file(sequence, data_root)
        expected_process = f"0709/{sequence:03d}.prt"
        if not (data_root / expected_process).is_file():
            raise FileNotFoundError(expected_process)
        before = (
            int(run["within_group_order"]),
            int(run["execution_order"]),
            run["dpv_csv"],
            run["process_export"],
        )
        after = (sequence, sequence, expected_csv, expected_process)
        if before != after:
            changed_run_ids.append(run["run_id"])
        run["within_group_order"] = sequence
        run["execution_order"] = sequence
        run["dpv_csv"] = expected_csv
        run["process_export"] = expected_process
        run["spray_distance_changed"] = float(run["spray_distance_mm"]) != 100.0

    manifest["runs"] = [*a_runs, *b_runs]
    experiment = manifest["experiment"]
    experiment["a_group_execution_order_rule"] = (
        "design_sequence_equals_execution_order_equals_file_number"
    )
    new_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    MANIFEST.write_bytes(new_bytes)

    record: dict[str, object] = {
        "schema": "pu_dcgp_v26_a_group_manifest_mapping_correction_v1",
        "status": "corrected_to_user_confirmed_one_to_one_mapping",
        "old_manifest_sha256": _sha256_bytes(old_bytes),
        "new_manifest_sha256": _sha256_bytes(new_bytes),
        "a_group_run_count": len(a_runs),
        "changed_mapping_count": len(changed_run_ids),
        "changed_run_ids": changed_run_ids,
        "mapping_rule": (
            "A design row i maps to execution order i, DPV CSV i, and process "
            "export i; spray distance does not reorder the campaign."
        ),
        "b_group_modified": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


if __name__ == "__main__":
    print(json.dumps(correct_manifest(), ensure_ascii=False, indent=2))
