
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_runner import BenchmarkReplicateRecord, run_benchmark_replicates
from .config import PUDCGPConfig


DatasetKey = tuple[str, int, int]


def benchmark_run_signature(
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
) -> str:

    payload = json.dumps(
        {"contract": asdict(contract), "config": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def append_checkpoint_records(
    path: Path,
    run_signature: str,
    records: tuple[BenchmarkReplicateRecord, ...],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(
            {"run_signature": run_signature, "record": asdict(record)},
            sort_keys=True,
        )
        + "\n"
        for record in records
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(lines)


def load_checkpoint_records(
    path: Path,
    expected_signature: str,
) -> tuple[BenchmarkReplicateRecord, ...]:

    if not path.exists():
        return ()
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["run_signature"] != expected_signature:
            raise ValueError("Checkpoint signature does not match this benchmark run")
        record = BenchmarkReplicateRecord(**payload["record"])
        key = (
            record.scenario_id,
            record.sample_size,
            record.replicate_index,
            record.method_name,
        )
        records[key] = record
    return tuple(records[key] for key in sorted(records))


def completed_dataset_keys(
    records: tuple[BenchmarkReplicateRecord, ...],
    method_names: tuple[str, ...],
) -> frozenset[DatasetKey]:

    methods_by_dataset: dict[DatasetKey, set[str]] = {}
    for record in records:
        key = (
            record.scenario_id,
            record.sample_size,
            record.replicate_index,
        )
        methods_by_dataset.setdefault(key, set()).add(record.method_name)
    required = set(method_names)
    return frozenset(
        key for key, methods in methods_by_dataset.items() if methods == required
    )


def merge_checkpoint_shards(
    shard_paths: tuple[Path, ...],
    output_path: Path,
    expected_signature: str,
) -> tuple[BenchmarkReplicateRecord, ...]:

    combined: dict[
        tuple[str, int, int, str], BenchmarkReplicateRecord
    ] = {}
    source_paths = shard_paths + ((output_path,) if output_path.exists() else ())
    for path in source_paths:
        for record in load_checkpoint_records(path, expected_signature):
            key = (
                record.scenario_id,
                record.sample_size,
                record.replicate_index,
                record.method_name,
            )
            if key in combined and combined[key] != record:
                raise ValueError(f"Conflicting checkpoint record for {key}")
            combined[key] = record
    existing = (
        {
            (
                record.scenario_id,
                record.sample_size,
                record.replicate_index,
                record.method_name,
            )
            for record in load_checkpoint_records(output_path, expected_signature)
        }
        if output_path.exists()
        else set()
    )
    missing = tuple(
        combined[key] for key in sorted(combined) if key not in existing
    )
    if missing:
        append_checkpoint_records(output_path, expected_signature, missing)
    return load_checkpoint_records(output_path, expected_signature)


def run_checkpointed_benchmark(
    path: Path,
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
    sample_sizes: tuple[int, ...],
    replicate_indices: tuple[int, ...],
    scenario_ids: tuple[str, ...],
    progress_callback: Callable[[DatasetKey, int], None] | None = None,
) -> tuple[BenchmarkReplicateRecord, ...]:

    signature = benchmark_run_signature(contract, config)
    records = load_checkpoint_records(path, signature)
    complete = completed_dataset_keys(records, contract.methods)
    for scenario_id in scenario_ids:
        for sample_size in sample_sizes:
            for replicate_index in replicate_indices:
                key = (scenario_id, sample_size, replicate_index)
                if key in complete:
                    continue
                dataset_records = run_benchmark_replicates(
                    contract,
                    config,
                    sample_size,
                    (replicate_index,),
                    (scenario_id,),
                )
                append_checkpoint_records(path, signature, dataset_records)
                complete = complete | {key}
                if progress_callback is not None:
                    progress_callback(key, len(complete))
    return load_checkpoint_records(path, signature)
