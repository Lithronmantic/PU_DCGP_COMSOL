
from dataclasses import asdict
import json
from pathlib import Path

from .benchmark_checkpoint import completed_dataset_keys
from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_hypotheses import evaluate_benchmark_hypotheses
from .benchmark_runner import (
    BenchmarkReplicateRecord,
    aggregate_benchmark_records,
)
from .benchmark_settings import benchmark_formal_config, formal_benchmark_plan
from .benchmark_completion_audit import audit_formal_checkpoint_records


def write_benchmark_summary(
    path: Path,
    records: tuple[BenchmarkReplicateRecord, ...],
    contract: SyntheticBenchmarkContract,
    run_signature: str,
) -> None:

    aggregates = aggregate_benchmark_records(records)
    hypotheses = evaluate_benchmark_hypotheses(aggregates, contract)
    plan = formal_benchmark_plan(contract, benchmark_formal_config(contract))
    completed = completed_dataset_keys(records, contract.methods)
    completion_audit = audit_formal_checkpoint_records(
        records,
        contract,
        plan,
    )
    path.write_text(
        json.dumps(
            {
                "run_signature": run_signature,
                "record_count": len(records),
                "completed_dataset_count": len(completed),
                "expected_dataset_count": plan.dataset_count,
                "formal_complete": completion_audit.formal_complete,
                "completion_audit": asdict(completion_audit),
                "aggregates": [asdict(record) for record in aggregates],
                "hypotheses": [asdict(decision) for decision in hypotheses],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
