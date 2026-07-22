
import argparse
import json
from pathlib import Path

from .benchmark_contract import pu_dcgp_benchmark_contract
from .benchmark_hypotheses import evaluate_benchmark_hypotheses
from .benchmark_postformal import build_postformal_diagnostics
from .benchmark_postformal_report import write_postformal_paper_tables
from .benchmark_runner import BenchmarkAggregateRecord


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if not payload["formal_complete"]:
        raise ValueError("post-formal tables require a complete formal summary")

    aggregates = tuple(
        BenchmarkAggregateRecord(**record) for record in payload["aggregates"]
    )
    contract = pu_dcgp_benchmark_contract()
    decisions = evaluate_benchmark_hypotheses(aggregates, contract)
    diagnostics = build_postformal_diagnostics(
        aggregates,
        decisions,
        contract.scenarios[0].sample_sizes,
    )
    write_postformal_paper_tables(args.output, diagnostics)
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
