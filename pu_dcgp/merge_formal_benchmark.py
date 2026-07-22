"""Merge independently written formal benchmark checkpoint shards."""

import argparse
from pathlib import Path

from .benchmark_checkpoint import (
    benchmark_run_signature,
    merge_checkpoint_shards,
)
from .benchmark_contract import pu_dcgp_benchmark_contract
from .benchmark_reporting import write_benchmark_summary
from .benchmark_report import write_formal_benchmark_report
from .benchmark_settings import benchmark_formal_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    contract = pu_dcgp_benchmark_contract()
    config = benchmark_formal_config(contract)
    signature = benchmark_run_signature(contract, config)
    records = merge_checkpoint_shards(
        tuple(args.shards),
        args.output,
        signature,
    )
    summary_path = args.output.with_suffix(".summary.json")
    write_benchmark_summary(summary_path, records, contract, signature)
    report_path = args.output.with_suffix(".report.md")
    write_formal_benchmark_report(report_path, records, contract)
    print(
        f"records={len(records)} output={args.output} "
        f"summary={summary_path} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
