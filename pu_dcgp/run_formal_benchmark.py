
import argparse
from pathlib import Path

from .benchmark_checkpoint import (
    benchmark_run_signature,
    run_checkpointed_benchmark,
)
from .benchmark_contract import pu_dcgp_benchmark_contract
from .benchmark_reporting import write_benchmark_summary
from .benchmark_settings import benchmark_formal_config, formal_benchmark_plan


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "data" / "formal_benchmark_records.jsonl",
    )
    parser.add_argument("--replicate-start", type=int)
    parser.add_argument("--replicate-stop", type=int)
    parser.add_argument("--sample-size", type=int, action="append")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract = pu_dcgp_benchmark_contract()
    config = benchmark_formal_config(contract)
    plan = formal_benchmark_plan(contract, config)
    start = (
        args.replicate_start
        if args.replicate_start is not None
        else plan.replicate_indices[0]
    )
    stop = (
        args.replicate_stop
        if args.replicate_stop is not None
        else plan.replicate_indices[-1] + 1
    )
    replicate_indices = tuple(range(start, stop))
    sample_sizes = tuple(args.sample_size) if args.sample_size else plan.sample_sizes
    scenario_ids = tuple(args.scenario) if args.scenario else plan.scenario_ids
    selected_count = len(replicate_indices) * len(sample_sizes) * len(scenario_ids)
    signature = benchmark_run_signature(contract, config)
    print(
        f"signature={signature} datasets={selected_count} "
        f"checkpoint={args.checkpoint}"
    )
    if args.dry_run:
        return 0

    def report_progress(key: tuple[str, int, int], completed_count: int) -> None:
        print(
            f"completed={completed_count} scenario={key[0]} "
            f"n={key[1]} replicate={key[2]}",
            flush=True,
        )

    records = run_checkpointed_benchmark(
        args.checkpoint,
        contract,
        config,
        sample_sizes,
        replicate_indices,
        scenario_ids,
        progress_callback=report_progress,
    )
    summary_path = args.checkpoint.with_suffix(".summary.json")
    write_benchmark_summary(summary_path, records, contract, signature)
    print(f"records={len(records)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
