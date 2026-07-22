from __future__ import annotations

import argparse
import runpy
import sys


def parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    return args.module, remaining


def main() -> None:
    module, arguments = parse_args()
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
