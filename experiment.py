"""Command line entry point for the prefixprobe experiment."""

import argparse
from pathlib import Path

from prefixprobe import DATA_DIR, run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate a model's Multi-step SWE step and solve rates from budgeted task prefixes."
    )
    parser.add_argument("--out", type=Path, required=True, help="path of the metrics JSON file to write")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="directory holding benchmark.js and tasks/")
    parser.add_argument("--replicates", type=int, default=run.REPLICATES, help="number of task bootstrap replicates")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run.main(args.data, args.out, args.replicates)


if __name__ == "__main__":
    main()
