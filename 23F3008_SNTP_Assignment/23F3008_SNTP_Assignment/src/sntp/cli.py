"""
CLI entry point for SNTP Assignment 2.

Usage:
    python -m sntp.cli run --config configs/fast_demo.yaml
    python -m sntp.cli run --config configs/default.yaml
    python -m sntp.cli inspect --data-dir Assignment
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_run(args):
    """Run experiments from config file."""
    from sntp.experiments import run_all

    config = load_config(args.config)
    # Override data dir if provided
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.results_dir:
        config["results_dir"] = args.results_dir

    run_all(config)


def cmd_inspect(args):
    """Inspect dataset and print statistics."""
    from sntp.data import load_dataset

    dataset = load_dataset(args.data_dir)
    summary = dataset.summary()
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="SNTP Assignment 2: Recommendation & Information Diffusion",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--data-dir", help="Override data directory")
    run_parser.add_argument("--results-dir", help="Override results directory")
    run_parser.set_defaults(func=cmd_run)

    # 'inspect' command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect dataset")
    inspect_parser.add_argument("--data-dir", required=True, help="Path to data directory")
    inspect_parser.set_defaults(func=cmd_inspect)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
