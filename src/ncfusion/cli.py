"""Command-line interface for the NC-Fusion reproducibility project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .runner import run_paper, run_smoke, validate_table4
from .spec import benchmarks, experiments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate NC-Fusion paper experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list configured benchmarks and experiments")

    smoke = subparsers.add_parser("smoke", help="run the dependency-free artifact validation")
    smoke.add_argument("--output", type=Path, default=Path("micro_artifact/results/smoke"))

    validate = subparsers.add_parser("validate", help="compare a generated Table 4 CSV with the reference")
    validate.add_argument("actual", type=Path)
    validate.add_argument("--reference", type=Path, default=Path("micro_artifact/results/reference/table4.csv"))
    validate.add_argument("--tolerance", type=int, default=0)

    run = subparsers.add_parser("run", help="run a configured paper experiment")
    run.add_argument("experiment", help="configured experiment name, e.g. table4")
    run.add_argument("--benchmark", action="append", dest="benchmarks", help="restrict to one or more benchmarks")
    run.add_argument("--method", action="append", dest="methods", help="restrict to one or more methods")
    run.add_argument("--output", type=Path, default=Path("micro_artifact/results/run"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print("Benchmarks:")
        for item in benchmarks():
            print(f"  {item.name:18} {item.family:10} {item.structure:5} {item.qubits:3} qubits {item.pauli_terms:6} Pauli terms")
        print("\nExperiments:")
        for item in experiments():
            print(f"  {item.name:18} Section {item.section}: {item.description}")
        return

    try:
        if args.command == "smoke":
            result = run_smoke(args.output)
            print(f"Smoke test: {result['status']}; wrote {args.output / 'smoke.json'}")
            return
        if args.command == "validate":
            result = validate_table4(args.actual, args.reference, args.tolerance)
            print(f"Table 4 validation: {result['status']}; checked {result['checked_values']} values")
            if result["mismatches"]:
                print(f"Mismatches: {len(result['mismatches'])}")
            return
        result = run_paper(args.experiment, args.output, args.benchmarks, args.seed, args.gpu, args.methods)
        print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
