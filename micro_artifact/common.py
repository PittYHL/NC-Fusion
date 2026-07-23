"""Shared helpers for the paper evaluation modules.

The public compiler lives in :mod:`ncfusion`.  This package is the artifact
layer: it selects a paper experiment, records the output directory, and gives
each evaluation a stable entry point without importing optional quantum
packages during discovery.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable

from ncfusion.runner import run_paper


class MissingEvaluationError(RuntimeError):
    """Raised when an evaluation needs source code that is not in the repo."""


# Status is intentionally explicit.  ``available`` means that the module is
# wired to the current configured runner.  ``partial`` means that some legacy
# helpers or configuration exist, but the paper-specific evaluator is not yet
# complete.  ``missing`` means that the implementation was not found.
EVALUATIONS: dict[str, dict[str, str]] = {
    "single_qubit_result": {
        "status": "available",
        "experiment": "table4",
        "source": "configured ncf-one runner",
    },
    "two_qubit_result": {
        "status": "available",
        "experiment": "table4",
        "source": "configured ncf-two runner; requires Synthetiq/Docker",
    },
    "analytical_estimation": {
        "status": "available",
        "experiment": "5.4 analytical T-count model",
        "source": "single/two-qubit producer CSVs plus paper scaling formulas",
    },
    "window_size_sensitivity": {
        "status": "available",
        "experiment": "sensitivity",
        "source": "configured sensitivity runner",
    },
    "pauli_string_order_sensitivity": {
        "status": "available",
        "experiment": "random-order",
        "source": "configured randomized-order runner",
    },
    "components_abalation": {
        "status": "available",
        "experiment": "5.6.1 NC-Fusion component ablation",
        "source": "main_alg.py lines 279, 282, and 287",
    },
    "precision_abalation": {
        "status": "available",
        "experiment": "5.6.2 synthesis precision ablation",
        "source": "NC_Fusion fix_error_threshold=0 versus 1",
    },
    "trotter_error": {
        "status": "available",
        "experiment": "5.7.1 Trotter operator-norm error",
        "source": "trotter_operator_norm_error(rz_qc)",
    },
    "application_level_fidelity": {
        "status": "available",
        "experiment": "5.7.2 application-level fidelity",
        "source": "density_matrix_error_from_hamiltonian(clifford_t_qc)",
    },
    "t_count_methods_comparison": {
        "status": "available",
        "experiment": "optimizer-comparison",
        "source": "tzap + PyZX + GridSynth + T-Optimizer workflow",
    },
    "space_volume_analysis": {
        "status": "available",
        "experiment": "spacetime-volume",
        "source": "stored Clifford+T QASM plus Infleqtion resource-superstaq",
    },
}


def run_configured(
    experiment: str,
    output: Path | str,
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    reuse_existing: bool = True,
    save_qasm: bool = True,
) -> dict[str, Any]:
    """Run one configured experiment and write its manifest and CSV output."""

    return run_paper(
        experiment,
        Path(output),
        requested_benchmarks=benchmarks,
        seed=seed,
        gpu=gpu,
        methods=methods,
        reuse_existing=reuse_existing,
        save_qasm=save_qasm,
    )


def missing_evaluation(name: str, details: str) -> None:
    """Raise a consistent, actionable error for an incomplete evaluator."""

    raise MissingEvaluationError(
        f"{name} is not yet reproducible from this checkout: {details}. "
        "Please upload the missing paper script or evaluation specification."
    )


def add_cli_arguments(parser: argparse.ArgumentParser, *, include_source: bool = False) -> None:
    """Add options shared by all evaluation modules."""

    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--output", type=Path, default=Path("micro_artifact/results/runs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    if include_source:
        parser.add_argument(
            "--source",
            choices=("existing", "generate"),
            default="existing",
            help="reuse stored QASM/records or regenerate them",
        )


def run_cli(
    name: str,
    run: Callable[..., dict[str, Any]],
    argv: list[str] | None = None,
    *,
    include_source: bool = False,
) -> dict[str, Any] | None:
    """Run a module's function from the command line."""

    parser = argparse.ArgumentParser(description=f"Run the {name} NC-Fusion evaluation")
    add_cli_arguments(parser, include_source=include_source)
    args = parser.parse_args(argv)
    try:
        arguments = dict(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
        )
        if include_source:
            arguments["source"] = args.source
        result = run(**arguments)
    except (KeyError, MissingEvaluationError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    manifest = result.get("manifest", {})
    print(f"Completed {manifest.get('record_count', 0)} records; wrote {args.output}")
    return result
