"""NC-Fusion synthesis-precision ablation from stored QASM.

Section 5.6.2 compares proportional precision scaling with the fixed
per-unitary precision used by the ``ncf_fix_error`` circuits.  Both circuit
variants and the GridSynth reference are already stored under
``micro_artifact/circuits/single-qubit``; this evaluator only reads those
files and never reruns synthesis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import warnings
from typing import Any

from ncfusion.metrics import merge_records, read_records_csv, write_json, write_records_csv
from ncfusion.runner import dependency_status

from .common import add_cli_arguments
from .data import PRIMARY_CIRCUIT_ROOT, benchmark_directory, existing_qasm_path
from ncfusion.spec import find_benchmark


STATUS = "available"
DEFAULT_BENCHMARKS = (
    "LiH",
    "H2O",
    "N2",
    "Ising-2D-30",
    "Ising-2D-60",
    "Ising-3D-30",
    "Ising-3D-60",
    "Heisenberg-2D-30",
    "Heisenberg-2D-60",
    "Heisenberg-3D-30",
    "Heisenberg-3D-60",
)
METRICS = ("t_count", "t_depth", "clifford_count")
METRIC_LABELS = {
    "t_count": "T-count",
    "t_depth": "T-depth",
    "clifford_count": "Clifford count",
}
PRECISION_VARIANTS = (
    {
        "same_error": 0,
        "name": "scaled_total_error",
        "qasm_suffix": "ncf_c+t",
        "t_budget": 60,
    },
    {
        "same_error": 1,
        "name": "fixed_per_unitary_error",
        "qasm_suffix": "ncf_fix_error_c+t",
        "t_budget": 40,
    },
)


def _qasm_metrics(path: Path) -> dict[str, int]:
    """Read QASM metrics using the paper's exact Qiskit T-depth definition."""

    from qiskit import QuantumCircuit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        circuit = QuantumCircuit.from_qasm_file(str(path))
    operations = list(circuit.data)
    t_names = {"t", "tdg"}
    t_count = sum(item.operation.name.lower() in t_names for item in operations)
    t_depth = int(
        circuit.depth(lambda gate: gate[0].name == "t" or gate[0].name == "tdg")
    )
    return {
        "t_count": int(t_count),
        "t_depth": t_depth,
        "clifford_count": len(operations) - int(t_count),
        "gate_count": len(operations),
    }


def _fix_error_qasm_path(benchmark: str) -> Path:
    directory = benchmark_directory(benchmark)
    return PRIMARY_CIRCUIT_ROOT / directory / f"{directory}_ncf_fix_error_c+t.qasm"


def _reduction_percent(reference: int, candidate: int) -> float:
    if reference == 0:
        return 0.0 if candidate == 0 else float("nan")
    return 100.0 * (reference - candidate) / reference


def _without_scaling_to_scaling_percent(
    without_scaling_reduction: float,
    scaled_reduction: float,
) -> float:
    """Return the no-scaling reduction as a percentage of scaled reduction."""

    if scaled_reduction == 0:
        return 0.0 if without_scaling_reduction == 0 else float("nan")
    return 100.0 * without_scaling_reduction / scaled_reduction


def _comparison_record(benchmark: str) -> dict[str, object]:
    grid_path = existing_qasm_path(benchmark, "gridsyn")
    scaled_path = existing_qasm_path(benchmark, "ncf-one")
    fixed_path = _fix_error_qasm_path(benchmark)
    missing = [
        label
        for label, path in (
            ("GridSynth", grid_path),
            ("scaled NC-Fusion", scaled_path),
            ("fixed-error NC-Fusion", fixed_path),
        )
        if path is None or not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{benchmark} is missing required QASM: {', '.join(missing)}"
        )

    grid = _qasm_metrics(grid_path)
    scaled = _qasm_metrics(scaled_path)
    without_scaling = _qasm_metrics(fixed_path)
    record: dict[str, object] = {
        "benchmark": find_benchmark(benchmark).name,
        "method": "ncf-one",
        "budget": 1,
        "window": 4,
        "data_source": "existing_qasm",
        "gridsyn_qasm_path": str(grid_path),
        "scaled_qasm_path": str(scaled_path),
        "without_scaling_qasm_path": str(fixed_path),
    }
    for metric in METRICS:
        scaled_reduction = _reduction_percent(grid[metric], scaled[metric])
        without_scaling_reduction = _reduction_percent(
            grid[metric], without_scaling[metric]
        )
        record.update(
            {
                f"scaled_{metric}_reduction_percent": scaled_reduction,
                f"without_scaling_{metric}_reduction_percent": without_scaling_reduction,
                f"without_scaling_to_scaling_{metric}_percent": (
                    _without_scaling_to_scaling_percent(
                        without_scaling_reduction, scaled_reduction
                    )
                ),
            }
        )
    return record


def _average_record(records: list[dict[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {
        "benchmark": f"AVERAGE_{len(records)}",
        "method": "ncf-one",
        "budget": 1,
        "window": 4,
        "data_source": "aggregate",
        "benchmark_count": len(records),
    }
    fields = tuple(
        field
        for metric in METRICS
        for field in (
            f"scaled_{metric}_reduction_percent",
            f"without_scaling_{metric}_reduction_percent",
            f"without_scaling_to_scaling_{metric}_percent",
        )
    )
    for field in fields:
        values = [float(item[field]) for item in records]
        record[field] = sum(values) / len(values) if values else None
    return record


def _capped_plot_average(records: list[dict[str, object]]) -> dict[str, object]:
    average: dict[str, object] = {
        "benchmark": "Average",
    }
    for metric in METRICS:
        field = f"without_scaling_to_scaling_{metric}_percent"
        values = [min(100.0, float(record[field])) for record in records]
        average[field] = sum(values) / len(values) if values else 0.0
    return average


def _write_plot(
    output_path: Path,
    records: list[dict[str, object]],
    *,
    filename: str = "precision_ablation_reductions.png",
    title: str = "Precision ablation relative to scaled NC-Fusion",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    labels = [str(record["benchmark"]) for record in records]
    x = np.arange(len(labels))
    width = 0.38
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    for axis, metric in zip(axes, METRICS):
        scaled = [100.0 for _ in records]
        without_scaling = [
            min(100.0, float(record[f"without_scaling_to_scaling_{metric}_percent"]))
            for record in records
        ]
        axis.bar(
            x,
            scaled,
            width * 1.6,
            label="",
            color="#4C78A8",
        )
        axis.bar(
            x,
            without_scaling,
            width,
            label="NC-Fusion Merging",
            color="#F58518",
        )
        axis.axhline(100, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Percentage of scaled NC-Fusion reduction (%)")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / filename
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def run(
    output: Path | str = "micro_artifact/results/runs/precision_abalation",
    *,
    benchmarks: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    budget: int = 1,
    window: int | None = None,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    pauli_order_seed: int | None = 0,
) -> dict[str, Any]:
    """Compare stored scaled and fixed-error single-qubit QASM metrics."""

    if budget != 1:
        raise ValueError("precision_abalation uses the single-qubit QASM set (budget=1)")
    selected_benchmarks = list(DEFAULT_BENCHMARKS if benchmarks is None else benchmarks)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for benchmark in selected_benchmarks:
        records.append(_comparison_record(benchmark))
        existing_checkpoint = [
            row
            for row in read_records_csv(output_path / "metrics.csv")
            if not str(row.get("benchmark", "")).startswith("AVERAGE_")
        ]
        checkpoint_records = merge_records(
            existing_checkpoint,
            records,
            ("benchmark", "method"),
        )
        write_records_csv(
            output_path / "metrics.csv",
            [*checkpoint_records, _average_record(checkpoint_records)],
        )
    existing_records = read_records_csv(output_path / "metrics.csv")
    existing_benchmark_records = [
        row
        for row in existing_records
        if not str(row.get("benchmark", "")).startswith("AVERAGE_")
    ]
    records = merge_records(existing_benchmark_records, records, ("benchmark", "method"))
    average = _average_record(records)
    plot_average = _capped_plot_average(records)
    plot_path = _write_plot(
        output_path,
        [*records, plot_average],
    )
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "precision_abalation",
        "paper_section": "5.6.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "budget": 1,
        "window": 4,
        "variants": list(PRECISION_VARIANTS),
        "reference": "gridsyn_c+t.qasm",
        "source": "existing QASM only",
        "synthesis_run": False,
        "qasm_saved": False,
        "reduction_definition": "100 * (GridSynth metric - NC-Fusion metric) / GridSynth metric",
        "without_scaling_to_scaling_definition": "100 * without-scaling reduction / scaled reduction",
        "record_count": len(records) + 1,
        "benchmark_record_count": len(records),
        "summary_record_count": 1,
        "csv_merge_policy": "replace matching benchmark/method rows and append new benchmarks; rebuild the aggregate row",
        "plot": str(plot_path) if plot_path is not None else None,
        "paper_dependencies": dependency_status(),
        "unused_generation_parameters": {
            "gpu": gpu,
            "error_threshold": error_threshold,
            "trotter_steps": trotter_steps,
            "evolution_time": evolution_time,
            "pauli_order_seed": pauli_order_seed,
        },
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", [*records, average])
    return {"manifest": manifest, "records": [*records, average], "plot": plot_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the NC-Fusion precision ablation from stored QASM"
    )
    add_cli_arguments(parser, include_method=False)
    parser.add_argument("--budget", type=int, choices=(1,), default=1)
    parser.add_argument("--window", type=int)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--pauli-order-seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            seed=args.seed,
            gpu=args.gpu,
            budget=args.budget,
            window=args.window,
            error_threshold=args.error_threshold,
            t_budget=args.t_budget,
            trotter_steps=args.trotter_steps,
            evolution_time=args.evolution_time,
            pauli_order_seed=args.pauli_order_seed,
        )
    except (FileNotFoundError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    main()
