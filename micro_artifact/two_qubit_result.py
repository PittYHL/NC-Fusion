"""Two-qubit NC-Fusion comparison for the Table 4 benchmark set.

The NCF-two generation path follows ``2025summer/same_error.py``.  Its
threshold is per generated NCF-two rotation, so the compressor is called with
``normalized=0``.  GridSynth, Rustiq, and NCF-one QASM are reused whenever
available; missing inputs are generated and saved under the artifact circuit
directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from ncfusion.legacy import run_benchmark
from ncfusion.metrics import (
    merge_records,
    metrics_from_qasm_file,
    read_records_csv,
    write_json,
    write_records_csv,
)
from ncfusion.spec import find_benchmark, find_experiment

from .common import add_cli_arguments
from .data import existing_qasm_path


STATUS = "available"
DEFAULT_OUTPUT = Path("micro_artifact/results/runs/two_qubit_result")
BENCHMARKS = tuple(find_experiment("table4").benchmarks)
METHODS = ("gridsyn", "rustiq", "ncf-one", "ncf-two")
METRICS = ("t_count", "t_depth", "clifford_count")
BENCHMARK_ALIASES = {
    "is-2d-30": "Ising-2D-30",
    "is-2d-60": "Ising-2D-60",
    "is-3d-30": "Ising-3D-30",
    "is-3d-60": "Ising-3D-60",
    "hei-2d-30": "Heisenberg-2D-30",
    "hei-2d-60": "Heisenberg-2D-60",
    "hei-3d-30": "Heisenberg-3D-30",
    "hei-3d-60": "Heisenberg-3D-60",
}


def _selected_benchmarks(requested: list[str] | None) -> list[str]:
    """Canonicalize and validate repeated ``--benchmark`` selections."""

    if requested is None:
        return list(BENCHMARKS)
    canonical_names = {name.lower(): name for name in BENCHMARKS}
    selected: list[str] = []
    for value in requested:
        key = value.strip().lower()
        name = BENCHMARK_ALIASES.get(key, canonical_names.get(key))
        if name is None:
            raise ValueError(
                f"unknown two-qubit benchmark {value!r}; choose from "
                + ", ".join(BENCHMARKS)
            )
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("at least one two-qubit benchmark must be selected")
    return selected


def _threshold(benchmark: Any) -> float:
    return 0.11 if benchmark.family == "Ising" else 0.12


def _prefix(method: str) -> str:
    return method.replace("-", "_")


def _load_qasm(benchmark: Any, method: str) -> dict[str, object] | None:
    path = existing_qasm_path(benchmark.name, method, prefer_two_qubit=True)
    if path is None:
        return None
    record: dict[str, object] = {
        **metrics_from_qasm_file(path).as_dict(),
        "benchmark": benchmark.name,
        "method": method,
        "qasm_path": str(path),
        "data_source": "existing_qasm",
    }
    return record


def _generate(
    benchmark: Any,
    method: str,
    *,
    gpu: int,
    threshold: float,
    single_error: float | None,
) -> dict[str, object]:
    if method == "ncf-two":
        settings: dict[str, object] = {
            "two_window": 128,
            "two_qubit_error": threshold,
            "ncf_two_normalized": 0,
            "synthesis_error": threshold,
            "trotter_steps": 1,
            "evolution_time": 1.0,
            "t_budget": 60,
        }
    else:
        error = 0.001 if single_error is None else single_error
        settings = {
            "single_window": 4,
            "synthesis_error": error,
            "trotter_steps": 1,
            "evolution_time": 1.0,
            "t_budget": 60,
        }
    return run_benchmark(
        benchmark,
        method=method,
        settings=settings,
        gpu=gpu,
        save_qasm=True,
    )


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reduction(reference: object, candidate: object) -> float | None:
    reference_value = _number(reference)
    candidate_value = _number(candidate)
    if reference_value in (None, 0.0) or candidate_value is None:
        return None
    return 100.0 * (reference_value - candidate_value) / reference_value


def _wide_record(
    benchmark: Any,
    threshold: float,
    method_records: dict[str, dict[str, object] | None],
    errors: dict[str, str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark": benchmark.name,
        "ncf_two_threshold_per_rotation": threshold,
        "source_script": "2025summer/same_error.py",
    }
    for method in METHODS:
        prefix = _prefix(method)
        record = method_records.get(method)
        row[f"{prefix}_qasm_path"] = record.get("qasm_path", "") if record else ""
        row[f"{prefix}_data_source"] = record.get("data_source", "") if record else ""
        row[f"{prefix}_compilation_time_seconds"] = (
            record.get("compilation_time_seconds", record.get("runtime_seconds", ""))
            if record else ""
        )
        if record is None and method in errors:
            row[f"{prefix}_error"] = errors[method]
        else:
            row[f"{prefix}_error"] = ""
        for metric in METRICS:
            row[f"{prefix}_{metric}"] = record.get(metric, "") if record else ""

    ncf_two = method_records.get("ncf-two")
    for method in ("gridsyn", "rustiq", "ncf-one"):
        other = method_records.get(method)
        other_prefix = _prefix(method)
        for metric in METRICS:
            row[f"ncf_two_vs_{other_prefix}_{metric}_reduction_percent"] = _reduction(
                other.get(metric) if other else None,
                ncf_two.get(metric) if ncf_two else None,
            )
    return row


def _average_record(records: list[dict[str, object]]) -> dict[str, object]:
    average: dict[str, object] = {
        "benchmark": f"AVERAGE_{len(records)}",
        "record_count": len(records),
        "source_script": "aggregate",
    }
    for method in METHODS:
        prefix = _prefix(method)
        for metric in METRICS:
            values = [_number(row.get(f"{prefix}_{metric}")) for row in records]
            values = [value for value in values if value is not None]
            average[f"{prefix}_{metric}"] = sum(values) / len(values) if values else None
        runtimes = [_number(row.get(f"{prefix}_compilation_time_seconds")) for row in records]
        runtimes = [value for value in runtimes if value is not None]
        average[f"{prefix}_compilation_time_seconds"] = sum(runtimes) / len(runtimes) if runtimes else None
    for method in ("gridsyn", "rustiq", "ncf-one"):
        prefix = _prefix(method)
        for metric in METRICS:
            values = [_number(row.get(f"ncf_two_vs_{prefix}_{metric}_reduction_percent")) for row in records]
            values = [value for value in values if value is not None]
            average[f"ncf_two_vs_{prefix}_{metric}_reduction_percent"] = (
                sum(values) / len(values) if values else None
            )
    return average


def run(
    output: Path | str = DEFAULT_OUTPUT,
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    source: str = "existing",
) -> dict[str, Any]:
    """Write the 11-benchmark comparison without running unless invoked."""

    if source not in {"auto", "existing", "generate"}:
        raise ValueError("source must be auto, existing, or generate")
    chosen_methods = list(methods or METHODS)
    if set(chosen_methods) != set(METHODS):
        raise ValueError("two_qubit_result records gridsyn, rustiq, ncf-one, and ncf-two together")
    selected = _selected_benchmarks(benchmarks)
    selected_specs = [find_benchmark(name) for name in selected]
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    generation_errors: dict[str, dict[str, str]] = {}

    for benchmark in selected_specs:
        threshold = _threshold(benchmark)
        method_records: dict[str, dict[str, object] | None] = {}
        errors: dict[str, str] = {}

        # Prefer all stored QASM, including a previously generated NCF-two
        # circuit. ``source=generate`` is the only mode that forces a fresh
        # generation of an available input.
        for method in METHODS:
            method_records[method] = None if source == "generate" else _load_qasm(benchmark, method)

        ncf_two = method_records["ncf-two"]
        if ncf_two is None:
            try:
                ncf_two = _generate(
                    benchmark,
                    "ncf-two",
                    gpu=gpu,
                    threshold=threshold,
                    single_error=None,
                )
                method_records["ncf-two"] = ncf_two
            except Exception as error:
                errors["ncf-two"] = f"{type(error).__name__}: {error}"

        # same_error.py derives the single-qubit/grid/rustiq error from the
        # total NCF-two rotation error when a fallback generation is needed.
        rotations = None
        if ncf_two is not None:
            unitary_count = _number(ncf_two.get("ncf_unitaries_generated"))
            rz_count = _number(ncf_two.get("ncf_rz_generated"))
            if unitary_count is not None and rz_count is not None:
                rotations = unitary_count + rz_count
        same_error = threshold * rotations / benchmark.pauli_terms if rotations else None

        for method in ("gridsyn", "rustiq", "ncf-one"):
            if method_records[method] is not None:
                continue
            if source == "existing":
                errors[method] = "required QASM is missing and source=existing forbids generation"
                continue
            try:
                method_records[method] = _generate(
                    benchmark,
                    method,
                    gpu=gpu,
                    threshold=threshold,
                    single_error=same_error,
                )
            except Exception as error:
                errors[method] = f"{type(error).__name__}: {error}"

        generation_errors[benchmark.name] = errors
        rows.append(_wide_record(benchmark, threshold, method_records, errors))
        checkpoint_existing = [
            row
            for row in read_records_csv(output_path / "metrics.csv")
            if not str(row.get("benchmark", "")).startswith("AVERAGE_")
        ]
        checkpoint_rows = merge_records(
            checkpoint_existing,
            rows,
            ("benchmark",),
        )
        write_records_csv(
            output_path / "metrics.csv",
            [*checkpoint_rows, _average_record(checkpoint_rows)],
        )

    existing_rows = read_records_csv(output_path / "metrics.csv")
    existing_benchmark_rows = [
        row
        for row in existing_rows
        if not str(row.get("benchmark", "")).startswith("AVERAGE_")
    ]
    rows = merge_records(existing_benchmark_rows, rows, ("benchmark",))
    rows.append(_average_record(rows))
    manifest = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "two_qubit_result",
        "paper_section": "5.3",
        "source_script": "2025summer/same_error.py",
        "source_mode": source,
        "benchmarks": [spec.name for spec in selected_specs],
        "methods": METHODS,
        "threshold_policy": "0.11 per NCF-two rotation for Ising benchmarks; 0.12 per NCF-two rotation otherwise",
        "ncf_two_normalized": 0,
        "reuse_policy": "read existing QASM by default; --source generate forces regeneration",
        "csv_merge_policy": "replace matching benchmark rows and append new benchmarks; rebuild AVERAGE rows",
        "qasm_saved_on_generation": True,
        "record_count": len(rows),
        "benchmark_record_count": len(rows) - 1,
        "average_reduction_definition": "100 * (other method metric - NCF-two metric) / other method metric",
        "generation_errors": generation_errors,
        "gpu": gpu,
        "seed": seed,
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", rows)
    return {"manifest": manifest, "records": rows}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the two-qubit NC-Fusion comparison")
    add_cli_arguments(parser)
    parser.set_defaults(output=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        choices=("existing", "generate", "auto"),
        default="existing",
        help="read stored QASM by default; use generate to regenerate all inputs",
    )
    args = parser.parse_args()
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            source=args.source,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    _main()
