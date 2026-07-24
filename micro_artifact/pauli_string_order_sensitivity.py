"""Pauli-string order sensitivity evaluation.

The compiler portion follows ``2025summer/random_order.py``: shuffle the
non-identity Pauli terms, group them with a single-qubit window of four,
transform the groups, reorder independent groups, and run the NC-Fusion
compressor.  This artifact wrapper adds benchmark selection, reproducible
seeds, checkpointed CSV output, and statistics over all accumulated runs.

Randomized Clifford+T QASM is deliberately not written.  GridSynth QASM is
read only as the fixed comparison baseline when it is already available.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from ncfusion.metrics import metrics_from_qasm_file, write_json, write_records_csv
from ncfusion.legacy import run_benchmark
from ncfusion.spec import find_benchmark

from .common import add_cli_arguments
from .data import existing_qasm_path


STATUS = "available"

DEFAULT_OUTPUT = Path("micro_artifact/results/runs/pauli_string_order_sensitivity")
DEFAULT_BENCHMARKS = (
    "LiH",
    "H2O",
    "Ising-2D-60",
    "Heisenberg-2D-60",
)
SOURCE_RELATIVE_PATH = Path("2025summer/random_order.py")
METRICS = ("t_count", "t_depth", "clifford_count")

_BENCHMARK_ALIASES = {
    "is-2d-60": "Ising-2D-60",
    "ising-2d-60": "Ising-2D-60",
    "hei-2d-60": "Heisenberg-2D-60",
    "heisenberg-2d-60": "Heisenberg-2D-60",
}


def _source_script() -> Path:
    """Locate the original source script without hard-coding its checkout."""

    this_file = Path(__file__).resolve()
    candidates = [
        this_file.parents[3] / SOURCE_RELATIVE_PATH,
        this_file.parents[2] / SOURCE_RELATIVE_PATH,
        Path.cwd() / SOURCE_RELATIVE_PATH,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "The original 2025summer/random_order.py was not found. "
        "Expected it alongside the NC-Fusion checkout."
    )


def _canonical_benchmarks(requested: Iterable[str] | None) -> list[str]:
    names = list(requested) if requested is not None else list(DEFAULT_BENCHMARKS)
    selected: list[str] = []
    allowed = {name.lower(): name for name in DEFAULT_BENCHMARKS}
    for name in names:
        key = name.strip().lower()
        canonical = _BENCHMARK_ALIASES.get(key, allowed.get(key))
        if canonical is None:
            raise ValueError(
                f"{name!r} is not in this experiment; choose LiH, H2O, "
                "IS-2D-60, or Hei-2D-60"
            )
        if canonical not in selected:
            selected.append(canonical)
    return selected


def _grid_metrics(spec: Any, gpu: int) -> tuple[dict[str, int], str]:
    """Load the fixed GridSynth baseline, generating it only if absent."""

    path = existing_qasm_path(spec.name, "gridsyn")
    if path is not None:
        return metrics_from_qasm_file(path).as_dict(), str(path)

    # The normal artifact data set contains these files.  This fallback keeps
    # the evaluator usable on a fresh checkout, while still never saving a
    # circuit from the random-order experiment.
    generated = run_benchmark(
        spec,
        method="gridsyn",
        settings={
            "synthesis_error": 0.001,
            "trotter_steps": 1,
            "evolution_time": 1.0,
            "t_budget": 60,
        },
        gpu=gpu,
        save_qasm=False,
    )
    return {metric: int(generated[metric]) for metric in METRICS}, "generated-in-memory"


def _reduction(grid_value: int, ncf_value: int) -> float | None:
    if grid_value == 0:
        return None
    return 100.0 * (grid_value - ncf_value) / grid_value


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_existing_rows(
    rows: Iterable[dict[str, str]],
    grid_by_benchmark: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    """Keep prior rows, including rows made by the old configured runner."""

    normalized: list[dict[str, object]] = []
    for index, old in enumerate(rows):
        benchmark = old.get("benchmark", "")
        grid = grid_by_benchmark.get(benchmark)
        if grid is None:
            stored_grid = {metric: _number(old.get(f"grid_{metric}")) for metric in METRICS}
            if all(value is not None for value in stored_grid.values()):
                grid = {metric: int(stored_grid[metric]) for metric in METRICS}
        if grid is None:
            continue
        ncf_values = {metric: _number(old.get(metric)) for metric in METRICS}
        if any(value is None for value in ncf_values.values()):
            continue
        reductions = {
            f"{metric}_reduction_percent": _number(old.get(f"{metric}_reduction_percent"))
            for metric in METRICS
        }
        if any(value is None for value in reductions.values()):
            reductions = {
                f"{metric}_reduction_percent": _reduction(grid[metric], int(ncf_values[metric]))
                for metric in METRICS
            }
        normalized.append(
            {
                "benchmark": benchmark,
                "run_id": old.get("run_id") or old.get("repetition") or f"legacy-{index}",
                "seed": old.get("seed") or old.get("pauli_order_seed") or "",
                "grid_t_count": grid["t_count"],
                "grid_t_depth": grid["t_depth"],
                "grid_clifford_count": grid["clifford_count"],
                "ncf_t_count": int(ncf_values["t_count"]),
                "ncf_t_depth": int(ncf_values["t_depth"]),
                "ncf_clifford_count": int(ncf_values["clifford_count"]),
                **reductions,
                "runtime_seconds": old.get("runtime_seconds", ""),
                "data_source": old.get("data_source", "previous-run"),
            }
        )
    return normalized


def _population_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _summary(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        benchmark = str(row.get("benchmark", ""))
        grouped.setdefault(benchmark, {})
        for metric in METRICS:
            value = _number(row.get(f"{metric}_reduction_percent"))
            if value is not None:
                grouped[benchmark].setdefault(metric, []).append(value)

    ordered = [name for name in DEFAULT_BENCHMARKS if name in grouped]
    ordered.extend(name for name in grouped if name not in ordered)
    all_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    result: list[dict[str, object]] = []
    for benchmark in ordered:
        result_row: dict[str, object] = {
            "benchmark": benchmark,
            "run_count": max((len(grouped[benchmark].get(metric, [])) for metric in METRICS), default=0),
        }
        for metric in METRICS:
            values = grouped[benchmark].get(metric, [])
            all_values[metric].extend(values)
            result_row[f"{metric}_avg_reduction_percent"] = (
                sum(values) / len(values) if values else ""
            )
            result_row[f"{metric}_min_reduction_percent"] = min(values) if values else ""
            result_row[f"{metric}_max_reduction_percent"] = max(values) if values else ""
            result_row[f"{metric}_std_reduction_percent"] = _population_std(values) if values else ""
        result.append(result_row)

    if result:
        average_row: dict[str, object] = {
            "benchmark": "Average",
            "run_count": max((len(values) for values in all_values.values()), default=0),
        }
        for metric in METRICS:
            values = all_values[metric]
            average_row[f"{metric}_avg_reduction_percent"] = sum(values) / len(values) if values else ""
            average_row[f"{metric}_min_reduction_percent"] = min(values) if values else ""
            average_row[f"{metric}_max_reduction_percent"] = max(values) if values else ""
            average_row[f"{metric}_std_reduction_percent"] = _population_std(values) if values else ""
        result.append(average_row)
    return result


def _checkpoint(
    output: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_records_csv(output / "metrics.csv", rows)
    write_records_csv(output / "summary.csv", summary)
    write_json(output / "manifest.json", manifest)


def _print_summary(summary: list[dict[str, object]]) -> None:
    if not summary:
        return
    print("\nPauli-string order sensitivity: reductions versus GridSynth (%)")
    print("Std is the population standard deviation over the accumulated runs.")
    for row in summary:
        values: list[str] = []
        for metric in METRICS:
            values.append(
                f"{metric}: avg={float(row[f'{metric}_avg_reduction_percent']):.2f}, "
                f"min={float(row[f'{metric}_min_reduction_percent']):.2f}, "
                f"max={float(row[f'{metric}_max_reduction_percent']):.2f}, "
                f"std={float(row[f'{metric}_std_reduction_percent']):.2f}"
            )
        print(f"{row['benchmark']} (runs={row['run_count']}): " + "; ".join(values))


def run(
    output: Path | str = DEFAULT_OUTPUT,
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 1,
    runs: int = 1,
    synthesis_error: float = 0.001,
) -> dict[str, Any]:
    """Append ``runs`` randomized executions per benchmark to the CSVs."""

    if runs < 1:
        raise ValueError("runs must be a positive integer")
    if synthesis_error <= 0:
        raise ValueError("synthesis_error must be positive")
    if methods and any(method != "ncf-one" for method in methods):
        raise ValueError("Pauli-string order sensitivity only supports method ncf-one")

    output_path = Path(output)
    selected_names = _canonical_benchmarks(benchmarks)
    selected_specs = [find_benchmark(name) for name in selected_names]
    source_script = _source_script()

    # Load baselines for all four experiment benchmarks so a later command
    # selecting only one benchmark never drops rows accumulated for another.
    grid_by_benchmark: dict[str, dict[str, int]] = {}
    grid_sources: dict[str, str] = {}
    for name in DEFAULT_BENCHMARKS:
        spec = find_benchmark(name)
        grid_by_benchmark[spec.name], grid_sources[spec.name] = _grid_metrics(spec, gpu)

    existing = _normalize_existing_rows(_read_rows(output_path / "metrics.csv"), grid_by_benchmark)
    rows = list(existing)
    next_run_id = 1
    for row in rows:
        try:
            next_run_id = max(next_run_id, int(str(row.get("run_id", ""))) + 1)
        except ValueError:
            continue

    manifest: dict[str, object] = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": str(source_script),
        "benchmarks": selected_names,
        "window": 4,
        "synthesis_error": synthesis_error,
        "trotter_steps": 1,
        "evolution_time": 1.0,
        "gpu": gpu,
        "seed_offset": seed,
        "runs_added": runs,
        "append_policy": "existing rows are preserved; each completed benchmark is checkpointed",
        "csv_merge_policy": "append new benchmark/run_id rows; accumulated summary is rebuilt",
        "metrics_csv": str(output_path / "metrics.csv"),
        "summary_csv": str(output_path / "summary.csv"),
        "grid_sources": grid_sources,
    }

    for offset in range(runs):
        run_id = next_run_id + offset
        run_seed = seed + run_id
        for spec in selected_specs:
            ncf = run_benchmark(
                spec,
                method="ncf-one",
                settings={
                    "single_window": 4,
                    "synthesis_error": synthesis_error,
                    "trotter_steps": 1,
                    "evolution_time": 1.0,
                    "t_budget": 60,
                    "pauli_order_seed": run_seed,
                },
                gpu=gpu,
                save_qasm=False,
            )
            grid = grid_by_benchmark[spec.name]
            row: dict[str, object] = {
                "benchmark": spec.name,
                "run_id": run_id,
                "seed": run_seed,
                "grid_t_count": grid["t_count"],
                "grid_t_depth": grid["t_depth"],
                "grid_clifford_count": grid["clifford_count"],
                "ncf_t_count": int(ncf["t_count"]),
                "ncf_t_depth": int(ncf["t_depth"]),
                "ncf_clifford_count": int(ncf["clifford_count"]),
                "t_count_reduction_percent": _reduction(grid["t_count"], int(ncf["t_count"])),
                "t_depth_reduction_percent": _reduction(grid["t_depth"], int(ncf["t_depth"])),
                "clifford_count_reduction_percent": _reduction(
                    grid["clifford_count"], int(ncf["clifford_count"])
                ),
                "synthesis_error": synthesis_error,
                "runtime_seconds": ncf.get("runtime_seconds", ""),
                "data_source": "2025summer/random_order.py",
            }
            rows.append(row)
            manifest["record_count"] = len(rows)
            manifest["last_completed_benchmark"] = spec.name
            manifest["last_completed_run_id"] = run_id
            _checkpoint(output_path, rows, _summary(rows), manifest)

    summary = _summary(rows)
    manifest["record_count"] = len(rows)
    _checkpoint(output_path, rows, summary, manifest)
    _print_summary(summary)
    return {"manifest": manifest, "records": rows, "summary": summary}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run NC-Fusion Pauli-string order sensitivity")
    add_cli_arguments(parser)
    parser.set_defaults(output=DEFAULT_OUTPUT, gpu=1)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="number of new randomized runs per selected benchmark (default: 1)",
    )
    parser.add_argument(
        "--precision",
        "--synthesis-error",
        dest="synthesis_error",
        type=float,
        default=0.001,
        help="NC-Fusion synthesis precision/error threshold (default: 0.001)",
    )
    args = parser.parse_args()
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            runs=args.runs,
            synthesis_error=args.synthesis_error,
        )
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {len(result['records'])} accumulated records; wrote {args.output}")


if __name__ == "__main__":
    _main()
