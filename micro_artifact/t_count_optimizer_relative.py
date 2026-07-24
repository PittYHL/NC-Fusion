"""Relative metric comparison for the completed optimizer benchmark set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.spec import find_experiment

from .data import existing_qasm_path


BENCHMARKS = tuple(find_experiment("table4").benchmarks) + ("H2S", "CO2")
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
METHODS = ("tzap", "t-optimizer", "pyzx", "ncf-one")
METRICS = ("t_count", "t_depth", "clifford_count")
METHOD_LABELS = {
    "gridsyn": "GridSynth",
    "tzap": "T-Zap",
    "pyzx": "PyZX",
    "t-optimizer": "Pauli rotation merging (T-optimizer)",
    "ncf-one": "NC-Fusion (single-qubit)",
}


def _metric(path: Path) -> dict[str, int]:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit.from_qasm_file(str(path))
    operations = list(circuit.data)
    counts = circuit.count_ops()
    t_count = int(counts.get("t", 0) + counts.get("tdg", 0))
    return {
        "t_count": t_count,
        "t_depth": int(
            circuit.depth(lambda item: item[0].name == "t" or item[0].name == "tdg")
        ),
        "clifford_count": int(len(operations) - t_count),
    }


def _optimizer_path(benchmark: str, method: str, root: Path) -> Path | None:
    if method == "ncf-one":
        path = existing_qasm_path(benchmark, method, synthesized=True)
        return path

    stems = {
        "tzap": ("t_count_methods_comparison_tzap_corrected", "tzap_final_clifford_t"),
        "pyzx": ("t_count_methods_comparison_pyzx", "pyzx_on_gridsyn_clifford_t"),
        "t-optimizer": ("t_count_methods_comparison_toptimizer", "t_optimizer"),
    }
    directory, stem = stems[method]
    consolidated = root / "t_count_optimizer_relative" / "circuits" / f"{benchmark}_{stem}.qasm"
    if consolidated.is_file():
        return consolidated
    path = root / f"{directory}_{benchmark}" / "circuits" / f"{benchmark}_{stem}.qasm"
    return path if path.is_file() else None


def _capped_relative(baseline: int, value: int) -> float:
    if baseline == 0:
        return 0.0
    return min(100.0, 100.0 * value / baseline)


def _cached_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["benchmark"]: row
            for row in csv.DictReader(handle)
            if row.get("benchmark") and not row["benchmark"].startswith("AVERAGE_")
        }


def _cached_number(value: str | None, *, integer: bool) -> int | float | None:
    if value in (None, ""):
        return None
    return int(float(value)) if integer else float(value)


def _selected_benchmarks(requested: list[str] | None) -> tuple[str, ...]:
    if requested is None:
        return BENCHMARKS
    canonical = {name.lower(): name for name in BENCHMARKS}
    selected: list[str] = []
    for value in requested:
        key = value.strip().lower()
        name = BENCHMARK_ALIASES.get(key, canonical.get(key))
        if name is None:
            choices = ", ".join(BENCHMARKS)
            raise ValueError(f"unknown optimizer benchmark {value!r}; choose from: {choices}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("at least one optimizer benchmark must be selected")
    return tuple(selected)


def _plot(average: dict[str, object], output: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [METHOD_LABELS[method] for method in METHODS]
    x = list(range(len(labels)))
    width = 0.22
    offsets = (-width, 0.0, width)
    colors = ("#4c78a8", "#f58518", "#54a24b")
    metric_labels = {
        "t_count": "T-count",
        "t_depth": "T-depth",
        "clifford_count": "Clifford count",
    }
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for offset, metric, color in zip(offsets, METRICS, colors):
        values = []
        for method in METHODS:
            value = average.get(f"{method}_{metric}_relative_percent")
            values.append(float(value) if value not in (None, "") else float("nan"))
        axis.bar(
            [value + offset for value in x],
            values,
            width=width,
            label=metric_labels[metric],
            color=color,
        )
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Average relative to GridSynth (%)")
    axis.set_title("Average T-count optimizer metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def run(
    output: Path | str = "micro_artifact/results/runs/t_count_optimizer_relative",
    *,
    benchmarks: list[str] | None = None,
) -> dict[str, object]:
    from .data import ARTIFACT_ROOT

    selected = _selected_benchmarks(benchmarks)
    output_path = Path(output)
    cached = _cached_rows(output_path / "relative_metrics.csv")
    rows: list[dict[str, object]] = []
    for benchmark in selected:
        baseline_path = existing_qasm_path(benchmark, "gridsyn", synthesized=True)
        if baseline_path is None:
            raise FileNotFoundError(f"No stored GridSynth QASM for {benchmark}")
        baseline = _metric(baseline_path)
        row: dict[str, object] = {
            "benchmark": benchmark,
            "baseline_qasm_path": str(baseline_path),
        }
        for metric in METRICS:
            row[f"gridsyn_{metric}"] = baseline[metric]
            row[f"gridsyn_{metric}_relative_percent"] = 100.0
        for method in METHODS:
            path = _optimizer_path(benchmark, method, ARTIFACT_ROOT / "results" / "runs")
            row[f"{method}_qasm_path"] = str(path) if path is not None else ""
            if path is None:
                cached_row = cached.get(benchmark)
                for metric in METRICS:
                    row[f"{method}_{metric}"] = _cached_number(
                        cached_row.get(f"{method}_{metric}") if cached_row else None,
                        integer=True,
                    )
                    row[f"{method}_{metric}_relative_percent"] = _cached_number(
                        cached_row.get(f"{method}_{metric}_relative_percent") if cached_row else None,
                        integer=False,
                    )
                continue
            metrics = _metric(path)
            for metric in METRICS:
                row[f"{method}_{metric}"] = metrics[metric]
                row[f"{method}_{metric}_relative_percent"] = _capped_relative(
                    baseline[metric], metrics[metric]
                )
        rows.append(row)

    # Preserve benchmark rows from earlier invocations when the user selects
    # only a subset. Selected benchmarks above replace their matching rows;
    # unselected cached rows are appended unchanged.
    selected_set = set(selected)
    rows.extend(
        cached[benchmark]
        for benchmark in BENCHMARKS
        if benchmark not in selected_set and benchmark in cached
    )

    average: dict[str, object] = {
        "benchmark": f"AVERAGE_{len(rows)}",
        "baseline_qasm_path": "",
    }
    for method in ("gridsyn", *METHODS):
        available_count = 0
        for metric in METRICS:
            values = [
                float(row[f"{method}_{metric}"])
                for row in rows
                if row.get(f"{method}_{metric}") not in (None, "")
            ]
            relative_values = [
                float(row[f"{method}_{metric}_relative_percent"])
                for row in rows
                if row.get(f"{method}_{metric}_relative_percent") not in (None, "")
            ]
            average[f"{method}_{metric}"] = sum(values) / len(values) if values else None
            average[f"{method}_{metric}_relative_percent"] = (
                sum(relative_values) / len(relative_values) if relative_values else None
            )
            available_count = max(available_count, len(values))
        average[f"{method}_available_count"] = available_count
    all_rows = [*rows, average]
    output_path.mkdir(parents=True, exist_ok=True)
    write_records_csv(output_path / "relative_metrics.csv", all_rows)
    plot_path = _plot(average, output_path / "relative_metrics.png")
    manifest = {
        "experiment": "t_count_optimizer_relative",
        "benchmarks": list(selected),
        "benchmark_count": len(selected),
        "methods": list(METHODS),
        "baseline": "stored GridSynth grid_c+t QASM",
        "relative_definition": "100 * method_metric / GridSynth_metric",
        "display_cap": 100.0,
        "averages": "arithmetic mean over available per-benchmark relative values",
        "available_counts": {
            method: average[f"{method}_available_count"]
            for method in ("gridsyn", *METHODS)
        },
        "t_depth_definition": "Qiskit depth filter for t and tdg",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "relative_metrics_file": str(output_path / "relative_metrics.csv"),
        "plot": str(plot_path) if plot_path is not None else None,
        "csv_merge_policy": "replace matching benchmark rows and append new benchmark rows; rebuild the average row",
    }
    write_json(output_path / "manifest.json", manifest)
    return {"manifest": manifest, "records": all_rows, "plot": plot_path}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="benchmark to include; repeat for multiple benchmarks (default: all 13)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("micro_artifact/results/runs/t_count_optimizer_relative"),
    )
    args = parser.parse_args()
    try:
        result = run(args.output, benchmarks=args.benchmarks)
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(f"Wrote {result['manifest']['relative_metrics_file']}")
    if result["plot"] is not None:
        print(f"Wrote {result['plot']}")


if __name__ == "__main__":
    _main()
