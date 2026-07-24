"""Window-size sensitivity evaluation.

This is the paper's Section 5.5 sweep, with the thresholds requested for the
artifact evaluation.  It regenerates circuits in memory and checkpoints one
CSV row after every window, so an interrupted sweep retains completed work.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from ncfusion.legacy import run_benchmark
from ncfusion.metrics import metrics_from_qasm_file, write_json, write_records_csv
from ncfusion.spec import find_benchmark

from .common import add_cli_arguments
from .data import existing_qasm_path


STATUS = "available"
DEFAULT_OUTPUT = Path("micro_artifact/results/runs/window_size_sensitivity")
DEFAULT_BENCHMARKS = (
    "LiH",
    "H2O",
    "Ising-2D-60",
    "Heisenberg-2D-60",
)
SINGLE_WINDOWS = ("full", 128, 64, 32, 16, 8, 4)
TWO_WINDOWS = ("full", 256, 128, 64, 32, 16)
METHODS = ("ncf-one", "ncf-two")
METRICS = ("t_count", "t_depth", "clifford_count")

_ALIASES = {
    "is-2d-60": "Ising-2D-60",
    "hei-2d-60": "Heisenberg-2D-60",
}


def _benchmarks(requested: Iterable[str] | None) -> list[str]:
    names = list(requested) if requested is not None else list(DEFAULT_BENCHMARKS)
    allowed = {name.lower(): name for name in DEFAULT_BENCHMARKS}
    selected: list[str] = []
    for name in names:
        key = name.strip().lower()
        canonical = _ALIASES.get(key, allowed.get(key))
        if canonical is None:
            raise ValueError(
                f"{name!r} is not in this experiment; choose LiH, H2O, "
                "Ising-2D-60, or Heisenberg-2D-60"
            )
        if canonical not in selected:
            selected.append(canonical)
    return selected


def _threshold(method: str, benchmark: str, single: float, two_molecule: float, two_spin: float) -> float:
    if method == "ncf-one":
        return single
    return two_molecule if benchmark in {"LiH", "H2O"} else two_spin


def _grid_baseline(benchmark: str) -> dict[str, int]:
    path = existing_qasm_path(benchmark, "gridsyn")
    if path is None:
        raise FileNotFoundError(f"No stored GridSynth QASM for {benchmark}")
    return metrics_from_qasm_file(path).as_dict()


def _reduction(reference: int, candidate: int) -> float | None:
    if reference == 0:
        return None
    return 100.0 * (reference - candidate) / reference


def _record_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    threshold = row.get("threshold", row.get("error_threshold", ""))
    return (
        str(row.get("benchmark", "")),
        str(row.get("method", "")),
        str(row.get("window", "")),
        str(threshold),
    )


def _merge(records: list[dict[str, object]], update: dict[str, object]) -> list[dict[str, object]]:
    key = _record_key(update)
    for index, record in enumerate(records):
        if _record_key(record) == key:
            records[index] = update
            return records
    records.append(update)
    return records


def _read_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _docker_status() -> str | None:
    if shutil.which("docker") is None:
        return "docker executable was not found"
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Docker is unavailable: {error}"
    if result.returncode != 0:
        detail = (result.stderr or "Docker daemon is unavailable").strip().splitlines()
        return detail[-1] if detail else "Docker daemon is unavailable"
    return None


def _relative_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize each window to its method/benchmark ``full`` result."""

    metrics = (*METRICS, "compilation_time_seconds")
    completed = [
        row for row in records if row.get("status", "completed") == "completed"
    ]
    full = {
        (str(row.get("benchmark")), str(row.get("method")), str(row.get("threshold"))): row
        for row in completed
        if str(row.get("window")) == "full"
    }
    relative: list[dict[str, object]] = []
    for row in completed:
        key = (
            str(row.get("benchmark")),
            str(row.get("method")),
            str(row.get("threshold")),
        )
        reference = full.get(key)
        if reference is None:
            continue
        item: dict[str, object] = {
            "benchmark": row.get("benchmark"),
            "method": row.get("method"),
            "window": row.get("window"),
            "threshold": row.get("threshold"),
            "status": "completed",
        }
        for metric in metrics:
            value = row.get(metric)
            baseline = reference.get(metric)
            try:
                item[f"{metric}_relative_percent"] = (
                    100.0 * float(value) / float(baseline)
                    if float(baseline) != 0
                    else None
                )
            except (TypeError, ValueError):
                item[f"{metric}_relative_percent"] = None
        relative.append(item)

    methods = sorted({str(row.get("method")) for row in relative})
    windows = {method: [str(window) for window in (SINGLE_WINDOWS if method == "ncf-one" else TWO_WINDOWS)] for method in methods}
    for method in methods:
        for window in windows[method]:
            rows_for_average = [
                row for row in relative
                if str(row.get("method")) == method and str(row.get("window")) == window
            ]
            if not rows_for_average:
                continue
            average: dict[str, object] = {
                "benchmark": f"AVERAGE_{len(rows_for_average)}",
                "method": method,
                "window": window,
                "threshold": rows_for_average[0].get("threshold"),
                "status": "average",
                "available_benchmarks": len(rows_for_average),
            }
            for metric in metrics:
                values = [
                    float(row[f"{metric}_relative_percent"])
                    for row in rows_for_average
                    if row.get(f"{metric}_relative_percent") not in (None, "")
                ]
                average[f"{metric}_relative_percent"] = sum(values) / len(values) if values else None
            relative.append(average)
    return relative


def _write_checkpoint(
    output: Path,
    records: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_records_csv(output / "metrics.csv", records)
    write_records_csv(output / "relative_metrics.csv", _relative_records(records))
    write_json(output / "manifest.json", manifest)


def run(
    output: Path | str = DEFAULT_OUTPUT,
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 1,
    single_threshold: float = 0.005,
    two_molecule_threshold: float = 0.03,
    two_spin_threshold: float = 0.07,
    source: str = "existing",
) -> dict[str, Any]:
    """Run and checkpoint the requested single- and two-qubit sweeps."""

    if any(value <= 0 for value in (single_threshold, two_molecule_threshold, two_spin_threshold)):
        raise ValueError("all synthesis thresholds must be positive")
    chosen_methods = list(methods or METHODS)
    unknown = [method for method in chosen_methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unsupported sensitivity method(s): {', '.join(unknown)}")
    if source not in {"existing", "generate"}:
        raise ValueError("source must be existing or generate")
    selected = _benchmarks(benchmarks)
    output_path = Path(output)
    records: list[dict[str, object]] = list(_read_records(output_path / "metrics.csv"))
    if source == "existing":
        if not records:
            raise FileNotFoundError(
                "No stored window-sensitivity results were found; rerun with "
                "--source generate."
            )
        manifest = {
            "artifact_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation": "window_size_sensitivity",
            "benchmarks": selected,
            "methods": chosen_methods,
            "source_mode": "existing",
            "relative_metrics_file": str(output_path / "relative_metrics.csv"),
            "csv_merge_policy": "read existing checkpointed results",
        }
        manifest["record_count"] = len(records)
        manifest["relative_record_count"] = len(_relative_records(records))
        _write_checkpoint(output_path, records, manifest)
        return {"manifest": manifest, "records": records}
    baselines = {benchmark: _grid_baseline(benchmark) for benchmark in selected}
    docker_error = _docker_status() if "ncf-two" in chosen_methods else None

    manifest: dict[str, object] = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "window_size_sensitivity",
        "paper_section": "5.5",
        "benchmarks": selected,
        "methods": chosen_methods,
        "single_windows": list(SINGLE_WINDOWS),
        "two_windows": list(TWO_WINDOWS),
        "thresholds": {
            "ncf-one": single_threshold,
            "ncf-two:LiH/H2O": two_molecule_threshold,
            "ncf-two:Ising-2D-60/Heisenberg-2D-60": two_spin_threshold,
        },
        "gpu": gpu,
        "seed": seed,
        "source_mode": source,
        "save_qasm": False,
        "relative_metrics_file": str(output_path / "relative_metrics.csv"),
        "relative_definition": "100 * metric(window) / metric(full) for the same benchmark, method, and threshold",
        "threshold_reference": "GridSynth threshold; NC-Fusion applies normalized scaling internally",
        "append_policy": "existing keys are replaced; new keys are appended; each job is checkpointed",
        "csv_merge_policy": "replace matching benchmark/method/window/threshold rows and append new configurations",
        "docker_preflight": docker_error or "available",
    }

    for benchmark in selected:
        for method in chosen_methods:
            windows = SINGLE_WINDOWS if method == "ncf-one" else TWO_WINDOWS
            threshold = _threshold(
                method,
                benchmark,
                single_threshold,
                two_molecule_threshold,
                two_spin_threshold,
            )
            for window in windows:
                row: dict[str, object] = {
                    "benchmark": benchmark,
                    "method": method,
                    "window": window,
                    "window_size": "",
                    "threshold": threshold,
                    "synthesis_error": threshold,
                    "two_qubit_error": threshold if method == "ncf-two" else "",
                    "status": "error",
                    "error": "",
                }
                baseline = baselines[benchmark]
                for metric in METRICS:
                    row[f"gridsyn_{metric}"] = baseline[metric]
                    row[f"{metric}"] = ""
                    row[f"{metric}_reduction_percent"] = ""

                if method == "ncf-two" and docker_error is not None:
                    row["error"] = docker_error
                else:
                    settings: dict[str, object] = {
                        "window": window,
                        "synthesis_error": threshold,
                        "trotter_steps": 1,
                        "evolution_time": 1.0,
                        "t_budget": 60,
                    }
                    if method == "ncf-one":
                        settings["single_window"] = window
                    else:
                        settings["two_window"] = window
                        settings["two_qubit_error"] = threshold
                    try:
                        generated = run_benchmark(
                            find_benchmark(benchmark),
                            method=method,
                            settings=settings,
                            gpu=gpu,
                            save_qasm=False,
                        )
                        row.update(generated)
                        row.update(
                            {
                                "benchmark": benchmark,
                                "method": method,
                                "window": window,
                                "window_size": (
                                    generated.get("original_rz_gate_count", "")
                                    if window == "full"
                                    else window
                                ),
                                "threshold": threshold,
                                "synthesis_error": threshold,
                                "two_qubit_error": threshold if method == "ncf-two" else "",
                                "status": "completed",
                                "error": "",
                            }
                        )
                        for metric in METRICS:
                            row[f"gridsyn_{metric}"] = baseline[metric]
                            row[f"{metric}_reduction_percent"] = _reduction(
                                baseline[metric], int(generated[metric])
                            )
                    except Exception as error:  # checkpoint failures and continue to the next window
                        row["error"] = f"{type(error).__name__}: {error}"

                records = _merge(records, row)
                manifest["record_count"] = len(records)
                manifest["last_completed_key"] = {
                    "benchmark": benchmark,
                    "method": method,
                    "window": window,
                }
                _write_checkpoint(output_path, records, manifest)

    manifest["completed_record_count"] = sum(row.get("status") == "completed" for row in records)
    manifest["error_record_count"] = sum(row.get("status") == "error" for row in records)
    manifest["relative_record_count"] = len(_relative_records(records))
    _write_checkpoint(output_path, records, manifest)
    return {"manifest": manifest, "records": records}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run NC-Fusion window-size sensitivity")
    add_cli_arguments(parser)
    parser.set_defaults(output=DEFAULT_OUTPUT, gpu=1)
    # Sensitivity inputs are regenerated only when explicitly requested.
    parser.add_argument(
        "--source",
        choices=("existing", "generate"),
        default="existing",
        help="read stored sweep results by default; use generate to rerun them",
    )
    parser.add_argument("--single-threshold", type=float, default=0.005)
    parser.add_argument("--two-molecule-threshold", type=float, default=0.03)
    parser.add_argument("--two-spin-threshold", type=float, default=0.07)
    args = parser.parse_args()
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            single_threshold=args.single_threshold,
            two_molecule_threshold=args.two_molecule_threshold,
            two_spin_threshold=args.two_spin_threshold,
            source=args.source,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed/recorded {len(result['records'])} sensitivity rows; wrote {args.output}")


if __name__ == "__main__":
    _main()
