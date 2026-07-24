"""Evaluate the paper's fault-tolerant spacetime-volume experiment.

This follows the original ``ncf/NCF/Superstaq_test.py`` driver: load the
stored GridSynth and single-qubit NC-Fusion Clifford+T QASM and estimate both
with Infleqtion's ``resource-superstaq`` pipeline for one and ten T factories.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import platform
from pathlib import Path
from typing import Any

from ncfusion.metrics import (
    merge_records,
    metrics_from_qasm_file,
    read_records_csv,
    write_json,
    write_records_csv,
)
from ncfusion.spec import find_benchmark, find_experiment

from .common import run_cli


STATUS = "available"

_ARTIFACT_ROOT = Path(__file__).resolve().parents[1]

# The original benchmark generator uses shorter directory names than the
# paper/configuration names used by the artifact harness.
_BENCHMARK_DIRS = {
    "Ising-2D-30": "IS-2D-30",
    "Ising-2D-60": "IS-2D-60",
    "Ising-3D-30": "IS-3D-30",
    "Ising-3D-60": "IS-3D-60",
    "Heisenberg-2D-30": "Hei-2D-30",
    "Heisenberg-2D-60": "Hei-2D-60",
    "Heisenberg-3D-30": "Hei-3D-30",
    "Heisenberg-3D-60": "Hei-3D-60",
}

_BENCHMARK_ALIASES = {
    "is-2d-30": "Ising-2D-30",
    "is-2d-60": "Ising-2D-60",
    "is-3d-30": "Ising-3D-30",
    "is-3d-60": "Ising-3D-60",
    "hei-2d-30": "Heisenberg-2D-30",
    "hei-2d-60": "Heisenberg-2D-60",
    "hei-3d-30": "Heisenberg-3D-30",
    "hei-3d-60": "Heisenberg-3D-60",
}

_METHOD_FILES = {
    "gridsyn": "grid",
    "ncf-one": "ncf",
}

_RESOURCE_FIELDS = (
    "re_superstaq_phys_qubits",
    "re_superstaq_parallel_time_us",
    "re_superstaq_serial_time_us",
    "re_superstaq_volume",
    "re_superstaq_d",
    "re_superstaq_num_t_factories",
    "re_superstaq_primitive_moments",
    "re_superstaq_error",
)


def _selected_benchmarks(experiment: Any, requested: list[str] | None) -> tuple[Any, ...]:
    """Canonicalize and validate the selected spacetime benchmarks."""

    allowed = {name.lower(): name for name in experiment.benchmarks}
    selected_names: list[str] = []
    for value in experiment.benchmarks if requested is None else requested:
        key = value.strip().lower()
        name = _BENCHMARK_ALIASES.get(key, allowed.get(key))
        if name is None:
            choices = ", ".join(experiment.benchmarks)
            raise ValueError(
                f"unknown spacetime-volume benchmark {value!r}; choose from: {choices}"
            )
        if name not in selected_names:
            selected_names.append(name)
    if not selected_names:
        raise ValueError("at least one spacetime-volume benchmark must be selected")
    return tuple(find_benchmark(name) for name in selected_names)


def _normalise_resource_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep CSV columns stable when one estimator row reports an error."""

    return {field: metrics.get(field) for field in _RESOURCE_FIELDS}


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("benchmark", "")),
        str(record.get("method", "")),
        str(record.get("magic_state_factories", "")),
    )


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for record in records:
        key = _record_key(record)
        if key not in by_key:
            order.append(key)
        by_key[key] = record
    return [by_key[key] for key in order]


def _record_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _estimates_from_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    estimates: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        method = str(record.get("method", ""))
        if method not in _METHOD_FILES:
            continue
        try:
            factory_count = int(record["magic_state_factories"])
        except (KeyError, TypeError, ValueError):
            continue
        estimate: dict[str, Any] = {}
        for field in _RESOURCE_FIELDS:
            value = record.get(field)
            estimate[field] = value if field == "re_superstaq_error" else _record_number(value)
        estimates[(str(record.get("benchmark", "")), factory_count, method)] = estimate
    return estimates


def _completed_benchmarks(
    selected: tuple[Any, ...],
    factory_counts: tuple[int, ...],
    estimates: dict[tuple[str, int, str], dict[str, Any]],
) -> set[str]:
    return {
        benchmark.name
        for benchmark in selected
        if all(
            (benchmark.name, factory_count, method) in estimates
            for factory_count in factory_counts
            for method in ("gridsyn", "ncf-one")
        )
    }


def _build_relative_records(
    selected: tuple[Any, ...],
    factory_counts: tuple[int, ...],
    estimates: dict[tuple[str, int, str], dict[str, Any]],
    completed_benchmarks: set[str],
) -> list[dict[str, Any]]:
    relative_records: list[dict[str, Any]] = []
    for factory_count in factory_counts:
        for benchmark in selected:
            if benchmark.name not in completed_benchmarks:
                continue
            gridsyn = estimates.get((benchmark.name, factory_count, "gridsyn"), {})
            ncf = estimates.get((benchmark.name, factory_count, "ncf-one"), {})
            gridsyn_volume = gridsyn.get("re_superstaq_volume")
            ncf_volume = ncf.get("re_superstaq_volume")
            relative: dict[str, Any] = {
                "benchmark": benchmark.name,
                "magic_state_factories": factory_count,
                "gridsyn_spacetime_volume": gridsyn_volume,
                "ncf_spacetime_volume": ncf_volume,
                "ncf_relative_spacetime_volume_percent": None,
                "ncf_spacetime_volume_reduction_percent": None,
                "available": False,
                "gridsyn_error": gridsyn.get("re_superstaq_error"),
                "ncf_error": ncf.get("re_superstaq_error"),
            }
            if gridsyn_volume is not None and ncf_volume is not None:
                relative_percent = 100.0 * float(ncf_volume) / float(gridsyn_volume)
                relative["ncf_relative_spacetime_volume_percent"] = relative_percent
                relative["ncf_spacetime_volume_reduction_percent"] = 100.0 - relative_percent
                relative["available"] = True
            relative_records.append(relative)

        available = [
            row for row in relative_records
            if row["magic_state_factories"] == factory_count and row["available"]
        ]
        average: dict[str, Any] = {
            "benchmark": f"AVERAGE_{len(completed_benchmarks)}",
            "magic_state_factories": factory_count,
            "available_benchmarks": len(available),
            "gridsyn_spacetime_volume": (
                sum(float(row["gridsyn_spacetime_volume"]) for row in available) / len(available)
                if available else None
            ),
            "ncf_spacetime_volume": (
                sum(float(row["ncf_spacetime_volume"]) for row in available) / len(available)
                if available else None
            ),
            "ncf_relative_spacetime_volume_percent": (
                sum(float(row["ncf_relative_spacetime_volume_percent"]) for row in available) / len(available)
                if available else None
            ),
            "ncf_spacetime_volume_reduction_percent": (
                sum(float(row["ncf_spacetime_volume_reduction_percent"]) for row in available) / len(available)
                if available else None
            ),
            "available": bool(available),
            "gridsyn_error": None,
            "ncf_error": None,
        }
        relative_records.append(average)
    return relative_records


def _write_checkpoint(
    output_path: Path,
    experiment: Any,
    report_benchmarks: tuple[Any, ...],
    requested_benchmarks: tuple[Any, ...],
    chosen_methods: tuple[str, ...],
    resource_estimation: Any,
    factory_counts: tuple[int, ...],
    records: list[dict[str, Any]],
    estimates: dict[tuple[str, int, str], dict[str, Any]],
    completed_benchmarks: set[str],
    *,
    status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _deduplicate_records(records)
    relative_records = _build_relative_records(
        report_benchmarks, factory_counts, estimates, completed_benchmarks
    )
    manifest = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "experiment": asdict(experiment),
        "benchmarks": [asdict(item) for item in report_benchmarks],
        "requested_benchmarks": [item.name for item in requested_benchmarks],
        "completed_benchmarks": [
            item.name for item in report_benchmarks if item.name in completed_benchmarks
        ],
        "methods": chosen_methods,
        "record_count": len(records),
        "csv_merge_policy": "replace matching benchmark/method/magic_state_factories rows and append new configurations",
        "relative_record_count": len(relative_records),
        "relative_metrics_file": str(output_path / "relative_metrics.csv"),
        "relative_definition": "100 * NC-Fusion resource-superstaq volume / GridSynth resource-superstaq volume",
        "relative_average": "arithmetic mean of available per-benchmark relative percentages",
        "python": platform.python_version(),
        "resource_superstaq": {
            "package": "Infleqtion/resource-superstaq",
            "import_name": "resource_estimation",
            "revision": "717cbbfc62e558be3f2f9acb512e992d3cd43529",
            "location": str(getattr(resource_estimation, "__file__", "")),
        },
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    write_records_csv(output_path / "relative_metrics.csv", relative_records)
    return manifest, relative_records


def _check_resource_package() -> Any:
    try:
        return importlib.import_module("resource_estimation")
    except ImportError as error:
        raise RuntimeError(
            "Spacetime-volume evaluation requires Infleqtion/resource-superstaq. "
            "Install the pinned revision 717cbbfc62e558be3f2f9acb512e992d3cd43529 "
            "from https://github.com/Infleqtion/resource-superstaq.git, then rerun."
        ) from error


def run(
    output: Path | str = "micro_artifact/results/runs/space_volume_analysis",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    source: str = "existing",
) -> dict[str, Any]:
    """Run Section 5.10 from stored GridSynth and NC-Fusion QASM files."""

    del seed, gpu  # retained for the common artifact CLI contract
    if source not in {"existing", "generate"}:
        raise ValueError("source must be existing or generate")
    output_path = Path(output)
    experiment = find_experiment("spacetime-volume")
    selected = _selected_benchmarks(experiment, benchmarks)
    chosen_methods = tuple(methods or ("gridsyn", "ncf-one"))
    unknown = sorted(set(chosen_methods).difference(_METHOD_FILES))
    if unknown:
        known = ", ".join(_METHOD_FILES)
        raise ValueError(f"Unknown spacetime-volume method(s) {unknown}; choose from {known}")

    records = read_records_csv(output_path / "metrics.csv")
    factory_counts = tuple(int(value) for value in experiment.settings["magic_state_factories"])
    estimates = _estimates_from_records(records)
    if source == "existing":
        completed_benchmarks = _completed_benchmarks(selected, factory_counts, estimates)
        missing = [
            benchmark.name
            for benchmark in selected
            if benchmark.name not in completed_benchmarks
        ]
        if missing:
            raise FileNotFoundError(
                "Stored spacetime-volume results are missing for "
                + ", ".join(missing)
                + "; rerun with --source generate."
            )
        manifest, relative_records = _write_checkpoint(
            output_path,
            experiment,
            selected,
            selected,
            chosen_methods,
            None,
            factory_counts,
            records,
            estimates,
            completed_benchmarks,
            status="existing",
        )
        manifest["source_mode"] = "existing"
        write_json(output_path / "manifest.json", manifest)
        return {
            "manifest": manifest,
            "records": records,
            "relative_records": relative_records,
        }

    resource_estimation = _check_resource_package()
    from qiskit import QuantumCircuit
    from .resource_estimators import superstaq_estimate

    jobs_by_benchmark: dict[str, list[tuple[Any, str, Path]]] = {
        benchmark.name: [] for benchmark in selected
    }
    missing_files = []
    from .data import existing_qasm_path

    for benchmark in selected:
        for method in chosen_methods:
            qasm_path = existing_qasm_path(benchmark.name, method)
            if qasm_path is None:
                expected = Path("micro_artifact/circuits/single-qubit") / benchmark.name
                missing_files.append(str(expected / f"{benchmark.name}_{_METHOD_FILES[method]}_c+t.qasm"))
            else:
                jobs_by_benchmark[benchmark.name].append((benchmark, method, qasm_path))
    if missing_files:
        raise RuntimeError(
            "Spacetime-volume inputs are missing from this checkout: "
            + ", ".join(missing_files)
            + ". Supply the original Clifford+T QASM files before running the full experiment."
        )

    records = read_records_csv(output_path / "metrics.csv")
    estimates = _estimates_from_records(records)
    completed_benchmarks: set[str] = _completed_benchmarks(
        selected, factory_counts, estimates
    )
    relative_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    from .data import producer_record

    for benchmark in selected:
        for _, method, qasm_path in jobs_by_benchmark[benchmark.name]:
            qc = QuantumCircuit.from_qasm_file(str(qasm_path))
            circuit_metrics = metrics_from_qasm_file(qasm_path).as_dict()
            producer = producer_record(benchmark.name, method) or {}
            compilation_time = producer.get("compilation_time_seconds") or producer.get("runtime_seconds")
            if compilation_time not in (None, ""):
                compilation_time = float(compilation_time)
            else:
                compilation_time = None
            for factory_count in factory_counts:
                estimate = superstaq_estimate(qc, num_t_factories=factory_count)
                record = {
                    "benchmark": benchmark.name,
                    "method": method,
                    "source_method": _METHOD_FILES[method],
                    "magic_state_factories": factory_count,
                    "qasm_path": str(qasm_path.relative_to(_ARTIFACT_ROOT)),
                    "data_source": "existing_qasm",
                    "compilation_time_seconds": compilation_time,
                    "qasm_qubits": qc.num_qubits,
                    **{f"qasm_{key}": value for key, value in circuit_metrics.items()},
                    **_normalise_resource_metrics(estimate),
                }
                records = merge_records(
                    records,
                    [record],
                    ("benchmark", "method", "magic_state_factories"),
                )
                estimates[(benchmark.name, factory_count, method)] = estimate
        completed_benchmarks.add(benchmark.name)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest, relative_records = _write_checkpoint(
            output_path,
            experiment,
            selected,
            selected,
            chosen_methods,
            resource_estimation,
            factory_counts,
            records,
            estimates,
            completed_benchmarks,
            status="in_progress",
        )

    manifest, relative_records = _write_checkpoint(
        output_path,
        experiment,
        selected,
        selected,
        chosen_methods,
        resource_estimation,
        factory_counts,
        records,
        estimates,
        completed_benchmarks,
        status="complete",
    )
    manifest["source_mode"] = source
    write_json(output_path / "manifest.json", manifest)
    return {"manifest": manifest, "records": records, "relative_records": relative_records}


if __name__ == "__main__":
    run_cli("space_volume_analysis", run, include_source=True)
