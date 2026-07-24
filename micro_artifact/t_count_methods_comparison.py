"""Compare tzap, PyZX, GridSynth, and T-Optimizer.

For every benchmark the workflow is:

* build the original Clifford+RZ circuit and the GridSynth Clifford+T circuit;
* run tzap on the original Clifford+RZ circuit;
* synthesize the tzap output with GridSynth and run tzap a second time;
* run PyZX only on the GridSynth Clifford+T circuit, synthesizing any RZ
  gates introduced by PyZX;
* run T-Optimizer on the Clifford+T circuit produced by GridSynth.

The output is long-form ``metrics.csv`` plus all intermediate QASM and
T-Optimizer ``.qc`` files.  The external tools are intentionally invoked at
runtime, so their versions and paths remain visible in the artifact manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

from ncfusion.metrics import merge_records, read_records_csv, write_json, write_records_csv
from ncfusion.runner import dependency_status
from ncfusion.spec import find_benchmark, find_experiment

from .from_qiskit import optimize_with_toptimizer
from .optimizer_common import circuit_metrics, run_tzap, synthesize_rz, write_qasm
from .pyzx import convert_pyzx_rz_to_clifford_t, optimize as optimize_pyzx


STATUS = "available"
DEFAULT_METHODS = ("gridsyn", "tzap", "pyzx", "t-optimizer")
DEFAULT_BENCHMARKS = tuple(find_experiment("table4").benchmarks) + ("H2S", "CO2")
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
    """Return canonical, deduplicated optimizer benchmarks.

    The command line accepts both the full benchmark names and the short
    directory-style ``IS-*``/``Hei-*`` names used by the artifact circuits.
    """

    if requested is None:
        return list(DEFAULT_BENCHMARKS)
    canonical = {name.lower(): name for name in DEFAULT_BENCHMARKS}
    selected: list[str] = []
    for value in requested:
        key = value.strip().lower()
        name = BENCHMARK_ALIASES.get(key, canonical.get(key))
        if name is None:
            choices = ", ".join(DEFAULT_BENCHMARKS)
            raise ValueError(
                f"unknown optimizer benchmark {value!r}; choose from: {choices}"
            )
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("at least one optimizer benchmark must be selected")
    return selected


def _canonical_methods(methods: list[str] | None) -> tuple[str, ...]:
    aliases = {
        "grid": "gridsyn",
        "gridsynth": "gridsyn",
        "t_optimizer": "t-optimizer",
        "toptimizer": "t-optimizer",
    }
    selected = tuple(aliases.get(method.lower(), method.lower()) for method in (methods or DEFAULT_METHODS))
    unknown = [method for method in selected if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(
            "methods must be selected from gridsyn, tzap, pyzx, and t-optimizer; "
            f"received {', '.join(unknown)}"
        )
    return tuple(dict.fromkeys(selected))


def _record(
    benchmark: str,
    method: str,
    stage: str,
    circuit: Any,
    qasm_path: Path,
    *,
    runtime_seconds: float = 0.0,
    compilation_time_seconds: float | None = None,
    data_source: str = "generated",
    command: str = "",
    source_stage: str = "",
    pyzx_stats: dict[str, Any] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark": benchmark,
        "method": method,
        "stage": stage,
        "source_stage": source_stage,
        "qasm_path": str(qasm_path),
        "runtime_seconds": round(runtime_seconds, 6),
        "compilation_time_seconds": (
            round(compilation_time_seconds, 6)
            if compilation_time_seconds is not None
            else (round(runtime_seconds, 6) if data_source != "existing_qasm" else None)
        ),
        "data_source": data_source,
        "command": command,
        "is_clifford_t": circuit_metrics(circuit)["rz_count"] == 0,
    }
    row.update(circuit_metrics(circuit))
    if pyzx_stats:
        row["pyzx_stats"] = str(pyzx_stats)
    return row


def _build_reference_circuits(
    spec: Any,
    settings: dict[str, object],
    gpu: int,
    *,
    need_gridsyn: bool,
) -> tuple[Any, Any | None]:
    from .data import existing_qasm_path

    existing_rz = existing_qasm_path(spec.name, "gridsyn", synthesized=False)
    existing_grid = existing_qasm_path(spec.name, "gridsyn", synthesized=True)
    if existing_rz is not None and (not need_gridsyn or existing_grid is not None):
        from qiskit import QuantumCircuit

        rz_qc = QuantumCircuit.from_qasm_file(str(existing_rz))
        gridsyn_qc = (
            QuantumCircuit.from_qasm_file(str(existing_grid))
            if need_gridsyn and existing_grid is not None
            else None
        )
        return rz_qc, gridsyn_qc

    from ncfusion.legacy import build_hamiltonian
    from baseline import baseline_circuit

    hamiltonian = build_hamiltonian(spec)
    rz_qc, gridsyn_qc = baseline_circuit(
        hamiltonian,
        1,
        error_threshold=float(settings["error_threshold"]),
        gpu=gpu,
        Trotter_steps=int(settings["trotter_steps"]),
        evolution_time=float(settings["evolution_time"]),
        rustiq=False,
        GRIDSYNTH=True,
        t_budget=int(settings["t_budget"]),
        synthesize=True,
        benchmark=None,
        method="grid",
    )
    return rz_qc, gridsyn_qc


def _merge_reduction_file(
    path: Path,
    updates: list[dict[str, object]],
    method: str,
) -> list[dict[str, object]]:
    """Merge per-benchmark reductions and rebuild one aggregate row."""

    existing = [
        row
        for row in read_records_csv(path)
        if not str(row.get("benchmark", "")).startswith("AVERAGE_")
    ]
    merged = merge_records(existing, updates, ("benchmark", "method"))
    if not merged:
        return []

    optimized_prefix = {
        "tzap": "tzap",
        "pyzx": "pyzx",
        "t-optimizer": "t_optimizer",
    }[method]
    average: dict[str, object] = {
        "benchmark": f"AVERAGE_{len(merged)}",
        "method": method,
        "baseline": "gridsyn",
        "baseline_qasm_path": "",
        "optimized_qasm_path": "",
    }
    for metric in ("t_count", "t_depth", "clifford_count"):
        fields = (
            f"gridsyn_{metric}",
            f"{optimized_prefix}_{metric}",
            f"{metric}_reduction_percent",
        )
        for field in fields:
            values = [
                float(row[field])
                for row in merged
                if row.get(field) not in (None, "")
            ]
            average[field] = sum(values) / len(values) if values else None
    merged.append(average)
    write_records_csv(path, merged)
    return merged


def run(
    output: Path | str = "micro_artifact/results/runs/t_count_methods_comparison",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    tzap_bin: str | None = None,
    t_optimizer_root: str | Path | None = None,
    source: str = "existing",
    install_missing: bool = False,
) -> dict[str, Any]:
    """Run the requested external optimizer comparison."""

    if error_threshold <= 0:
        raise ValueError("error_threshold must be positive")
    if t_budget < 1 or trotter_steps < 1:
        raise ValueError("t_budget and trotter_steps must be positive")
    if source not in {"existing", "generate"}:
        raise ValueError("source must be existing or generate")
    if install_missing and source != "generate":
        raise ValueError("--install-missing requires --source generate")

    selected_benchmarks = _selected_benchmarks(benchmarks)
    selected_methods = _canonical_methods(methods)
    output_path = Path(output)
    if source == "existing":
        cached_records = read_records_csv(output_path / "metrics.csv")
        cached_benchmarks = {
            str(row.get("benchmark", ""))
            for row in cached_records
            if not str(row.get("benchmark", "")).startswith("AVERAGE_")
        }
        missing = [benchmark for benchmark in selected_benchmarks if benchmark not in cached_benchmarks]
        if missing:
            raise FileNotFoundError(
                "Stored optimizer results are missing for "
                + ", ".join(missing)
                + "; rerun with --source generate."
            )
        manifest_path = output_path / "manifest.json"
        manifest = {}
        if manifest_path.is_file():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "source_mode": "existing",
                "benchmarks": selected_benchmarks,
                "methods": selected_methods,
            }
        )
        return {"manifest": manifest, "records": cached_records}

    circuit_dir = output_path / "circuits"
    circuit_dir.mkdir(parents=True, exist_ok=True)
    tzap_command = tzap_bin or os.environ.get("TZAP_BIN", "tzap")
    optimizer_installation: dict[str, str] = {}
    if install_missing:
        from .install_optimizers import ensure_optimizers

        optimizer_installation = ensure_optimizers(
            selected_methods,
            tzap_bin=tzap_bin,
            t_optimizer_root=t_optimizer_root,
        )
        tzap_command = optimizer_installation.get("tzap_bin", tzap_command)
        t_optimizer_root = optimizer_installation.get("t_optimizer_root", t_optimizer_root)
    settings = {
        "error_threshold": error_threshold,
        "t_budget": t_budget,
        "trotter_steps": trotter_steps,
        "evolution_time": evolution_time,
    }

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    records: list[dict[str, object]] = []
    tzap_reductions: list[dict[str, object]] = []
    t_optimizer_reductions: list[dict[str, object]] = []
    pyzx_reductions: list[dict[str, object]] = []
    pyzx_failures: list[dict[str, str]] = []
    need_gridsyn = any(method in selected_methods for method in ("gridsyn", "pyzx", "t-optimizer"))

    def checkpoint() -> None:
        merged = merge_records(
            read_records_csv(output_path / "metrics.csv"),
            records,
            ("benchmark", "method", "stage"),
        )
        write_records_csv(output_path / "metrics.csv", merged)
        for path, updates, method in (
            (output_path / "tzap_reductions.csv", tzap_reductions, "tzap"),
            (
                output_path / "t_optimizer_reductions.csv",
                t_optimizer_reductions,
                "t-optimizer",
            ),
            (output_path / "pyzx_reductions.csv", pyzx_reductions, "pyzx"),
        ):
            if updates:
                _merge_reduction_file(path, updates, method)

    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        reference_started = time.perf_counter()
        rz_qc, gridsyn_qc = _build_reference_circuits(
            spec, settings, gpu, need_gridsyn=need_gridsyn
        )
        from .data import existing_qasm_path, producer_record

        reference_is_existing = existing_qasm_path(spec.name, "gridsyn", synthesized=False) is not None
        grid_record = producer_record(spec.name, "gridsyn") or {}
        reference_compilation_time = grid_record.get("compilation_time_seconds") or grid_record.get("runtime_seconds")
        if reference_compilation_time is not None:
            reference_compilation_time = float(reference_compilation_time)
        reference_source = "existing_qasm" if reference_is_existing else "generated"
        if not reference_is_existing:
            reference_compilation_time = time.perf_counter() - reference_started
        original_path = write_qasm(rz_qc, circuit_dir / f"{spec.name}_original_clifford_rz.qasm")
        records.append(
            _record(
                spec.name,
                "input",
                "original_clifford_rz",
                rz_qc,
                original_path,
                compilation_time_seconds=reference_compilation_time,
                data_source=reference_source,
                source_stage="Hamiltonian simulation",
            )
        )

        if "gridsyn" in selected_methods:
            if gridsyn_qc is None:
                raise RuntimeError(f"No GridSynth reference circuit available for {spec.name}")
            grid_path = write_qasm(gridsyn_qc, circuit_dir / f"{spec.name}_gridsyn_clifford_t.qasm")
            records.append(
                _record(
                    spec.name,
                    "gridsyn",
                    "gridsyn_clifford_t",
                    gridsyn_qc,
                    grid_path,
                    compilation_time_seconds=reference_compilation_time,
                    data_source=reference_source,
                    source_stage="original_clifford_rz",
                )
            )

        if "tzap" in selected_methods:
            # tzap_test.py constructs the comparison baseline by synthesizing
            # the original grid_rz circuit with the same GridSynth plugin.
            # Do not use the separately stored grid_c+t file for this arm.
            original_synth_qc = synthesize_rz(rz_qc, error_threshold)
            original_synth_path = write_qasm(
                original_synth_qc,
                circuit_dir / f"{spec.name}_original_gridsyn_clifford_t.qasm",
            )
            tzap_rz_path = circuit_dir / f"{spec.name}_tzap_clifford_rz.qasm"
            tzap_rz, elapsed, command = run_tzap(
                rz_qc,
                circuit_dir / f"{spec.name}_tzap_input.qasm",
                tzap_rz_path,
                tzap_command,
            )
            records.append(
                _record(
                    spec.name,
                    "tzap",
                    "tzap_on_original_clifford_rz",
                    tzap_rz,
                    tzap_rz_path,
                    runtime_seconds=elapsed,
                    command=command,
                    source_stage="original_clifford_rz",
                )
            )

            tzap_grid = synthesize_rz(tzap_rz, error_threshold)
            tzap_grid_path = write_qasm(
                tzap_grid,
                circuit_dir / f"{spec.name}_tzap_then_gridsyn_clifford_t.qasm",
            )
            records.append(
                _record(
                    spec.name,
                    "tzap",
                    "tzap_then_gridsyn_clifford_t",
                    tzap_grid,
                    tzap_grid_path,
                    source_stage="tzap_on_original_clifford_rz",
                )
            )

            tzap_final_path = circuit_dir / f"{spec.name}_tzap_final_clifford_t.qasm"
            tzap_final, elapsed, command = run_tzap(
                tzap_grid,
                circuit_dir / f"{spec.name}_tzap_second_input.qasm",
                tzap_final_path,
                tzap_command,
            )
            records.append(
                _record(
                    spec.name,
                    "tzap",
                    "tzap_after_rz_synthesis",
                    tzap_final,
                    tzap_final_path,
                    runtime_seconds=elapsed,
                    command=command,
                    source_stage="tzap_then_gridsyn_clifford_t",
                )
            )
            reference_metrics = circuit_metrics(original_synth_qc)
            optimized_metrics = circuit_metrics(tzap_final)
            reduction: dict[str, object] = {
                "benchmark": spec.name,
                "method": "tzap",
                "baseline": "gridsyn",
                "baseline_qasm_path": str(original_synth_path),
                "optimized_qasm_path": str(tzap_final_path),
            }
            for metric in ("t_count", "t_depth", "clifford_count"):
                baseline_value = int(reference_metrics[metric])
                optimized_value = int(optimized_metrics[metric])
                reduction[f"gridsyn_{metric}"] = baseline_value
                reduction[f"tzap_{metric}"] = optimized_value
                reduction[f"{metric}_reduction_percent"] = (
                    100.0 * (baseline_value - optimized_value) / baseline_value
                    if baseline_value
                    else 0.0
                )
            tzap_reductions.append(reduction)

        if "pyzx" in selected_methods:
            if gridsyn_qc is None:
                raise RuntimeError(f"No GridSynth reference circuit available for {spec.name}")
            # This is the legacy 2025summer/pyzx_path.py workflow: PyZX
            # receives only the already synthesized GridSynth Clifford+T
            # circuit, drops extracted swap lines, and converts only exact
            # eighth-turn RZ phases back to Clifford+T.
            started = time.perf_counter()
            try:
                pyzx_from_grid, pyzx_grid_stats = optimize_pyzx(gridsyn_qc)
                pyzx_from_grid_ct = convert_pyzx_rz_to_clifford_t(pyzx_from_grid)
            except Exception as error:
                failure = {
                    "benchmark": spec.name,
                    "method": "pyzx",
                    "error": repr(error),
                }
                pyzx_failures.append(failure)
                records.append(
                    {
                        "benchmark": spec.name,
                        "method": "pyzx",
                        "stage": "pyzx_failed",
                        "source_stage": "gridsyn_clifford_t",
                        "qasm_path": "",
                        "runtime_seconds": round(time.perf_counter() - started, 6),
                        "data_source": "failed",
                        "command": "pyzx.full_reduce",
                        "error": repr(error),
                    }
                )
                print(f"WARNING: PyZX failed for {spec.name}: {error}", file=sys.stderr)
                checkpoint()
                continue
            pyzx_grid_elapsed = time.perf_counter() - started
            pyzx_grid_raw_path = write_qasm(
                pyzx_from_grid,
                circuit_dir / f"{spec.name}_pyzx_on_gridsyn.qasm",
            )
            records.append(
                _record(
                    spec.name,
                    "pyzx",
                    "pyzx_on_gridsyn_clifford_t",
                    pyzx_from_grid,
                    pyzx_grid_raw_path,
                    runtime_seconds=pyzx_grid_elapsed,
                    command="pyzx.full_reduce",
                    source_stage="gridsyn_clifford_t",
                    pyzx_stats=pyzx_grid_stats,
                )
            )
            pyzx_grid_final_path = write_qasm(
                pyzx_from_grid_ct,
                circuit_dir / f"{spec.name}_pyzx_on_gridsyn_clifford_t.qasm",
            )
            records.append(
                _record(
                    spec.name,
                    "pyzx",
                    "pyzx_on_gridsyn_after_rz_synthesis",
                    pyzx_from_grid_ct,
                    pyzx_grid_final_path,
                    source_stage="pyzx_on_gridsyn_clifford_t",
                )
            )
            reference_metrics = circuit_metrics(gridsyn_qc)
            optimized_metrics = circuit_metrics(pyzx_from_grid_ct)
            reduction: dict[str, object] = {
                "benchmark": spec.name,
                "method": "pyzx",
                "baseline": "gridsyn",
                "baseline_qasm_path": str(
                    existing_qasm_path(spec.name, "gridsyn", synthesized=True) or ""
                ),
                "optimized_qasm_path": str(pyzx_grid_final_path),
            }
            for metric in ("t_count", "t_depth", "clifford_count"):
                baseline_value = int(reference_metrics[metric])
                optimized_value = int(optimized_metrics[metric])
                reduction[f"gridsyn_{metric}"] = baseline_value
                reduction[f"pyzx_{metric}"] = optimized_value
                reduction[f"{metric}_reduction_percent"] = (
                    100.0 * (baseline_value - optimized_value) / baseline_value
                    if baseline_value
                    else 0.0
                )
            pyzx_reductions.append(reduction)

        if "t-optimizer" in selected_methods:
            if gridsyn_qc is None:
                raise RuntimeError(f"No GridSynth reference circuit available for {spec.name}")
            optimizer_path = circuit_dir / f"{spec.name}_t_optimizer_input.qc"
            started = time.perf_counter()
            optimized = optimize_with_toptimizer(gridsyn_qc, t_optimizer_root, optimizer_path)
            elapsed = time.perf_counter() - started
            optimized_path = write_qasm(optimized, circuit_dir / f"{spec.name}_t_optimizer.qasm")
            records.append(
                _record(
                    spec.name,
                    "t-optimizer",
                    "t_optimizer_on_gridsyn_clifford_t",
                    optimized,
                    optimized_path,
                    runtime_seconds=elapsed,
                    command="optimize.T_optimizer.remove_duplicates",
                    source_stage="gridsyn_clifford_t",
                )
            )
            reference_metrics = circuit_metrics(gridsyn_qc)
            optimized_metrics = circuit_metrics(optimized)
            reduction: dict[str, object] = {
                "benchmark": spec.name,
                "method": "t-optimizer",
                "baseline": "gridsyn",
                "baseline_qasm_path": str(
                    existing_qasm_path(spec.name, "gridsyn", synthesized=True) or ""
                ),
                "optimized_qasm_path": str(optimized_path),
            }
            for metric in ("t_count", "t_depth", "clifford_count"):
                baseline_value = int(reference_metrics[metric])
                optimized_value = int(optimized_metrics[metric])
                reduction[f"gridsyn_{metric}"] = baseline_value
                reduction[f"t_optimizer_{metric}"] = optimized_value
                reduction[f"{metric}_reduction_percent"] = (
                    100.0 * (baseline_value - optimized_value) / baseline_value
                    if baseline_value
                    else 0.0
                )
            t_optimizer_reductions.append(reduction)

        checkpoint()

    records = merge_records(
        read_records_csv(output_path / "metrics.csv"),
        records,
        ("benchmark", "method", "stage"),
    )
    if tzap_reductions:
        _merge_reduction_file(
            output_path / "tzap_reductions.csv", tzap_reductions, "tzap"
        )
    if t_optimizer_reductions:
        _merge_reduction_file(
            output_path / "t_optimizer_reductions.csv",
            t_optimizer_reductions,
            "t-optimizer",
        )
    if pyzx_reductions:
        _merge_reduction_file(
            output_path / "pyzx_reductions.csv", pyzx_reductions, "pyzx"
        )

    manifest = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "t_count_methods_comparison",
        "benchmarks": selected_benchmarks,
        "methods": selected_methods,
        "source_mode": source,
        "install_missing": install_missing,
        "optimizer_installation": optimizer_installation,
        "seed": seed,
        "gpu": gpu,
        "settings": settings,
        "tzap_bin": tzap_command,
        "t_optimizer_root": str(t_optimizer_root or os.environ.get("T_OPTIMIZER_ROOT", "")),
        "external_tools": {
            "tzap_found": shutil.which(tzap_command.split()[0]) is not None,
            "pyzx_importable": importlib.util.find_spec("pyzx") is not None,
            "t_optimizer_root_configured": bool(t_optimizer_root or os.environ.get("T_OPTIMIZER_ROOT")),
        },
        "paper_dependencies": dependency_status(),
        "record_count": len(records),
        "csv_merge_policy": "replace matching benchmark/method/stage rows and append new configurations; reduction CSVs merge by benchmark/method",
        "pyzx_failures": pyzx_failures,
        "workflow": {
            "tzap": (
                "grid_rz -> tzap pre -> per-RZ GridSynth synthesis -> tzap post; "
                "baseline is fresh per-RZ GridSynth synthesis of the same grid_rz"
            ),
            "pyzx": (
                "GridSynth Clifford+T -> PyZX full_reduce -> remove swaps -> "
                "exact eighth-turn RZ conversion"
            ),
            "t_optimizer": "GridSynth Clifford+T -> T-Optimizer IR -> optimized Clifford+T",
        },
    }
    if tzap_reductions:
        average: dict[str, object] = {
            "benchmark": f"AVERAGE_{len(tzap_reductions)}",
            "method": "tzap",
            "baseline": "gridsyn",
            "baseline_qasm_path": "",
            "optimized_qasm_path": "",
        }
        for metric in ("t_count", "t_depth", "clifford_count"):
            average[f"gridsyn_{metric}"] = sum(
                float(row[f"gridsyn_{metric}"]) for row in tzap_reductions
            ) / len(tzap_reductions)
            average[f"tzap_{metric}"] = sum(
                float(row[f"tzap_{metric}"]) for row in tzap_reductions
            ) / len(tzap_reductions)
            field = f"{metric}_reduction_percent"
            average[field] = sum(float(row[field]) for row in tzap_reductions) / len(tzap_reductions)
        _merge_reduction_file(output_path / "tzap_reductions.csv", tzap_reductions, "tzap")
        manifest["tzap_reduction_file"] = str(output_path / "tzap_reductions.csv")
        manifest["tzap_reduction_benchmark_count"] = len(tzap_reductions)
    if t_optimizer_reductions:
        average = {
            "benchmark": f"AVERAGE_{len(t_optimizer_reductions)}",
            "method": "t-optimizer",
            "baseline": "gridsyn",
            "baseline_qasm_path": "",
            "optimized_qasm_path": "",
        }
        for metric in ("t_count", "t_depth", "clifford_count"):
            average[f"gridsyn_{metric}"] = sum(
                float(row[f"gridsyn_{metric}"]) for row in t_optimizer_reductions
            ) / len(t_optimizer_reductions)
            average[f"t_optimizer_{metric}"] = sum(
                float(row[f"t_optimizer_{metric}"]) for row in t_optimizer_reductions
            ) / len(t_optimizer_reductions)
            field = f"{metric}_reduction_percent"
            average[field] = sum(float(row[field]) for row in t_optimizer_reductions) / len(t_optimizer_reductions)
        _merge_reduction_file(
            output_path / "t_optimizer_reductions.csv",
            t_optimizer_reductions,
            "t-optimizer",
        )
        manifest["t_optimizer_reduction_file"] = str(output_path / "t_optimizer_reductions.csv")
        manifest["t_optimizer_reduction_benchmark_count"] = len(t_optimizer_reductions)
    if pyzx_reductions:
        average = {
            "benchmark": f"AVERAGE_{len(pyzx_reductions)}",
            "method": "pyzx",
            "baseline": "gridsyn",
            "baseline_qasm_path": "",
            "optimized_qasm_path": "",
        }
        for metric in ("t_count", "t_depth", "clifford_count"):
            average[f"gridsyn_{metric}"] = sum(
                float(row[f"gridsyn_{metric}"]) for row in pyzx_reductions
            ) / len(pyzx_reductions)
            average[f"pyzx_{metric}"] = sum(
                float(row[f"pyzx_{metric}"]) for row in pyzx_reductions
            ) / len(pyzx_reductions)
            field = f"{metric}_reduction_percent"
            average[field] = sum(float(row[field]) for row in pyzx_reductions) / len(pyzx_reductions)
        _merge_reduction_file(output_path / "pyzx_reductions.csv", pyzx_reductions, "pyzx")
        manifest["pyzx_reduction_file"] = str(output_path / "pyzx_reductions.csv")
        manifest["pyzx_reduction_benchmark_count"] = len(pyzx_reductions)
    for method, filename in (
        ("tzap", "tzap_reductions.csv"),
        ("t_optimizer", "t_optimizer_reductions.csv"),
        ("pyzx", "pyzx_reductions.csv"),
    ):
        reduction_path = output_path / filename
        if reduction_path.is_file():
            reduction_rows = [
                row
                for row in read_records_csv(reduction_path)
                if not str(row.get("benchmark", "")).startswith("AVERAGE_")
            ]
            manifest[f"{method}_reduction_file"] = str(reduction_path)
            manifest[f"{method}_reduction_benchmark_count"] = len(reduction_rows)
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Compare T-count optimizers for NC-Fusion circuits")
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--output", type=Path, default=Path("micro_artifact/results/runs/t_count_methods_comparison"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--tzap-bin", default=None, help="tzap executable or command; defaults to TZAP_BIN/tzap")
    parser.add_argument("--t-optimizer-root", type=Path, default=None, help="cloned T-Optimizer repository")
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="with --source generate, install missing PyZX, T-Zap, and T-Optimizer dependencies",
    )
    parser.add_argument(
        "--source",
        choices=("existing", "generate"),
        default="existing",
        help="read stored optimizer results by default; use generate to rerun them",
    )
    args = parser.parse_args()
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            error_threshold=args.error_threshold,
            t_budget=args.t_budget,
            trotter_steps=args.trotter_steps,
            evolution_time=args.evolution_time,
            tzap_bin=args.tzap_bin,
            t_optimizer_root=args.t_optimizer_root,
            source=args.source,
            install_missing=args.install_missing,
        )
    except (ImportError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    _main()
