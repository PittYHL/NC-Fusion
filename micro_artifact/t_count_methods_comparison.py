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

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.runner import dependency_status
from ncfusion.spec import find_benchmark, find_experiment

from .from_qiskit import optimize_with_toptimizer
from .optimizer_common import circuit_metrics, run_tzap, synthesize_rz, write_qasm
from .pyzx import optimize as optimize_pyzx


STATUS = "available"
DEFAULT_METHODS = ("gridsyn", "tzap", "pyzx", "t-optimizer")


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
        "command": command,
        "is_clifford_t": circuit_metrics(circuit)["rz_count"] == 0,
    }
    row.update(circuit_metrics(circuit))
    if pyzx_stats:
        row["pyzx_stats"] = str(pyzx_stats)
    return row


def _build_reference_circuits(spec: Any, settings: dict[str, object], gpu: int) -> tuple[Any, Any]:
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


def run(
    output: Path | str = "results/runs/t_count_methods_comparison",
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
) -> dict[str, Any]:
    """Run the requested external optimizer comparison."""

    if error_threshold <= 0:
        raise ValueError("error_threshold must be positive")
    if t_budget < 1 or trotter_steps < 1:
        raise ValueError("t_budget and trotter_steps must be positive")

    experiment = find_experiment("optimizer-comparison")
    selected_benchmarks = benchmarks or list(experiment.benchmarks)
    selected_methods = _canonical_methods(methods)
    output_path = Path(output)
    circuit_dir = output_path / "circuits"
    circuit_dir.mkdir(parents=True, exist_ok=True)
    tzap_command = tzap_bin or os.environ.get("TZAP_BIN", "tzap")
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
    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        rz_qc, gridsyn_qc = _build_reference_circuits(spec, settings, gpu)
        original_path = write_qasm(rz_qc, circuit_dir / f"{spec.name}_original_clifford_rz.qasm")
        records.append(
            _record(
                spec.name,
                "input",
                "original_clifford_rz",
                rz_qc,
                original_path,
                source_stage="Hamiltonian simulation",
            )
        )

        if "gridsyn" in selected_methods:
            grid_path = write_qasm(gridsyn_qc, circuit_dir / f"{spec.name}_gridsyn_clifford_t.qasm")
            records.append(
                _record(
                    spec.name,
                    "gridsyn",
                    "gridsyn_clifford_t",
                    gridsyn_qc,
                    grid_path,
                    source_stage="original_clifford_rz",
                )
            )

        if "tzap" in selected_methods:
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

        if "pyzx" in selected_methods:
            # This is the legacy pyzx_path.py workflow: PyZX receives only
            # the already synthesized GridSynth Clifford+T circuit. PyZX can
            # re-express phases as RZ, so record that intermediate and
            # normalize it back to Clifford+T before comparing T counts.
            started = time.perf_counter()
            pyzx_from_grid, pyzx_grid_stats = optimize_pyzx(gridsyn_qc)
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
            pyzx_from_grid_ct = synthesize_rz(pyzx_from_grid, error_threshold)
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

        if "t-optimizer" in selected_methods:
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

    manifest = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "t_count_methods_comparison",
        "benchmarks": selected_benchmarks,
        "methods": selected_methods,
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
        "workflow": {
            "tzap": "original Clifford+RZ -> tzap -> GridSynth RZ synthesis -> tzap",
            "pyzx": (
                "original Clifford+RZ -> PyZX -> GridSynth RZ synthesis; "
                "GridSynth Clifford+T -> PyZX -> RZ synthesis"
            ),
            "t_optimizer": "GridSynth Clifford+T -> T-Optimizer IR -> optimized Clifford+T",
        },
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Compare T-count optimizers for NC-Fusion circuits")
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--output", type=Path, default=Path("results/runs/t_count_methods_comparison"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--tzap-bin", default=None, help="tzap executable or command; defaults to TZAP_BIN/tzap")
    parser.add_argument("--t-optimizer-root", type=Path, default=None, help="cloned T-Optimizer repository")
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
        )
    except (ImportError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    _main()
