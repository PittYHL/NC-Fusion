"""Trotter operator-norm error evaluation from the paper's Section 5.7.1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.runner import dependency_status

from .common import add_cli_arguments
from .error_common import compile_method, load_existing_method_circuits, validate_methods


STATUS = "available"
DEFAULT_TROTTER_STEPS = (1, 5, 10, 20)


def run(
    output: Path | str = "micro_artifact/results/runs/trotter_error",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    trotter_steps: Iterable[int] = DEFAULT_TROTTER_STEPS,
    evolution_time: float = 1.0,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    window: int | None = 4,
) -> dict[str, Any]:
    """Measure ``||exp(-iHt) - U_rz||_2`` for Gridsynth and NC-Fusion.

    This intentionally passes the unsynthesized ``rz_qc`` to
    ``trotter_operator_norm_error``, matching ``main_alg.py``.
    """

    from ncfusion.legacy import build_hamiltonian
    from ncfusion.spec import find_benchmark, find_experiment
    from error_evaluation import trotter_operator_norm_error

    steps = tuple(int(value) for value in trotter_steps)
    if not steps or any(value < 1 for value in steps):
        raise ValueError("trotter_steps must contain positive integers")
    if evolution_time == 0:
        raise ValueError("evolution_time must be non-zero")
    if error_threshold <= 0 or t_budget < 1:
        raise ValueError("error_threshold must be positive and t_budget must be positive")

    selected_methods = validate_methods(methods)
    selected_benchmarks = benchmarks or list(find_experiment("error-evaluation").benchmarks)
    output_path = Path(output)
    cache_path = output_path / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        hamiltonian = build_hamiltonian(spec)
        for method in selected_methods:
            for step_count in steps:
                start = time.perf_counter()
                existing = (
                    load_existing_method_circuits(spec.name, method)
                    if step_count == 1 else None
                )
                if existing is not None:
                    rz_qc, _, compilation_time = existing
                else:
                    compile_started = time.perf_counter()
                    rz_qc, _ = compile_method(
                        hamiltonian,
                        method,
                        synthesize=False,
                        error_threshold=error_threshold,
                        t_budget=t_budget,
                        gpu=gpu,
                        trotter_steps=step_count,
                        evolution_time=evolution_time,
                        window=window,
                        pauli_order_seed=seed if method == "ncf-one" else None,
                    )
                    compilation_time = time.perf_counter() - compile_started
                error = trotter_operator_norm_error(
                    hamiltonian,
                    rz_qc,
                    evolution_time,
                    cache_file=cache_path / f"{spec.name}_U_exact.npy",
                )
                rz_gate_count = sum(
                    (item.operation if hasattr(item, "operation") else item[0]).name.lower() == "rz"
                    for item in rz_qc.data
                )
                records.append(
                    {
                        "benchmark": spec.name,
                        "method": method,
                        "trotter_steps": step_count,
                        "evolution_time": evolution_time,
                        "operator_norm_error": float(error),
                        "rz_gate_count": int(rz_gate_count),
                        "runtime_seconds": round(time.perf_counter() - start, 4),
                        "compilation_time_seconds": round(compilation_time, 4) if compilation_time is not None else None,
                        "data_source": "existing_qasm" if existing is not None else "generated",
                    }
                )

    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "trotter_error",
        "paper_section": "5.7.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "methods": selected_methods,
        "trotter_steps": steps,
        "error_function": "legacy.error_evaluation.trotter_operator_norm_error",
        "circuit_argument": "rz_qc",
        "record_count": len(records),
        "paper_dependencies": dependency_status(),
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the NC-Fusion Trotter-error evaluation")
    add_cli_arguments(parser)
    parser.add_argument("--trotter-steps", action="append", type=int, default=None)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--window", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            trotter_steps=args.trotter_steps or DEFAULT_TROTTER_STEPS,
            evolution_time=args.evolution_time,
            error_threshold=args.error_threshold,
            t_budget=args.t_budget,
            window=args.window,
        )
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    main()
