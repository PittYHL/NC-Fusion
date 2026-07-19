"""Application-level fidelity evaluation from the paper's Section 5.7.2."""

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
from .error_common import compile_method, validate_methods


STATUS = "available"
DEFAULT_LOGICAL_ERRORS = (1e-6, 1e-7)


def run(
    output: Path | str = "results/runs/application_level_fidelity",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    trotter_steps: Iterable[int] = (1, 5, 10, 20),
    logical_errors: Iterable[float] = DEFAULT_LOGICAL_ERRORS,
    evolution_time: float = 1.0,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    window: int | None = 4,
) -> dict[str, Any]:
    """Measure ideal and noisy application fidelity.

    This passes the synthesized ``clifford_t_qc`` to
    ``density_matrix_error_from_hamiltonian``, matching ``main_alg.py``.
    """

    from ncfusion.legacy import build_hamiltonian
    from ncfusion.spec import find_benchmark, find_experiment
    from error_evaluation import density_matrix_error_from_hamiltonian

    steps = tuple(int(value) for value in trotter_steps)
    rates = tuple(float(value) for value in logical_errors)
    if not steps or any(value < 1 for value in steps):
        raise ValueError("trotter_steps must contain positive integers")
    if any(value <= 0 for value in rates):
        raise ValueError("logical_errors must be positive")
    if evolution_time == 0:
        raise ValueError("evolution_time must be non-zero")
    if error_threshold <= 0 or t_budget < 1:
        raise ValueError("error_threshold must be positive and t_budget must be positive")

    selected_methods = validate_methods(methods)
    selected_benchmarks = benchmarks or list(find_experiment("error-evaluation").benchmarks)
    output_path = Path(output)
    records: list[dict[str, object]] = []

    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        hamiltonian = build_hamiltonian(spec)
        for method in selected_methods:
            for step_count in steps:
                start = time.perf_counter()
                _, clifford_t_qc = compile_method(
                    hamiltonian,
                    method,
                    synthesize=True,
                    error_threshold=error_threshold,
                    t_budget=t_budget,
                    gpu=gpu,
                    trotter_steps=step_count,
                    evolution_time=evolution_time,
                    window=window,
                    pauli_order_seed=seed if method == "ncf-one" else None,
                )
                if clifford_t_qc is None:
                    raise RuntimeError("compiled method did not return clifford_t_qc")
                fidelity, noisy_fidelity = density_matrix_error_from_hamiltonian(
                    hamiltonian,
                    clifford_t_qc,
                    evolution_time,
                    gpu=bool(gpu),
                    logical_errors=list(rates),
                )
                elapsed = round(time.perf_counter() - start, 4)
                if noisy_fidelity:
                    for logical_error, value in noisy_fidelity.items():
                        records.append(
                            {
                                "benchmark": spec.name,
                                "method": method,
                                "trotter_steps": step_count,
                                "evolution_time": evolution_time,
                                "logical_error_rate": float(logical_error),
                                "ideal_fidelity": float(fidelity) if fidelity is not None else None,
                                "noisy_fidelity": float(value),
                                "clifford_t_gate_count": len(clifford_t_qc.data),
                                "runtime_seconds": elapsed,
                            }
                        )
                else:
                    records.append(
                        {
                            "benchmark": spec.name,
                            "method": method,
                            "trotter_steps": step_count,
                            "evolution_time": evolution_time,
                            "logical_error_rate": None,
                            "ideal_fidelity": float(fidelity) if fidelity is not None else None,
                            "noisy_fidelity": None,
                            "clifford_t_gate_count": len(clifford_t_qc.data),
                            "runtime_seconds": elapsed,
                        }
                    )

    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "application_level_fidelity",
        "paper_section": "5.7.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "methods": selected_methods,
        "trotter_steps": steps,
        "logical_error_rates": rates,
        "error_function": "legacy.error_evaluation.density_matrix_error_from_hamiltonian",
        "circuit_argument": "clifford_t_qc",
        "record_count": len(records),
        "paper_dependencies": dependency_status(),
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the NC-Fusion application-fidelity evaluation")
    add_cli_arguments(parser)
    parser.add_argument("--trotter-steps", action="append", type=int, default=None)
    parser.add_argument("--logical-error", action="append", dest="logical_errors", type=float, default=None)
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
            trotter_steps=args.trotter_steps or (1, 5, 10, 20),
            logical_errors=args.logical_errors or DEFAULT_LOGICAL_ERRORS,
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
