"""Application-level fidelity evaluation from the paper's Section 5.7.2."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.runner import dependency_status

from .common import add_cli_arguments
from .error_common import (
    compile_method,
    load_existing_method_circuits,
    load_generated_method_circuits,
    save_generated_method_circuits,
    validate_methods,
)


STATUS = "available"
DEFAULT_LOGICAL_ERRORS = (1e-6, 1e-7)
DEFAULT_TROTTER_STEPS = (1, 5, 10, 20)


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _record_key(record: dict[str, object]) -> tuple[str, str, str, str]:
    def value(field: str) -> str:
        item = record.get(field, "")
        return "" if item in (None, "") else str(item)

    return (
        value("benchmark"),
        value("method"),
        value("trotter_steps"),
        value("logical_error_rate"),
    )


def _merge_record(
    records: list[dict[str, object]],
    update: dict[str, object],
) -> list[dict[str, object]]:
    key = _record_key(update)
    for index, record in enumerate(records):
        if _record_key(record) == key:
            records[index] = update
            return records
    records.append(update)
    return records


def run(
    output: Path | str = "micro_artifact/results/runs/application_level_fidelity",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 1,
    trotter_steps: Iterable[int] = DEFAULT_TROTTER_STEPS,
    logical_errors: Iterable[float] = DEFAULT_LOGICAL_ERRORS,
    evolution_time: float = 1.0,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    window: int | None = 4,
    trotter_circuit_dir: Path | str = "micro_artifact/results/runs/trotter_error",
    source: str = "existing",
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
    if gpu not in (0, 1):
        raise ValueError("gpu must be 0 (CPU) or 1 (GPU)")
    if evolution_time != 1.0:
        raise ValueError(
            "the application-fidelity evaluation requires evolution_time=1.0"
        )
    if error_threshold <= 0 or t_budget < 1:
        raise ValueError("error_threshold must be positive and t_budget must be positive")
    if source not in {"existing", "generate"}:
        raise ValueError("source must be existing or generate")

    selected_methods = validate_methods(methods)
    selected_benchmarks = benchmarks or list(find_experiment("error-evaluation").benchmarks)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    records = _read_records(output_path / "metrics.csv")
    if source == "existing" and records:
        return {
            "manifest": {
                "evaluation": "application_level_fidelity",
                "source_mode": "existing",
                "record_count": len(records),
                "status": "read_existing",
            },
            "records": records,
        }

    def add_record(record: dict[str, object]) -> None:
        nonlocal records
        records = _merge_record(records, record)
        write_records_csv(output_path / "metrics.csv", records)

    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        hamiltonian = build_hamiltonian(spec)
        for method in selected_methods:
            for step_count in steps:
                start = time.perf_counter()
                if source == "existing":
                    trotter_generated = load_generated_method_circuits(
                        trotter_circuit_dir, spec.name, method, step_count
                    )
                    existing = trotter_generated
                    if existing is None and step_count == 1:
                        existing = load_existing_method_circuits(spec.name, method)
                else:
                    trotter_generated = None
                    existing = None
                if existing is not None:
                    _, clifford_t_qc, compilation_time = existing
                    data_source = (
                        "trotter_error_generated_qasm"
                        if trotter_generated is not None
                        else "existing_qasm"
                    )
                else:
                    if source == "existing":
                        raise FileNotFoundError(
                            f"No stored circuit for {spec.name}, {method}, Trotter step "
                            f"{step_count}; run trotter_error with --source generate first "
                            "or use --source generate here."
                        )
                    compile_started = time.perf_counter()
                    rz_qc, clifford_t_qc = compile_method(
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
                    compilation_time = time.perf_counter() - compile_started
                    if clifford_t_qc is None:
                        raise RuntimeError("compiled method did not return clifford_t_qc")
                    save_generated_method_circuits(
                        output_path,
                        spec.name,
                        method,
                        step_count,
                        rz_qc,
                        clifford_t_qc,
                    )
                    data_source = "generated_and_saved"
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
                        add_record(
                            {
                                "benchmark": spec.name,
                                "method": method,
                                "trotter_steps": step_count,
                                "evolution_time": evolution_time,
                                "logical_error_rate": float(logical_error),
                                "ideal_fidelity": float(fidelity) if fidelity is not None else None,
                                "noisy_fidelity": float(value),
                                "clifford_t_gate_count": len(clifford_t_qc.data),
                                "gpu": int(gpu),
                                "fidelity_function": "density_matrix_error_from_hamiltonian",
                                "runtime_seconds": elapsed,
                                "compilation_time_seconds": round(compilation_time, 4) if compilation_time is not None else None,
                                "data_source": data_source,
                            }
                        )
                else:
                    add_record(
                        {
                            "benchmark": spec.name,
                            "method": method,
                            "trotter_steps": step_count,
                            "evolution_time": evolution_time,
                            "logical_error_rate": None,
                            "ideal_fidelity": float(fidelity) if fidelity is not None else None,
                            "noisy_fidelity": None,
                            "clifford_t_gate_count": len(clifford_t_qc.data),
                            "gpu": int(gpu),
                            "fidelity_function": "density_matrix_error_from_hamiltonian",
                            "runtime_seconds": elapsed,
                            "compilation_time_seconds": round(compilation_time, 4) if compilation_time is not None else None,
                            "data_source": data_source,
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
        "source_mode": source,
        "evolution_time": 1.0,
        "gpu": int(gpu),
        "logical_error_rates": rates,
        "error_function": "legacy.error_evaluation.density_matrix_error_from_hamiltonian",
        "circuit_argument": "clifford_t_qc",
        "trotter_circuit_dir": str(trotter_circuit_dir),
        "generated_circuit_root": str(output_path / "circuits"),
        "record_count": len(records),
        "csv_merge_policy": "replace matching benchmark/method/trotter_steps/logical_error_rate rows and append new configurations",
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
    parser.set_defaults(gpu=1)
    parser.add_argument(
        "--trotter-circuit-dir",
        type=Path,
        default=Path("micro_artifact/results/runs/trotter_error"),
    )
    parser.add_argument(
        "--source",
        choices=("existing", "generate"),
        default="existing",
        help="read stored circuits by default; use generate to regenerate them",
    )
    args = parser.parse_args(argv)
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            trotter_steps=args.trotter_steps or DEFAULT_TROTTER_STEPS,
            logical_errors=args.logical_errors or DEFAULT_LOGICAL_ERRORS,
            evolution_time=args.evolution_time,
            error_threshold=args.error_threshold,
            t_budget=args.t_budget,
            window=args.window,
            trotter_circuit_dir=args.trotter_circuit_dir,
            source=args.source,
        )
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    main()
