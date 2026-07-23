"""NC-Fusion synthesis-precision ablation.

Section 5.6.2 compares NC-Fusion with and without the proportional precision
scaling described in Section 5.1.4. In the cleaned API this is exactly the
``fix_error_threshold`` switch:

* ``False`` / ``0``: distribute the total error threshold across rotations;
* ``True`` / ``1``: use the same error threshold independently per rotation.

Both variants are compiled from the same Hamiltonian with the same grouping
seed, then their synthesized circuit metrics are written side by side.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time
from typing import Any

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.runner import dependency_status

from .common import add_cli_arguments
from .data import reusable_record


STATUS = "available"
DEFAULT_BENCHMARKS = (
    "LiH",
    "H2O",
    "N2",
    "Ising-2D-30",
    "Ising-2D-60",
    "Ising-3D-30",
    "Ising-3D-60",
    "Heisenberg-2D-30",
    "Heisenberg-2D-60",
    "Heisenberg-3D-30",
    "Heisenberg-3D-60",
)


def _metrics(circuit: Any) -> dict[str, int]:
    def operation(item: Any) -> Any:
        return item.operation if hasattr(item, "operation") else item[0]

    operations = list(circuit.data)
    t_names = {"t", "tdg"}
    t_count = sum(operation(item).name.lower() in t_names for item in operations)
    t_depth = int(
        circuit.depth(lambda item: operation(item).name.lower() in t_names)
    )
    return {
        "t_count": int(t_count),
        "t_depth": t_depth,
        "clifford_count": len(operations) - int(t_count),
        "gate_count": len(operations),
    }


def _seed_randomness(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def run(
    output: Path | str = "micro_artifact/results/runs/precision_abalation",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    budget: int = 1,
    window: int | None = None,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    pauli_order_seed: int | None = 0,
) -> dict[str, Any]:
    """Compare synthesized NC-Fusion circuits for both threshold policies."""

    if methods:
        accepted = {"ncf-one": 1, "ncf-two": 2}
        unknown = [method for method in methods if method not in accepted]
        if unknown:
            raise ValueError("methods must be ncf-one or ncf-two")
        budgets = tuple(dict.fromkeys(accepted[method] for method in methods))
    else:
        budgets = (budget,)
    if any(item not in (1, 2) for item in budgets):
        raise ValueError("budget must be 1 or 2")
    if error_threshold <= 0:
        raise ValueError("error_threshold must be positive")
    if t_budget < 1 or trotter_steps < 1:
        raise ValueError("t_budget and trotter_steps must be positive")

    from ncfusion import NC_Fusion
    from ncfusion.legacy import build_hamiltonian
    from ncfusion.spec import find_benchmark

    selected_benchmarks = benchmarks or list(DEFAULT_BENCHMARKS)
    variants = ((0, False, "scaled_total_error"), (1, True, "fixed_per_unitary_error"))
    records: list[dict[str, object]] = []

    for selected_budget in budgets:
        for benchmark_name in selected_benchmarks:
            spec = find_benchmark(benchmark_name)
            hamiltonian = build_hamiltonian(spec)
            for fix_value, fix_error_threshold, variant in variants:
                if selected_budget == 1 and not fix_error_threshold:
                    existing = reusable_record(spec, "ncf-one")
                    if existing is not None:
                        record = {
                            key: existing[key]
                            for key in ("t_count", "t_depth", "clifford_count", "gate_count")
                        }
                        record.update(
                            {
                                "benchmark": spec.name,
                                "budget": selected_budget,
                                "variant": variant,
                                "fix_error_threshold": fix_value,
                                "error_threshold": error_threshold,
                                "window": window if window is not None else 4,
                                "trotter_steps": trotter_steps,
                                "evolution_time": evolution_time,
                                "runtime_seconds": existing.get("compilation_time_seconds"),
                                "compilation_time_seconds": existing.get("compilation_time_seconds"),
                                "data_source": "single_qubit_result",
                                "qasm_path": existing.get("qasm_path"),
                            }
                        )
                        records.append(record)
                        continue

                # Reset the grouping order before each arm so this comparison
                # isolates threshold handling rather than heuristic ordering.
                _seed_randomness(seed)
                start = time.perf_counter()
                _, clifford_t_qc = NC_Fusion(
                    hamiltonian,
                    budget=selected_budget,
                    window=window,
                    error_threshold=error_threshold,
                    t_budget=t_budget,
                    gpu=gpu,
                    trotter_steps=trotter_steps,
                    evolution_time=evolution_time,
                    synthesize=True,
                    fix_error_threshold=fix_error_threshold,
                    pauli_order_seed=pauli_order_seed,
                )
                if clifford_t_qc is None:
                    raise RuntimeError("NC_Fusion did not return a synthesized circuit")
                record = _metrics(clifford_t_qc)
                record.update(
                    {
                        "benchmark": spec.name,
                        "budget": selected_budget,
                        "variant": variant,
                        "fix_error_threshold": fix_value,
                        "error_threshold": error_threshold,
                        "window": window if window is not None else (4 if selected_budget == 1 else 128),
                        "trotter_steps": trotter_steps,
                        "evolution_time": evolution_time,
                        "runtime_seconds": round(time.perf_counter() - start, 4),
                        "compilation_time_seconds": round(time.perf_counter() - start, 4),
                        "data_source": "generated",
                    }
                )
                records.append(record)

    output_path = Path(output)
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "precision_abalation",
        "paper_section": "5.6.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "budgets": budgets,
        "variants": [
            {"fix_error_threshold": value, "name": name}
            for value, _, name in variants
        ],
        "record_count": len(records),
        "paper_dependencies": dependency_status(),
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the NC-Fusion precision ablation")
    add_cli_arguments(parser)
    parser.add_argument("--budget", type=int, choices=(1, 2), default=1)
    parser.add_argument("--window", type=int)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--pauli-order-seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            budget=args.budget,
            window=args.window,
            error_threshold=args.error_threshold,
            t_budget=args.t_budget,
            trotter_steps=args.trotter_steps,
            evolution_time=args.evolution_time,
            pauli_order_seed=args.pauli_order_seed,
        )
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    main()
