"""NC-Fusion component ablation from the legacy ``main_alg.py`` flow.

The three configurations are:

1. ``anti_commuting_only``: transform and compress ``no_commute_group``;
2. ``anti_commuting_plus_commuting``: transform and compress ``group``
   without reordering;
3. ``full_ncfusion``: transform ``group + no_commute_group`` and apply the
   group reordering/scheduling pass before compression.

These correspond to the calls around lines 279, 282, and 287 of the original
``main_alg.py``.
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
from .data import existing_qasm_path, reusable_record
from .error_common import compile_method


STATUS = "available"
DEFAULT_VARIANTS = (
    "anti_commuting_only",
    "anti_commuting_plus_commuting",
    "full_ncfusion",
)
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


def _transform_groups(
    groups: list[dict[Any, Any]],
    pauli_list: dict[str, float],
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[list[dict[str, Any]]]]:
    """Apply the same conjugation transformation used by ``main_alg.py``."""

    import copy

    from circuit_generation_greedy import greedy_circuit_generation, new_paulis_transform
    from grouping import permute_keys_after_weight_sort

    new_paulis: list[dict[str, float]] = []
    commute_paulis: list[dict[str, float]] = []
    circuits: list[list[dict[str, Any]]] = []
    for subgroup in groups:
        permuted = permute_keys_after_weight_sort(copy.deepcopy(subgroup))
        transformed, circuit, signs = greedy_circuit_generation(permuted[0])
        new, commute = new_paulis_transform(pauli_list, transformed, permuted[0], signs)
        new_paulis.append(new)
        commute_paulis.append(commute)
        circuits.append(circuit)
    return new_paulis, commute_paulis, circuits


def _group_inputs(
    hamiltonian: Any,
    *,
    budget: int,
    window: int,
    pauli_order_seed: int | None,
) -> tuple[dict[str, Any], int]:
    """Build the three ablation inputs before compression."""

    import numpy as np

    from grouping import grouping
    from reorder import reorder_pauli_groups

    labels = list(hamiltonian.paulis.to_labels())
    coefficients = hamiltonian.coeffs
    identity = "I" * hamiltonian.num_qubits
    filtered = [(label, coefficient) for label, coefficient in zip(labels, coefficients) if label != identity]
    labels = [label for label, _ in filtered]
    coefficients = np.asarray([coefficient for _, coefficient in filtered])
    if pauli_order_seed is not None:
        order = np.random.default_rng(pauli_order_seed).permutation(len(labels))
        labels = [labels[index] for index in order]
        coefficients = coefficients[order]

    pauli_list = {label: coefficients[index].real for index, label in enumerate(labels)}
    group, no_commute_group = grouping(labels, budget, window, use_window=True)
    transformed_group = _transform_groups(group, pauli_list)
    transformed_no_commute = _transform_groups(no_commute_group, pauli_list)
    combined = tuple(
        left + right
        for left, right in zip(transformed_group, transformed_no_commute)
    )
    reordered = reorder_pauli_groups(*combined)
    return {
        "anti_commuting_only": transformed_no_commute,
        "anti_commuting_plus_commuting": transformed_group,
        "full_ncfusion": reordered,
    }, len(labels)


def _circuit_metrics(circuit: Any) -> dict[str, int]:
    def operation(item: Any) -> Any:
        return item.operation if hasattr(item, "operation") else item[0]

    operations = list(circuit.data)
    t_names = {"t", "tdg"}
    t_count = sum(operation(item).name.lower() in t_names for item in operations)
    t_depth = int(circuit.depth(lambda item: operation(item).name.lower() in t_names))
    return {
        "t_count": int(t_count),
        "t_depth": t_depth,
        "clifford_count": len(operations) - int(t_count),
        "gate_count": len(operations),
    }


def run(
    output: Path | str = "micro_artifact/results/runs/components_abalation",
    *,
    benchmarks: list[str] | None = None,
    variants: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    budget: int = 1,
    window: int = 4,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    pauli_order_seed: int | None = None,
) -> dict[str, Any]:
    """Run the three component configurations and write circuit metrics."""

    selected_variants = tuple(variants or methods or DEFAULT_VARIANTS)
    unknown = [variant for variant in selected_variants if variant not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"unknown component-ablation variant(s): {', '.join(unknown)}")
    if budget not in (1, 2):
        raise ValueError("budget must be 1 or 2")
    if window < 1 or error_threshold <= 0 or t_budget < 1 or trotter_steps < 1:
        raise ValueError("window, t_budget, trotter_steps, and error_threshold must be positive")
    if evolution_time == 0:
        raise ValueError("evolution_time must be non-zero")

    import numpy as np

    from ncfusion.legacy import build_hamiltonian
    from ncfusion.spec import find_benchmark
    from compressor import compressor_circuit

    random.seed(seed)
    np.random.seed(seed)
    selected_benchmarks = benchmarks or list(DEFAULT_BENCHMARKS)
    records: list[dict[str, object]] = []

    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        hamiltonian = build_hamiltonian(spec)
        grouped_inputs = None
        pauli_count = int(spec.pauli_terms)
        if any(variant != "full_ncfusion" for variant in selected_variants):
            grouped_inputs, pauli_count = _group_inputs(
                hamiltonian,
                budget=budget,
                window=window,
                pauli_order_seed=pauli_order_seed,
            )

        gridsyn_qc = None
        existing_grid = existing_qasm_path(spec.name, "gridsyn")
        if existing_grid is not None:
            from qiskit import QuantumCircuit

            gridsyn_qc = QuantumCircuit.from_qasm_file(str(existing_grid))
        if gridsyn_qc is None:
            _, gridsyn_qc = compile_method(
                hamiltonian,
                "gridsyn",
                synthesize=True,
                error_threshold=error_threshold,
                t_budget=t_budget,
                gpu=gpu,
                trotter_steps=trotter_steps,
                evolution_time=evolution_time,
                window=None,
                pauli_order_seed=None,
            )
        if gridsyn_qc is None:
            raise RuntimeError("Gridsynth did not return a reference circuit")
        gridsyn_metrics = _circuit_metrics(gridsyn_qc)
        for variant in selected_variants:
            if variant == "full_ncfusion" and budget == 1:
                existing = reusable_record(spec, "ncf-one")
                if existing is not None:
                    record = {
                        key: existing[key]
                        for key in ("t_count", "t_depth", "clifford_count", "gate_count")
                    }
                    record.update(
                        {
                            "benchmark": spec.name,
                            "variant": variant,
                            "budget": budget,
                            "window": window,
                            "pauli_strings": pauli_count,
                            "trotter_steps": trotter_steps,
                            "evolution_time": evolution_time,
                            "rz_gate_count": existing.get("original_rz_gate_count"),
                            "ncf_unitaries_generated": existing.get("ncf_unitaries_generated"),
                            "original_rz_gate_count": existing.get("original_rz_gate_count"),
                            "runtime_seconds": existing.get("compilation_time_seconds"),
                            "compilation_time_seconds": existing.get("compilation_time_seconds"),
                            "data_source": "single_qubit_result",
                            "qasm_path": existing.get("qasm_path"),
                            "gridsyn_t_count": gridsyn_metrics["t_count"],
                            "gridsyn_t_depth": gridsyn_metrics["t_depth"],
                            "gridsyn_clifford_count": gridsyn_metrics["clifford_count"],
                        }
                    )
                    record.update(
                        {
                            "t_count_reduction_percent": (
                                100.0 * (gridsyn_metrics["t_count"] - record["t_count"])
                                / gridsyn_metrics["t_count"] if gridsyn_metrics["t_count"] else 0.0
                            ),
                            "t_depth_reduction_percent": (
                                100.0 * (gridsyn_metrics["t_depth"] - record["t_depth"])
                                / gridsyn_metrics["t_depth"] if gridsyn_metrics["t_depth"] else 0.0
                            ),
                            "clifford_reduction_percent": (
                                100.0 * (gridsyn_metrics["clifford_count"] - record["clifford_count"])
                                / gridsyn_metrics["clifford_count"] if gridsyn_metrics["clifford_count"] else 0.0
                            ),
                        }
                    )
                    records.append(record)
                    continue

            start = time.perf_counter()
            if grouped_inputs is None:
                raise RuntimeError("ablation inputs were not generated")
            new_paulis, commute_paulis, circuits = grouped_inputs[variant]
            rz_qc, clifford_t_qc = compressor_circuit(
                new_paulis,
                commute_paulis,
                circuits,
                error_threshold,
                budget,
                hamiltonian.num_qubits,
                gpu=gpu,
                num_paulis=pauli_count,
                fix_error_threshold=0,
                trotter_steps=trotter_steps,
                evolution_time=evolution_time,
                synthesize=True,
                benchmark=None,
                t_budget=t_budget,
            )
            if clifford_t_qc is None:
                raise RuntimeError(f"component variant {variant} did not synthesize a circuit")
            record = _circuit_metrics(clifford_t_qc)
            record.update(
                {
                    "benchmark": spec.name,
                    "variant": variant,
                    "budget": budget,
                    "window": window,
                    "pauli_strings": pauli_count,
                    "trotter_steps": trotter_steps,
                    "evolution_time": evolution_time,
                    "rz_gate_count": sum(
                        (item.operation if hasattr(item, "operation") else item[0]).name.lower() == "rz"
                        for item in rz_qc.data
                    ),
                    "runtime_seconds": round(time.perf_counter() - start, 4),
                    "compilation_time_seconds": round(time.perf_counter() - start, 4),
                    "data_source": "generated",
                    "gridsyn_t_count": gridsyn_metrics["t_count"],
                    "gridsyn_t_depth": gridsyn_metrics["t_depth"],
                    "gridsyn_clifford_count": gridsyn_metrics["clifford_count"],
                    "t_count_reduction_percent": (
                        100.0
                        * (gridsyn_metrics["t_count"] - record["t_count"])
                        / gridsyn_metrics["t_count"]
                        if gridsyn_metrics["t_count"]
                        else 0.0
                    ),
                    "t_depth_reduction_percent": (
                        100.0
                        * (gridsyn_metrics["t_depth"] - record["t_depth"])
                        / gridsyn_metrics["t_depth"]
                        if gridsyn_metrics["t_depth"]
                        else 0.0
                    ),
                    "clifford_reduction_percent": (
                        100.0
                        * (gridsyn_metrics["clifford_count"] - record["clifford_count"])
                        / gridsyn_metrics["clifford_count"]
                        if gridsyn_metrics["clifford_count"]
                        else 0.0
                    ),
                }
            )
            records.append(record)

    output_path = Path(output)
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "components_abalation",
        "paper_section": "5.6.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "variants": selected_variants,
        "variant_definitions": {
            "anti_commuting_only": "no_commute_new_paulis; main_alg.py line 279",
            "anti_commuting_plus_commuting": "new_paulis without reordering; main_alg.py line 282",
            "full_ncfusion": "reordered new_paulis plus no_commute_new_paulis; main_alg.py line 287",
        },
        "record_count": len(records),
        "paper_dependencies": dependency_status(),
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the NC-Fusion component ablation")
    add_cli_arguments(parser)
    parser.add_argument("--variant", action="append", dest="variants", choices=DEFAULT_VARIANTS)
    parser.add_argument("--budget", type=int, choices=(1, 2), default=1)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--pauli-order-seed", type=int)
    args = parser.parse_args(argv)
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            variants=args.variants,
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
