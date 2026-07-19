"""Analytical T-count estimation from unsynthesized NC-Fusion circuits.

Section 5.4 of the paper models the synthesis cost of each generated unitary
instead of running a synthesizer for every precision value:

* single-qubit ``RZ``/``U3``: ``3 * log2(1 / epsilon)`` T gates;
* two-qubit unitary: ``15 * log2.76(1 / epsilon)`` T gates.

Section 5.1.4 scales the NC-Fusion precision by ``N_PS / N_U`` so that the
total synthesis error is comparable with the baseline.  This module obtains
``N_U`` by calling :func:`ncfusion.NC_Fusion` with ``synthesize=False`` and
counting the non-Clifford rotation units in its returned circuit.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.runner import dependency_status

from .common import add_cli_arguments


STATUS = "available"
DEFAULT_EPSILONS = tuple(10.0 ** (-power) for power in range(1, 10))
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


def _is_clifford_rz(angle: object) -> bool:
    """Return whether an RZ angle is a multiple of pi/2."""

    try:
        ratio = float(angle) / (math.pi / 2)
    except (TypeError, ValueError):
        return False
    return math.isclose(ratio, round(ratio), abs_tol=1e-8)


def count_synthesis_unitaries(circuit: Any) -> int:
    """Count non-Clifford rotation units in an unsynthesized circuit.

    ``U3``/``U`` operations are counted as one unit. ``RZ`` operations whose
    angle is a Clifford angle are excluded because their estimated T cost is
    zero. The helper accepts Qiskit's current ``CircuitInstruction`` objects
    and older tuple-style circuit data.
    """

    count = 0
    for item in circuit.data:
        operation = item.operation if hasattr(item, "operation") else item[0]
        name = operation.name.lower()
        if name in {"u", "u3"}:
            count += 1
        elif name == "rz" and not _is_clifford_rz(operation.params[0]):
            count += 1
    return count


def estimate_t_gates_per_unitary(
    epsilon: float,
    *,
    budget: int = 1,
    precision_scale: float = 1.0,
) -> float:
    """Estimate T gates for one generated unitary using the paper's model.

    ``precision_scale`` is normally ``N_PS / N_U`` for NC-Fusion. The
    returned value is deliberately a float because the paper uses an
    asymptotic estimate rather than rounding to an integer T count.
    """

    if budget not in (1, 2):
        raise ValueError("budget must be 1 or 2")
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if precision_scale <= 0:
        raise ValueError("precision_scale must be positive")

    effective_epsilon = epsilon * precision_scale
    if effective_epsilon >= 1:
        return 0.0
    if budget == 1:
        return 3.0 * math.log2(1.0 / effective_epsilon)
    return 15.0 * math.log(1.0 / effective_epsilon, 2.76)


def estimate_from_unitary_count(
    unitary_count: int,
    pauli_string_count: int,
    epsilon: float,
    *,
    budget: int = 1,
    trotter_steps: int = 1,
) -> dict[str, float | int]:
    """Return baseline and NC-Fusion estimates from ``N_U``.

    The baseline is Gridsynth-style synthesis of one RZ per Pauli string. The
    NC-Fusion precision is scaled by ``N_PS / N_U`` as specified in the paper.
    Counts include all requested Trotter repetitions.
    """

    if unitary_count < 1:
        raise ValueError("unitary_count must be positive")
    if pauli_string_count < 1:
        raise ValueError("pauli_string_count must be positive")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be positive")

    total_unitaries = unitary_count * trotter_steps
    total_pauli_strings = pauli_string_count * trotter_steps
    precision_scale = total_pauli_strings / total_unitaries
    ncf_per_unitary = estimate_t_gates_per_unitary(
        epsilon,
        budget=budget,
        precision_scale=precision_scale,
    )
    baseline_per_unitary = estimate_t_gates_per_unitary(epsilon, budget=1)
    ncf_t_count = total_unitaries * ncf_per_unitary
    baseline_t_count = total_pauli_strings * baseline_per_unitary
    reduction = 100.0 * (baseline_t_count - ncf_t_count) / baseline_t_count
    return {
        "ncf_unitaries": total_unitaries,
        "baseline_unitaries": total_pauli_strings,
        "ncf_precision": epsilon * precision_scale,
        "ncf_t_gates_per_unitary": ncf_per_unitary,
        "baseline_t_gates_per_unitary": baseline_per_unitary,
        "ncf_estimated_t_count": ncf_t_count,
        "baseline_estimated_t_count": baseline_t_count,
        "t_count_reduction_percent": reduction,
    }


def _non_identity_pauli_count(hamiltonian: Any) -> int:
    labels = hamiltonian.paulis.to_labels()
    identity = "I" * hamiltonian.num_qubits
    return sum(label != identity for label in labels)


def _budgets(budget: int, methods: list[str] | None) -> tuple[int, ...]:
    if methods is None:
        if budget not in (1, 2):
            raise ValueError("budget must be 1 or 2")
        return (budget,)
    mapping = {"ncf-one": 1, "ncf-two": 2}
    unknown = [method for method in methods if method not in mapping]
    if unknown:
        raise ValueError("analytical estimation methods must be ncf-one or ncf-two")
    return tuple(dict.fromkeys(mapping[method] for method in methods))


def run(
    output: Path | str = "results/runs/analytical_estimation",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    budget: int = 1,
    window: int | None = None,
    epsilons: Iterable[float] = DEFAULT_EPSILONS,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    fix_error_threshold: bool = False,
    pauli_order_seed: int | None = None,
) -> dict[str, Any]:
    """Generate analytical T-count estimates for selected benchmarks."""

    import numpy as np

    from ncfusion import NC_Fusion
    from ncfusion.legacy import build_hamiltonian
    from ncfusion.spec import find_benchmark

    epsilon_values = tuple(float(value) for value in epsilons)
    if not epsilon_values:
        raise ValueError("at least one epsilon is required")
    for epsilon in epsilon_values:
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("every epsilon must be between 0 and 1")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be positive")
    if evolution_time == 0:
        raise ValueError("evolution_time must be non-zero")

    np.random.seed(seed)
    selected_benchmarks = benchmarks or list(DEFAULT_BENCHMARKS)
    selected_budgets = _budgets(budget, methods)
    records: list[dict[str, object]] = []

    for selected_budget in selected_budgets:
        for benchmark_name in selected_benchmarks:
            spec = find_benchmark(benchmark_name)
            hamiltonian = build_hamiltonian(spec)
            compiled_qc, _ = NC_Fusion(
                hamiltonian,
                budget=selected_budget,
                window=window,
                error_threshold=error_threshold,
                t_budget=t_budget,
                gpu=gpu,
                trotter_steps=trotter_steps,
                evolution_time=evolution_time,
                synthesize=False,
                fix_error_threshold=fix_error_threshold,
                pauli_order_seed=pauli_order_seed,
            )
            unitary_count = count_synthesis_unitaries(compiled_qc)
            pauli_string_count = _non_identity_pauli_count(hamiltonian)
            for epsilon in epsilon_values:
                estimate = estimate_from_unitary_count(
                    unitary_count,
                    pauli_string_count * trotter_steps,
                    epsilon,
                    budget=selected_budget,
                    trotter_steps=1,
                )
                estimate.update(
                    {
                        "benchmark": spec.name,
                        "budget": selected_budget,
                        "epsilon": epsilon,
                        "pauli_strings": pauli_string_count,
                        "compiled_gate_count": len(compiled_qc.data),
                        "window": window if window is not None else (4 if selected_budget == 1 else 128),
                        "trotter_steps": trotter_steps,
                        "evolution_time": evolution_time,
                    }
                )
                records.append(estimate)

    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "analytical_estimation",
        "paper_section": "5.4",
        "created_by": "micro_artifact.analytical_estimation",
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "budgets": selected_budgets,
        "epsilons": epsilon_values,
        "formula_single_qubit": "3 * log2(1 / epsilon)",
        "formula_two_qubit": "15 * log_base_2.76(1 / epsilon)",
        "precision_rule": "epsilon_ncf = epsilon * N_PS / N_U",
        "record_count": len(records),
        "paper_dependencies": dependency_status(),
    }
    output_path = Path(output)
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run NC-Fusion analytical T-count estimation")
    add_cli_arguments(parser)
    parser.add_argument("--budget", type=int, choices=(1, 2), default=1)
    parser.add_argument("--window", type=int)
    parser.add_argument("--epsilon", action="append", dest="epsilons", type=float)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    parser.add_argument("--t-budget", type=int, default=60)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--evolution-time", type=float, default=1.0)
    parser.add_argument("--fix-error-threshold", action="store_true")
    parser.add_argument("--pauli-order-seed", type=int)
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
            epsilons=args.epsilons or DEFAULT_EPSILONS,
            error_threshold=args.error_threshold,
            t_budget=args.t_budget,
            trotter_steps=args.trotter_steps,
            evolution_time=args.evolution_time,
            fix_error_threshold=args.fix_error_threshold,
            pauli_order_seed=args.pauli_order_seed,
        )
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    main()
