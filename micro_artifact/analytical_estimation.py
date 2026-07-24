"""Analytical T-count estimation from unsynthesized NC-Fusion circuits.

Section 5.4 of the paper models the synthesis cost of each generated unitary
instead of running a synthesizer for every precision value:

* single-qubit ``RZ``/``U3``: ``3 * log2(1 / epsilon)`` T gates;
* two-qubit unitary: ``15 * log2.76(1 / epsilon)`` T gates.

Section 5.1.4 scales the NC-Fusion precision by ``N_RZ / N_U`` so that the
total synthesis error is comparable with the original RZ baseline. This
module consumes ``N_U`` and ``N_RZ`` from the producer datasets. Its artifact
output contains average reductions plus figures for estimated T gates and
reductions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from ncfusion.metrics import merge_records, read_records_csv, write_json, write_records_csv
from ncfusion.runner import dependency_status

from .common import add_cli_arguments
from .data import PRODUCER_OUTPUTS, producer_record, read_records
from ncfusion.spec import find_experiment


STATUS = "available"
DEFAULT_EPSILONS = tuple(10.0 ** (-power) for power in range(1, 10))
DEFAULT_BENCHMARKS = tuple(find_experiment("table4").benchmarks) + ("H2S", "CO2")
COMMON_COMPARISON_BENCHMARKS = DEFAULT_BENCHMARKS


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
    budget: int | None = None,
    precision_scale: float = 1.0,
) -> float:
    """Estimate T gates for one generated unitary using the paper's model.

    ``precision_scale`` is normally ``N_RZ / N_U`` for NC-Fusion. The
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
    ncf_rz_generated: int = 0,
    trotter_steps: int = 1,
) -> dict[str, float | int]:
    """Return baseline and NC-Fusion estimates from ``N_U``.

    The baseline is Gridsynth-style synthesis of the recorded original RZ
    circuit. The NC-Fusion precision is scaled by ``N_RZ / N_U``.
    Counts include all requested Trotter repetitions.
    """

    if unitary_count < 1:
        raise ValueError("unitary_count must be positive")
    if pauli_string_count < 1:
        raise ValueError("pauli_string_count must be positive")
    if ncf_rz_generated < 0:
        raise ValueError("ncf_rz_generated must be non-negative")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be positive")

    total_unitaries = unitary_count * trotter_steps
    total_pauli_strings = pauli_string_count * trotter_steps
    # The separated commuting-RZ field belongs only to the two-qubit model.
    # For single-qubit NC-Fusion, N_U already is the original combined
    # compressor rotation count.
    total_ncf_rz = ncf_rz_generated * trotter_steps if budget == 2 else 0
    total_ncf_rotations = total_unitaries + total_ncf_rz
    precision_scale = total_pauli_strings / total_ncf_rotations
    scaled_precision = epsilon * precision_scale
    if budget == 2:
        ncf_t_count = (
            total_unitaries
            * estimate_t_gates_per_unitary(
                epsilon, budget=2, precision_scale=precision_scale
            )
            + total_ncf_rz
            * estimate_t_gates_per_unitary(
                epsilon, budget=1, precision_scale=precision_scale
            )
        )
    else:
        ncf_t_count = total_ncf_rotations * estimate_t_gates_per_unitary(
            epsilon, budget=1, precision_scale=precision_scale
        )
    baseline_per_unitary = estimate_t_gates_per_unitary(epsilon, budget=1)
    baseline_t_count = total_pauli_strings * baseline_per_unitary
    reduction = 100.0 * (baseline_t_count - ncf_t_count) / baseline_t_count
    return {
        "ncf_unitaries": total_unitaries,
        "ncf_rz_generated": total_ncf_rz,
        "baseline_unitaries": total_pauli_strings,
        "ncf_total_rotations": total_ncf_rotations,
        "ncf_precision": scaled_precision,
        "ncf_precision_scale": precision_scale,
        "baseline_t_gates_per_unitary": baseline_per_unitary,
        "ncf_estimated_t_count": ncf_t_count,
        "baseline_estimated_t_count": baseline_t_count,
        "t_count_reduction_percent": reduction,
    }


def _non_identity_pauli_count(hamiltonian: Any) -> int:
    labels = hamiltonian.paulis.to_labels()
    identity = "I" * hamiltonian.num_qubits
    return sum(label != identity for label in labels)


def _budgets(budget: int | None, methods: list[str] | None) -> tuple[int, ...]:
    if methods is None:
        if budget is None:
            return (1, 2)
        if budget not in (1, 2):
            raise ValueError("budget must be 1 or 2")
        return (budget,)
    mapping = {"ncf-one": 1, "ncf-two": 2}
    unknown = [method for method in methods if method not in mapping]
    if unknown:
        raise ValueError("analytical estimation methods must be ncf-one or ncf-two")
    return tuple(dict.fromkeys(mapping[method] for method in methods))


def _ensure_two_qubit_metadata() -> dict[str, object]:
    """Create a fill-in template without synthesis or QASM."""

    output_path = PRODUCER_OUTPUTS["ncf-two"]
    existing_rows = read_records(output_path)

    from ncfusion.spec import find_benchmark

    template_rows: list[dict[str, object]] = []
    for benchmark_name in COMMON_COMPARISON_BENCHMARKS:
        spec = find_benchmark(benchmark_name)
        template_rows.append(
            {
                "t_count": "",
                "t_depth": "",
                "clifford_count": "",
                "gate_count": "",
                "benchmark": spec.name,
                "method": "ncf-two",
                "qubits": spec.qubits,
                "runtime_seconds": "",
                "compilation_time_seconds": "",
                "data_source": "template_to_fill",
                "original_rz_gate_count": "",
                "ncf_unitaries_generated": "",
                "ncf_rz_generated": "",
                "qasm_path": "",
            }
        )

    merged_rows = merge_records(existing_rows, template_rows, ("benchmark", "method"))
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "two_qubit_result",
        "source": "template",
        "template": True,
        "synthesis_run": False,
        "qasm_saved": False,
        "benchmarks": list(COMMON_COMPARISON_BENCHMARKS),
        "methods": ["ncf-two"],
        "record_count": len(merged_rows),
        "required_fields": [
            "ncf_unitaries_generated",
            "ncf_rz_generated",
            "original_rz_gate_count",
        ],
        "note": "Fill the required fields before running analytical_estimation.",
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", merged_rows)
    return {
        "source": "existing_plus_template" if existing_rows else "template",
        "benchmarks": list(COMMON_COMPARISON_BENCHMARKS),
        "record_count": len(merged_rows),
    }


def _average_reduction_table(
    estimates: list[dict[str, object]],
    epsilon_values: tuple[float, ...],
    comparison_benchmarks: tuple[str, ...],
) -> list[dict[str, object]]:
    """Return reductions for the common benchmark set, without T-counts."""

    budgets = tuple(dict.fromkeys(int(item["budget"]) for item in estimates))
    table: list[dict[str, object]] = []
    for budget_value in budgets:
        budget_estimates = [
            item
            for item in estimates
            if int(item["budget"]) == budget_value
            and str(item["benchmark"]) in comparison_benchmarks
        ]
        method = "ncf-one" if budget_value == 1 else "ncf-two"
        benchmark_count = len({str(item["benchmark"]) for item in budget_estimates})
        for epsilon in epsilon_values:
            epsilon_estimates = [
                item
                for item in budget_estimates
                if math.isclose(float(item["epsilon"]), epsilon, rel_tol=0, abs_tol=1e-15)
            ]
            if not epsilon_estimates:
                continue
            average_gridsynth = sum(
                float(item["baseline_estimated_t_count"]) for item in epsilon_estimates
            ) / len(epsilon_estimates)
            average_ncf = sum(
                float(item["ncf_estimated_t_count"]) for item in epsilon_estimates
            ) / len(epsilon_estimates)
            reduction = (
                100.0
                * (average_gridsynth - average_ncf)
                / average_gridsynth
            )
            row: dict[str, object] = {
                "precision": epsilon,
                "method": method,
                "average_t_count_reduction_percent": reduction,
                "benchmark_count": benchmark_count,
            }
            table.append(row)
    return table


def _comparison_values(
    estimates: list[dict[str, object]],
    budget_value: int,
    epsilon: float,
    comparison_benchmarks: tuple[str, ...],
) -> tuple[float, float] | None:
    rows = [
        item
        for item in estimates
        if int(item["budget"]) == budget_value
        and str(item["benchmark"]) in comparison_benchmarks
        and math.isclose(float(item["epsilon"]), epsilon, rel_tol=0, abs_tol=1e-15)
    ]
    if not rows:
        return None
    average_gridsynth = sum(
        float(item["baseline_estimated_t_count"]) for item in rows
    ) / len(rows)
    average_ncf = sum(float(item["ncf_estimated_t_count"]) for item in rows) / len(rows)
    return average_gridsynth, average_ncf


def _write_t_count_plot(
    output_path: Path,
    estimates: list[dict[str, object]],
    epsilon_values: tuple[float, ...],
    comparison_benchmarks: tuple[str, ...],
) -> Path | None:
    """Write GridSynth, single-qubit, and two-qubit estimated T gates."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    budgets = tuple(dict.fromkeys(int(item["budget"]) for item in estimates))
    if not budgets or not comparison_benchmarks:
        return None
    x_values = list(epsilon_values)
    baseline_budget = budgets[0]
    grid_values: list[float] = []
    ncf_values: dict[int, list[float]] = {budget: [] for budget in budgets}
    for epsilon in x_values:
        baseline_pair = _comparison_values(
            estimates, baseline_budget, epsilon, comparison_benchmarks
        )
        if baseline_pair is None:
            return None
        grid_values.append(baseline_pair[0])
        for budget_value in budgets:
            pair = _comparison_values(
                estimates, budget_value, epsilon, comparison_benchmarks
            )
            if pair is None:
                return None
            ncf_values[budget_value].append(pair[1])

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        x_values,
        grid_values,
        color="tab:blue",
        marker="o",
        linewidth=2.0,
        label="GridSynth",
    )
    labels = {1: "NC-Fusion (single-qubit)", 2: "NC-Fusion (two-qubit)"}
    colors = {1: "tab:orange", 2: "tab:green"}
    for budget_value in budgets:
        axis.plot(
            x_values,
            ncf_values[budget_value],
            color=colors[budget_value],
            marker="o",
            linewidth=2.0,
            label=labels[budget_value],
        )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("Synthesis precision ε")
    axis.set_ylabel("Estimated T gates")
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"10$^{{-{round(-math.log10(value))}}}$" for value in x_values])
    axis.grid(True, which="both", linestyle=":", alpha=0.45)
    axis.legend()
    fig.tight_layout()
    plot_path = output_path / "estimated_t_count.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _write_reduction_plot(
    output_path: Path,
    estimates: list[dict[str, object]],
    epsilon_values: tuple[float, ...],
    comparison_benchmarks: tuple[str, ...],
) -> Path | None:
    """Write single- and two-qubit reductions with precision on x-axis."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    budgets = tuple(dict.fromkeys(int(item["budget"]) for item in estimates))
    if not budgets or not comparison_benchmarks:
        return None
    x_values = list(epsilon_values)
    reductions: dict[int, list[float]] = {budget: [] for budget in budgets}
    for epsilon in x_values:
        for budget_value in budgets:
            pair = _comparison_values(
                estimates, budget_value, epsilon, comparison_benchmarks
            )
            if pair is None:
                return None
            grid, ncf = pair
            reductions[budget_value].append(100.0 * (grid - ncf) / grid)

    fig, axis = plt.subplots(figsize=(11, 7))
    labels = {1: "NC-Fusion (single-qubit)", 2: "NC-Fusion (two-qubit)"}
    colors = {1: "tab:orange", 2: "tab:green"}
    for budget_value in budgets:
        axis.plot(
            x_values,
            reductions[budget_value],
            color=colors[budget_value],
            marker="o",
            linewidth=2.0,
            label=labels[budget_value],
        )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("Synthesis precision ε")
    axis.set_ylabel("Estimated T-count reduction (%)")
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"10$^{{-{round(-math.log10(value))}}}$" for value in x_values])
    axis.grid(True, which="both", linestyle=":", alpha=0.45)
    axis.legend()
    fig.tight_layout()
    plot_path = output_path / "t_count_reduction.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def run(
    output: Path | str = "micro_artifact/results/runs/analytical_estimation",
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
    """Generate 13-benchmark average T-count estimates by precision."""

    import numpy as np

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
    benchmarks_by_method: dict[str, list[str]] = {}
    missing_by_method: dict[str, list[str]] = {}
    two_qubit_metadata = None
    if 2 in selected_budgets:
        two_qubit_metadata = _ensure_two_qubit_metadata()

    for selected_budget in selected_budgets:
        producer_method = "ncf-one" if selected_budget == 1 else "ncf-two"
        method_benchmarks: list[str] = []
        for benchmark_name in selected_benchmarks:
            spec = find_benchmark(benchmark_name)
            source_record = producer_record(spec.name, producer_method)
            unitary_value = (source_record or {}).get("ncf_unitaries_generated")
            rz_value = (source_record or {}).get("original_rz_gate_count")
            ncf_rz_value = (
                (source_record or {}).get("ncf_rz_generated")
                if selected_budget == 2
                else 0
            )
            if (
                unitary_value in (None, "")
                or rz_value in (None, "")
                or (selected_budget == 2 and ncf_rz_value in (None, ""))
            ):
                if selected_budget == 2 and benchmark_name not in COMMON_COMPARISON_BENCHMARKS:
                    missing_by_method.setdefault(producer_method, []).append(benchmark_name)
                    continue
                raise RuntimeError(
                    f"Analytical estimation needs {producer_method} producer data for "
                    f"{spec.name}. Run micro_artifact.{('single' if selected_budget == 1 else 'two')}_qubit_result "
                    "first (use --source generate if needed). The producer metrics.csv "
                    "must contain ncf_unitaries_generated, original_rz_gate_count, "
                    "and ncf_rz_generated for ncf-two."
                )
            unitary_count = int(unitary_value)
            original_rz_count = int(rz_value)
            ncf_rz_count = int(ncf_rz_value or 0)
            pauli_string_count = int((source_record or {}).get("pauli_strings") or spec.pauli_terms)
            for epsilon in epsilon_values:
                estimate = estimate_from_unitary_count(
                    unitary_count,
                    original_rz_count,
                    epsilon,
                    budget=selected_budget,
                    ncf_rz_generated=ncf_rz_count,
                    trotter_steps=trotter_steps,
                )
                estimate.update(
                    {
                        "benchmark": spec.name,
                        "method": producer_method,
                        "budget": selected_budget,
                        "epsilon": epsilon,
                        "pauli_strings": pauli_string_count,
                        "original_rz_gate_count": original_rz_count,
                        "ncf_unitaries_generated": unitary_count,
                        "ncf_rz_generated": ncf_rz_count,
                        "compiled_gate_count": (source_record or {}).get("gate_count"),
                        "source_dataset": producer_method,
                        "window": window if window is not None else (4 if selected_budget == 1 else 128),
                        "trotter_steps": trotter_steps,
                        "evolution_time": evolution_time,
                    }
                )
                records.append(estimate)
            method_benchmarks.append(spec.name)
        benchmarks_by_method[producer_method] = method_benchmarks

    available_sets = [set(values) for values in benchmarks_by_method.values()]
    comparison_benchmarks = tuple(
        benchmark
        for benchmark in selected_benchmarks
        if all(benchmark in available for available in available_sets)
    )
    if len(selected_budgets) > 1 and not comparison_benchmarks:
        raise RuntimeError(
            "Analytical estimation needs overlapping ncf-one and ncf-two producer "
            "metrics. Run the two-qubit result first."
        )

    average_records = _average_reduction_table(
        records, epsilon_values, comparison_benchmarks
    )
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "analytical_estimation",
        "paper_section": "5.4",
        "created_by": "micro_artifact.analytical_estimation",
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "benchmarks_by_method": benchmarks_by_method,
        "comparison_benchmarks": comparison_benchmarks,
        "skipped_benchmarks": missing_by_method,
        "two_qubit_metadata": two_qubit_metadata,
        "budgets": selected_budgets,
        "epsilons": epsilon_values,
        "formula_grid_synth": "N_RZ * 3 * log2(1 / epsilon)",
        "formula_single_qubit_ncf": "N_U * 3 * log2(1 / epsilon_scaled)",
        "formula_two_qubit_ncf": "N_U * 15 * log_base_2.76(1 / epsilon_scaled) + N_RZ_NCF * 3 * log2(1 / epsilon_scaled)",
        "precision_rule": {
            "single_qubit": "epsilon_scaled = epsilon * N_RZ / N_U",
            "two_qubit": "epsilon_scaled = epsilon * N_RZ / (N_U + N_RZ_NCF)",
        },
        "baseline_count_source": "original_rz_gate_count from producer dataset",
        "ncf_count_source": "ncf_unitaries_generated and ncf_rz_generated from producer dataset",
        "input_source": "producer metrics.csv only; no QASM fallback",
        "output": "average reductions plus estimated-T-count and reduction figures; average counts hidden from CSV",
        "benchmark_count": len(comparison_benchmarks),
        "average_precision_rule": "NC-Fusion uses epsilon_scaled per benchmark before averaging",
        "record_count": len(average_records),
        "paper_dependencies": dependency_status(),
    }
    output_path = Path(output)
    existing_average_records = [
        row
        for row in read_records_csv(output_path / "metrics.csv")
        if row.get("precision") not in (None, "")
        or row.get("benchmark", "").startswith("AVERAGE_")
    ]
    average_records = merge_records(
        existing_average_records,
        average_records,
        ("method", "precision"),
    )
    manifest["record_count"] = len(average_records)
    manifest["csv_merge_policy"] = (
        "replace matching method/precision rows and append new precisions; "
        "producer templates are merged by benchmark/method"
    )
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", average_records)
    t_count_plot = _write_t_count_plot(
        output_path, records, epsilon_values, comparison_benchmarks
    )
    reduction_plot = _write_reduction_plot(
        output_path, records, epsilon_values, comparison_benchmarks
    )
    plots = {
        "estimated_t_count": str(t_count_plot) if t_count_plot is not None else None,
        "t_count_reduction": str(reduction_plot) if reduction_plot is not None else None,
    }
    manifest["plots"] = plots
    write_json(output_path / "manifest.json", manifest)
    return {
        "manifest": manifest,
        "records": average_records,
        "estimates": records,
        "plots": plots,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run NC-Fusion analytical T-count estimation")
    add_cli_arguments(parser)
    parser.add_argument("--budget", type=int, choices=(1, 2))
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
