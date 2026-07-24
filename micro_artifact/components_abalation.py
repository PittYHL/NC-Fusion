"""NC-Fusion component ablation from the legacy ``main_alg.py`` flow."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
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
    "scheduling",
    "commuting-grouping",
    "anti-commuting-grouping",
)
VARIANT_ALIASES = {
    "ncf-one": "scheduling",
    "ncf-one-without-reordering": "commuting-grouping",
    "ncf-one-without-reordering-and-commuting-grouping": "anti-commuting-grouping",
    "full_ncfusion": "scheduling",
    "anti_commuting_plus_commuting": "commuting-grouping",
    "anti_commuting_only": "anti-commuting-grouping",
}
VARIANT_LABELS = {
    "anti-commuting-grouping": "Anti-commuting grouping",
    "commuting-grouping": "Commuting grouping",
    "scheduling": "Scheduling",
}
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
    without_reordering = transformed_group
    reordered = reorder_pauli_groups(*transformed_group)
    return {
        "scheduling": reordered,
        "commuting-grouping": without_reordering,
        "anti-commuting-grouping": transformed_no_commute,
    }, len(labels)


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _record_key(record: dict[str, object]) -> tuple[str, ...]:
    def value(field: str) -> str:
        item = record.get(field, "")
        return "" if item in (None, "") else str(item)

    return tuple(
        value(field)
        for field in (
            "benchmark",
            "variant",
            "budget",
            "window",
            "trotter_steps",
            "evolution_time",
            "error_threshold",
            "pauli_order_seed",
        )
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


def _reusable_scheduling_record(
    benchmark: Any,
    error_threshold: float,
) -> dict[str, Any] | None:
    """Reuse canonical producer QASM only for its default threshold."""

    record = reusable_record(benchmark, "ncf-one")
    if record is None:
        return None
    stored_threshold = record.get("error_threshold")
    if stored_threshold in (None, ""):
        stored_threshold = 0.001
    try:
        matches_threshold = math.isclose(
            float(stored_threshold), float(error_threshold), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        matches_threshold = False
    return record if matches_threshold else None


def _circuit_metrics(circuit: Any) -> dict[str, int]:
    def operation(item: Any) -> Any:
        return item.operation if hasattr(item, "operation") else item[0]

    operations = list(circuit.data)
    t_names = {"t", "tdg"}
    t_count = sum(operation(item).name.lower() in t_names for item in operations)
    t_depth = int(
        circuit.depth(lambda gate: gate[0].name == "t" or gate[0].name == "tdg")
    )
    return {
        "t_count": int(t_count),
        "t_depth": t_depth,
        "clifford_count": len(operations) - int(t_count),
        "gate_count": len(operations),
    }


_RELATIVE_METRICS = ("t_count", "t_depth", "clifford_count")
_RELATIVE_LABELS = {
    "t_count": "T-count",
    "t_depth": "T-depth",
    "clifford_count": "Clifford count",
}


def _raw_reduction_field(metric: str) -> str:
    return (
        "clifford_reduction_percent"
        if metric == "clifford_count"
        else f"{metric}_reduction_percent"
    )


def _relative_reduction_percent(
    variant: str,
    scheduling_reduction: object,
    variant_reduction: object,
) -> float | None:
    if variant == "scheduling":
        return 100.0
    if scheduling_reduction in (None, "", "nan") or variant_reduction in (None, "", "nan"):
        return None
    baseline = float(scheduling_reduction)
    candidate = float(variant_reduction)
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return 100.0 * candidate / baseline


def _relative_records(
    records: list[dict[str, object]],
    selected_variants: tuple[str, ...],
) -> list[dict[str, object]]:
    by_benchmark: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        benchmark = str(record.get("benchmark", ""))
        variant = str(record.get("variant", ""))
        if not benchmark or benchmark.startswith("AVERAGE_") or variant not in DEFAULT_VARIANTS:
            continue
        by_benchmark.setdefault(benchmark, {})[variant] = record

    relative: list[dict[str, object]] = []
    for benchmark, variants in by_benchmark.items():
        scheduling = variants.get("scheduling")
        if scheduling is None:
            continue
        row_by_variant = {
            variant: variants.get(variant)
            for variant in selected_variants
            if variants.get(variant) is not None
        }
        for variant, source in row_by_variant.items():
            item: dict[str, object] = {
                "benchmark": benchmark,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "data_source": "derived_from_metrics",
            }
            for metric in _RELATIVE_METRICS:
                item[f"{metric}_relative_reduction_percent"] = _relative_reduction_percent(
                    variant,
                    scheduling.get(_raw_reduction_field(metric)),
                    source.get(_raw_reduction_field(metric)),
                )
            relative.append(item)
    return relative


def _relative_average_records(
    relative: list[dict[str, object]],
    selected_variants: tuple[str, ...],
) -> list[dict[str, object]]:
    benchmark_count = len({str(item["benchmark"]) for item in relative})
    averages: list[dict[str, object]] = []
    for variant in selected_variants:
        rows = [item for item in relative if item["variant"] == variant]
        average: dict[str, object] = {
            "benchmark": f"AVERAGE_{benchmark_count}",
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "data_source": "aggregate",
            "benchmark_count": len(rows),
        }
        for metric in _RELATIVE_METRICS:
            field = f"{metric}_relative_reduction_percent"
            values = [float(item[field]) for item in rows if item.get(field) is not None]
            average[field] = sum(values) / len(values) if values else None
        averages.append(average)
    return averages


def _write_relative_plot(
    output_path: Path,
    relative: list[dict[str, object]],
    averages: list[dict[str, object]],
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    variants = tuple(item["variant"] for item in averages)
    benchmarks = list(dict.fromkeys(str(item["benchmark"]) for item in relative))
    plot_records = [
        *relative,
        *[
            {**average, "benchmark": "Average"}
            for average in averages
        ],
    ]
    labels = [*benchmarks, "Average"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    for axis, metric in zip(axes, _RELATIVE_METRICS):
        field = f"{metric}_relative_reduction_percent"
        widths = np.linspace(0.78, 0.34, max(len(variants), 1))
        for index, variant in enumerate(variants):
            values = []
            for label in labels:
                row = next(
                    (
                        item
                        for item in plot_records
                        if item["benchmark"] == label
                        and item["variant"] == variant
                    ),
                    None,
                )
                value = None if row is None else row.get(field)
                values.append(min(100.0, float(value)) if value is not None else 0.0)
            axis.bar(
                x,
                values,
                widths[index],
                label=VARIANT_LABELS[variant],
            )
        axis.axhline(100, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(_RELATIVE_LABELS[metric])
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Percentage of Scheduling (NCF-one) reduction (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(variants),
        frameon=False,
    )
    figure.suptitle("Component ablation relative to Scheduling (NCF-one)", y=0.91)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "components_abalation_relative_reductions.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return plot_path


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

    requested_variants = tuple(variants or methods or DEFAULT_VARIANTS)
    selected_variants = tuple(
        dict.fromkeys(VARIANT_ALIASES.get(variant, variant) for variant in requested_variants)
    )
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
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    records = _read_records(output_path / "metrics.csv")

    def add_record(record: dict[str, object]) -> None:
        nonlocal records
        records = _merge_record(records, record)
        write_records_csv(output_path / "metrics.csv", records)

    for benchmark_name in selected_benchmarks:
        spec = find_benchmark(benchmark_name)
        hamiltonian = build_hamiltonian(spec)
        grouped_inputs = None
        pauli_count = int(spec.pauli_terms)
        if any(variant != "scheduling" for variant in selected_variants) or budget != 1:
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
            if variant == "scheduling" and budget == 1:
                existing = _reusable_scheduling_record(spec, error_threshold)
                if existing is not None:
                    record = {
                        key: int(existing[key])
                        for key in ("t_count", "t_depth", "clifford_count", "gate_count")
                    }
                    record.update(
                        {
                            "benchmark": spec.name,
                            "variant": variant,
                            "variant_label": VARIANT_LABELS[variant],
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
                            "error_threshold": error_threshold,
                            "pauli_order_seed": pauli_order_seed,
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
                    add_record(record)
                    continue

            start = time.perf_counter()
            compile_started = time.perf_counter()
            if variant == "scheduling" and budget == 1:
                rz_qc, clifford_t_qc = compile_method(
                    hamiltonian,
                    "ncf-one",
                    synthesize=True,
                    error_threshold=error_threshold,
                    t_budget=t_budget,
                    gpu=gpu,
                    trotter_steps=trotter_steps,
                    evolution_time=evolution_time,
                    window=window,
                    pauli_order_seed=pauli_order_seed,
                )
            else:
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
            compilation_time = time.perf_counter() - compile_started
            if clifford_t_qc is None:
                raise RuntimeError(f"component variant {variant} did not synthesize a circuit")
            record = _circuit_metrics(clifford_t_qc)
            record.update(
                {
                    "benchmark": spec.name,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
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
                    "compilation_time_seconds": round(compilation_time, 4),
                    "data_source": "generated",
                    "error_threshold": error_threshold,
                    "pauli_order_seed": pauli_order_seed,
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
            record["runtime_seconds"] = round(time.perf_counter() - start, 4)
            add_record(record)

    relative = _relative_records(records, selected_variants)
    relative_averages = _relative_average_records(relative, selected_variants)
    relative_records = [*relative, *relative_averages]
    relative_plot = _write_relative_plot(output_path, relative, relative_averages)
    write_records_csv(output_path / "relative_metrics.csv", relative_records)

    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "components_abalation",
        "paper_section": "5.6.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "benchmarks": selected_benchmarks,
        "variants": selected_variants,
        "variant_definitions": {
            "scheduling": "normal NC-Fusion; reuse single-qubit producer QASM when available",
            "commuting-grouping": "new_paulis, commute_paulis, and circuits without reorder_pauli_groups",
            "anti-commuting-grouping": "no_commute_new_paulis, no_commute_commute_paulis, and no_commute_circuits only",
        },
        "record_count": len(records),
        "csv_merge_policy": "replace matching benchmark/variant/budget/window/Trotter configuration rows and append new configurations",
        "relative_metrics_file": str(output_path / "relative_metrics.csv"),
        "relative_plot": str(relative_plot) if relative_plot is not None else None,
        "relative_definition": "100 * component reduction / scheduling (NCF-one) reduction",
        "paper_dependencies": dependency_status(),
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {
        "manifest": manifest,
        "records": records,
        "relative_records": relative_records,
        "relative_plot": relative_plot,
    }


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
