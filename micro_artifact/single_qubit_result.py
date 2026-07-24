"""Single-qubit NC-Fusion result evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from ncfusion.metrics import read_records_csv, write_json, write_records_csv
from ncfusion.spec import find_benchmark, find_experiment

from .common import add_cli_arguments, run_configured
from .data import exact_t_depth, reusable_record


STATUS = "available"
MAIN_BENCHMARKS = tuple(find_experiment("table4").benchmarks)
RUSTIQ_PHOENIX_BENCHMARKS = MAIN_BENCHMARKS + ("H2S", "CO2")
EXTRA_COMPARISON_BENCHMARKS = ("H2S", "CO2", "MgO", "NaCl")
DEFAULT_BENCHMARKS = MAIN_BENCHMARKS + EXTRA_COMPARISON_BENCHMARKS
BASELINE_METHODS = ("gridsyn", "rustiq", "phoenix")
METHOD_ALIASES = {
    "ncf": "ncf-one",
    "ncf_one": "ncf-one",
    "grid": "gridsyn",
    "gridsynth": "gridsyn",
}
SUPPORTED_METHODS = ("ncf-one", *BASELINE_METHODS)


def _selected_baselines(methods: list[str] | None) -> tuple[str, ...]:
    """Resolve method selection to the comparison arms to refresh.

    NC-Fusion is always the candidate circuit in this producer.  Selecting
    ``ncf-one`` by itself therefore refreshes the default GridSynth comparison;
    selecting a baseline refreshes only that baseline comparison.  With no
    selection, preserve the original all-comparisons behavior.
    """

    if methods is None:
        return BASELINE_METHODS
    normalized = tuple(
        dict.fromkeys(METHOD_ALIASES.get(method.lower(), method.lower()) for method in methods)
    )
    unknown = [method for method in normalized if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            "single-qubit methods must be selected from ncf-one, gridsyn, rustiq, "
            f"and phoenix; received {', '.join(unknown)}"
        )
    baselines = tuple(method for method in normalized if method in BASELINE_METHODS)
    if baselines:
        return baselines
    return ("gridsyn",)


def _merge_partial_records(
    existing: list[dict[str, object]],
    updates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge benchmark rows while preserving fields from other method runs."""

    merged = [dict(row) for row in existing]
    positions = {
        str(row.get("benchmark", "")): index
        for index, row in enumerate(merged)
    }
    for update in updates:
        benchmark = str(update.get("benchmark", ""))
        if benchmark in positions:
            combined = dict(merged[positions[benchmark]])
            combined.update(update)
            merged[positions[benchmark]] = combined
        else:
            positions[benchmark] = len(merged)
            merged.append(dict(update))
    return merged


def _reduction_percent(reference: object, candidate: object) -> float | None:
    reference_value = float(reference)
    candidate_value = float(candidate)
    if reference_value == 0:
        return 0.0 if candidate_value == 0 else None
    return 100.0 * (reference_value - candidate_value) / reference_value


def _set_exact_ncf_depth(ncf_record: dict[str, Any]) -> None:
    qasm_path = Path(str(ncf_record.get("qasm_path", "")))
    if qasm_path.is_file():
        ncf_record["t_depth"] = exact_t_depth(qasm_path)


def _add_baseline_comparison(
    benchmark_name: str,
    ncf_record: dict[str, Any],
    baseline_method: str,
) -> dict[str, Any]:
    """Add one baseline's metrics and reductions to a benchmark row."""

    baseline_record = reusable_record(
        find_benchmark(benchmark_name), baseline_method, use_exact_t_depth=True
    )
    if baseline_record is None:
        if baseline_method != "gridsyn":
            return ncf_record
        raise RuntimeError(
            f"No stored GridSynth QASM for {benchmark_name}; "
            "the single-qubit comparison requires the *_grid_c+t.qasm file."
        )
    prefix = baseline_method.replace("-", "_")
    reductions = {
        f"{prefix}_t_count_reduction_percent": _reduction_percent(
            baseline_record["t_count"], ncf_record["t_count"]
        ),
        f"{prefix}_t_depth_reduction_percent": _reduction_percent(
            baseline_record["t_depth"], ncf_record["t_depth"]
        ),
        f"{prefix}_clifford_reduction_percent": _reduction_percent(
            baseline_record["clifford_count"], ncf_record["clifford_count"]
        ),
    }
    ncf_record.update(
        {
            f"{prefix}_t_count": baseline_record["t_count"],
            f"{prefix}_t_depth": baseline_record["t_depth"],
            f"{prefix}_clifford_count": baseline_record["clifford_count"],
            f"{prefix}_qasm_path": baseline_record["qasm_path"],
            **reductions,
        }
    )
    if baseline_method == "gridsyn":
        ncf_record.update(
            {
                "comparison_method": "gridsyn",
                "gridsyn_t_count": baseline_record["t_count"],
                "gridsyn_t_depth": baseline_record["t_depth"],
                "gridsyn_clifford_count": baseline_record["clifford_count"],
                "gridsyn_qasm_path": baseline_record["qasm_path"],
                "t_count_reduction_percent": reductions["gridsyn_t_count_reduction_percent"],
                "t_depth_reduction_percent": reductions["gridsyn_t_depth_reduction_percent"],
                "clifford_reduction_percent": reductions["gridsyn_clifford_reduction_percent"],
            }
        )
    return ncf_record


def _add_average_record(
    records: list[dict[str, Any]], baseline_method: str
) -> dict[str, Any]:
    prefix = baseline_method.replace("-", "_")
    reduction_fields = tuple(
        f"{prefix}_{metric}_reduction_percent"
        for metric in ("t_count", "t_depth", "clifford")
    )
    valid_records: list[dict[str, Any]] = []
    for record in records:
        try:
            float(record.get(reduction_fields[0], ""))
        except (TypeError, ValueError):
            continue
        valid_records.append(record)
    averages = {}
    for field in reduction_fields:
        values: list[float] = []
        for record in valid_records:
            try:
                values.append(float(record.get(field, "")))
            except (TypeError, ValueError):
                continue
        averages[field] = sum(values) / len(values) if values else None
    if baseline_method == "gridsyn":
        averages.update(
            {
                "t_count_reduction_percent": averages["gridsyn_t_count_reduction_percent"],
                "t_depth_reduction_percent": averages["gridsyn_t_depth_reduction_percent"],
                "clifford_reduction_percent": averages["gridsyn_clifford_reduction_percent"],
            }
        )
    return {
        "benchmark": f"AVERAGE_{prefix.upper()}_{len(valid_records)}",
        "method": "ncf-one",
        "comparison_method": baseline_method,
        "data_source": "aggregate",
        "comparison_benchmark_count": len(valid_records),
        **averages,
    }


def run(
    output: Path | str = "micro_artifact/results/runs/single_qubit_result",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    source: str = "existing",
    error_threshold: float = 0.001,
) -> dict[str, Any]:
    """Load or generate the single-qubit NC-Fusion producer dataset."""

    if source not in {"existing", "generate"}:
        raise ValueError("source must be existing or generate")
    if error_threshold <= 0:
        raise ValueError("error_threshold must be positive")
    selected_baselines = _selected_baselines(methods)
    selected = list(DEFAULT_BENCHMARKS if benchmarks is None else benchmarks)
    if source == "generate":
        configured_experiments = {
            "table4": set(MAIN_BENCHMARKS),
            "scalability": set(find_experiment("scalability").benchmarks),
        }
        generated_records: list[dict[str, Any]] = []
        for experiment_name, allowed in configured_experiments.items():
            experiment_benchmarks = [name for name in selected if name in allowed]
            if not experiment_benchmarks:
                continue
            result = run_configured(
                experiment_name,
                output,
                benchmarks=experiment_benchmarks,
                methods=["ncf-one"],
                seed=seed,
                gpu=gpu,
                reuse_existing=False,
                save_qasm=True,
                synthesis_error=error_threshold,
            )
            generated_records.extend(result["records"])
        unsupported = [
            name
            for name in selected
            if not any(name in allowed for allowed in configured_experiments.values())
        ]
        if unsupported:
            raise ValueError(
                "single-qubit generation does not support benchmark(s): "
                + ", ".join(unsupported)
            )
        records = generated_records
    else:
        records = []
        for benchmark_name in selected:
            record = reusable_record(
                find_benchmark(benchmark_name), "ncf-one", use_exact_t_depth=True
            )
            if record is None:
                raise RuntimeError(
                    f"No stored single-qubit QASM for {benchmark_name}; rerun with --source generate."
                )
            records.append(record)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    processed_records: list[dict[str, Any]] = []
    for record in records:
        _set_exact_ncf_depth(record)
        for baseline_method in selected_baselines:
            _add_baseline_comparison(record["benchmark"], record, baseline_method)
        processed_records.append(record)
        checkpoint_existing = read_records_csv(output_path / "metrics.csv")
        checkpoint_benchmarks = [
            row
            for row in checkpoint_existing
            if not str(row.get("benchmark", "")).startswith("AVERAGE_")
        ]
        checkpoint_records = _merge_partial_records(
            checkpoint_benchmarks,
            processed_records,
        )
        checkpoint_summaries = [
            _add_average_record(checkpoint_records, method)
            for method in BASELINE_METHODS
        ]
        write_records_csv(
            output_path / "metrics.csv",
            [*checkpoint_records, *checkpoint_summaries],
        )

    existing_records = read_records_csv(output_path / "metrics.csv")
    existing_benchmark_records = [
        row
        for row in existing_records
        if not str(row.get("benchmark", "")).startswith("AVERAGE_")
    ]
    records = _merge_partial_records(existing_benchmark_records, records)
    summary_records = [_add_average_record(records, method) for method in BASELINE_METHODS]
    records.extend(summary_records)
    average_reductions = {
        method: {
            field: summary[field]
            for field in (
                f"{method}_t_count_reduction_percent",
                f"{method}_t_depth_reduction_percent",
                f"{method}_clifford_reduction_percent",
            )
        }
        for method, summary in zip(BASELINE_METHODS, summary_records)
    }
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "single_qubit_result",
        "source": source,
        "gpu": gpu,
        "benchmarks": selected,
        "methods": list(
            dict.fromkeys(
                METHOD_ALIASES.get(method.lower(), method.lower())
                for method in (methods or SUPPORTED_METHODS)
            )
        ),
        "comparison_methods": list(selected_baselines),
        "record_count": len(records),
        "benchmark_record_count": len(records) - len(summary_records),
        "summary_record_count": len(summary_records),
        "average_reductions": average_reductions,
        "reduction_baselines": {
            "gridsyn": "15 benchmarks",
            "rustiq": "13 benchmarks (excluding MgO and NaCl)",
            "phoenix": "13 benchmarks (excluding MgO and NaCl)",
        },
        "producer_dataset": "single_qubit_result",
        "csv_merge_policy": "replace matching benchmark rows and append new benchmarks; rebuild aggregate rows",
    }
    if source == "generate":
        manifest["source"] = "generate"
        manifest["synthesis_error_override"] = error_threshold
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the single-qubit NC-Fusion evaluation")
    add_cli_arguments(parser, include_source=True)
    parser.add_argument("--error-threshold", type=float, default=0.001)
    args = parser.parse_args(argv)
    try:
        result = run(
            output=args.output,
            benchmarks=args.benchmarks,
            methods=args.methods,
            seed=args.seed,
            gpu=args.gpu,
            source=args.source,
            error_threshold=args.error_threshold,
        )
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Completed {result['manifest']['record_count']} records; wrote {args.output}")


if __name__ == "__main__":
    main()
