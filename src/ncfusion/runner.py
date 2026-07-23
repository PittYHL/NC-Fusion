"""Artifact command orchestration.

The smoke workflow is dependency-free.  Full paper workflows are delegated to
the legacy research implementation through :mod:`legacy`, but all imports are
lazy so evaluators can inspect and validate the artifact without installing
GPU- and synthesizer-specific packages first.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys
import csv
from typing import Any

from .metrics import CircuitMetrics, metrics_from_qasm, write_json, write_records_csv
from .pauli import anticommuting_pairs, commutes, group_by_support
from .spec import ExperimentSpec, BenchmarkSpec, find_experiment, select_benchmarks


PAPER_DEPENDENCIES = (
    "qiskit",
    "qiskit_nature",
    "qiskit_aer",
    "qiskit_gridsynth_plugin",
    "trasyn",
    "rustiq",
)


def dependency_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in PAPER_DEPENDENCIES}


def missing_paper_dependencies(methods: tuple[str, ...] | None = None) -> list[str]:
    """Return missing packages required by the selected methods."""

    required = set(PAPER_DEPENDENCIES)
    if methods is not None and "rustiq" not in methods:
        required.discard("rustiq")
    status = dependency_status()
    return sorted(name for name in required if not status[name])


def smoke_result() -> dict[str, Any]:
    labels = ["XZI", "YZI", "IIZ"]
    groups = group_by_support(labels)
    assert commutes("XZI", "YZI") is False
    assert commutes("XZI", "IIZ") is True
    assert anticommuting_pairs(labels) == 1

    qasm = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[3];
t q[0];
t q[1];
h q[2];
cx q[1],q[2];
t q[0];
"""
    metrics = metrics_from_qasm(qasm)
    assert metrics == CircuitMetrics(t_count=3, t_depth=2, clifford_count=2, gate_count=5)
    return {
        "status": "passed",
        "pauli_groups": {"/".join(map(str, key)): value for key, value in groups.items()},
        "anticommuting_pairs": anticommuting_pairs(labels),
        "qasm_metrics": metrics.as_dict(),
    }


def _manifest(experiment: ExperimentSpec, selected: tuple[BenchmarkSpec, ...], seed: int) -> dict[str, Any]:
    return {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "experiment": asdict(experiment),
        "benchmarks": [asdict(item) for item in selected],
        "paper_dependencies": dependency_status(),
    }


def run_smoke(output: Path) -> dict[str, Any]:
    result = smoke_result()
    result["manifest"] = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(output / "smoke.json", result)
    return result


def run_paper(
    experiment_name: str,
    output: Path,
    requested_benchmarks: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    methods: list[str] | None = None,
    *,
    reuse_existing: bool = True,
    save_qasm: bool = True,
) -> dict[str, Any]:
    experiment = find_experiment(experiment_name)
    selected = select_benchmarks(experiment, requested_benchmarks)
    chosen_methods = tuple(methods or experiment.methods)
    from micro_artifact.data import reusable_record

    # Only the fixed main/scalability configurations can be satisfied solely
    # from the stored producer QASM.  Error, random-order, and sensitivity
    # experiments intentionally change compilation inputs, so they must still
    # validate (and, when needed, import) their synthesis dependencies.
    all_inputs_reusable = reuse_existing and experiment.name in {"table4", "scalability"}
    if all_inputs_reusable:
        for benchmark in selected:
            for method in chosen_methods:
                if method == "ncf-two" and not benchmark.two_qubit_supported:
                    continue
                if reusable_record(benchmark, method) is None:
                    all_inputs_reusable = False
                    break
            if not all_inputs_reusable:
                break
    missing = missing_paper_dependencies(chosen_methods)
    if all_inputs_reusable:
        missing = []
    if missing:
        raise RuntimeError(
            "Full paper workflows require the packages missing from this environment: "
            + ", ".join(missing)
            + ". Install requirements-paper.txt, then rerun the command."
        )
    if "ncf-two" in chosen_methods and shutil.which("docker") is None and not all_inputs_reusable:
        raise RuntimeError(
            "The ncf-two workflow requires Docker and the Synthetiq image. "
            "Install/configure Docker and set SYNTHETIQ_IMAGE, SYNTHETIQ_INPUT_FILE, "
            "and SYNTHETIQ_OUTPUT_DIR before running it."
        )

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import numpy as np

    np.random.seed(seed)
    from .legacy import run_benchmark
    records: list[dict[str, object]] = []
    for benchmark in selected:
        for method in chosen_methods:
            if method == "ncf-two" and not benchmark.two_qubit_supported:
                continue
            settings_runs: list[dict[str, object]] = [dict(experiment.settings)]
            if experiment.name == "sensitivity":
                windows = experiment.settings.get(
                    "single_windows" if method == "ncf-one" else "two_windows", []
                )
                settings_runs = []
                for window in windows:
                    if window == "full":
                        # Full-window values are represented by the measured
                        # Pauli count; run_benchmark resolves it after the
                        # Hamiltonian is loaded.
                        settings_runs.append({**experiment.settings, "window": window})
                    else:
                        key = "single_window" if method == "ncf-one" else "two_window"
                        settings_runs.append({**experiment.settings, key: window, "window": window})
            elif experiment.name == "error-evaluation":
                settings_runs = [
                    {**experiment.settings, "trotter_steps": steps, "trotter_step": steps}
                    for steps in experiment.settings["trotter_steps"]
                ]
            elif experiment.name == "random-order":
                settings_runs = [
                    {**experiment.settings, "pauli_order_seed": seed, "repetition": seed}
                    for seed in range(int(experiment.settings["repetitions"]))
                ]

            for run_settings in settings_runs:
                can_reuse = reuse_existing and experiment.name != "sensitivity"
                can_reuse = can_reuse and int(run_settings.get("trotter_steps", 1)) == 1
                can_reuse = can_reuse and run_settings.get("pauli_order_seed") is None
                if method == "ncf-one":
                    can_reuse = can_reuse and int(run_settings.get("single_window", 4)) == 4
                if method == "ncf-two":
                    can_reuse = can_reuse and int(run_settings.get("two_window", 128)) == 128

                record = reusable_record(benchmark, method) if can_reuse else None
                if record is None:
                    record = run_benchmark(
                        benchmark,
                        method=method,
                        settings=run_settings,
                        gpu=gpu,
                        save_qasm=save_qasm and experiment.name != "sensitivity",
                    )
                for key in ("window", "trotter_step", "repetition"):
                    if key in run_settings:
                        record[key] = run_settings[key]
                records.append(record)

    manifest = _manifest(experiment, selected, seed)
    manifest["methods"] = chosen_methods
    manifest["record_count"] = len(records)
    manifest["reuse_existing"] = reuse_existing
    manifest["save_qasm"] = save_qasm and experiment.name != "sensitivity"
    write_json(output / "manifest.json", manifest)
    write_records_csv(output / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


def validate_table4(actual: Path, reference: Path, tolerance: int = 0) -> dict[str, object]:
    """Compare generated long-form metrics with the paper's wide Table 4."""

    with actual.open(encoding="utf-8", newline="") as handle:
        actual_rows = list(csv.DictReader(handle))
    with reference.open(encoding="utf-8", newline="") as handle:
        reference_rows = {row["benchmark"]: row for row in csv.DictReader(handle)}

    method_prefixes = {
        "gridsyn": "gridsyn",
        "rustiq": "rustiq",
        "ncf-one": "ncf_one",
    }
    mismatches: list[dict[str, object]] = []
    checked = 0
    for row in actual_rows:
        benchmark = row.get("benchmark", "")
        prefix = method_prefixes.get(row.get("method", ""))
        reference_row = reference_rows.get(benchmark)
        if prefix is None or reference_row is None:
            continue
        for metric in ("t_count", "t_depth", "clifford_count"):
            checked += 1
            reference_metric = "clifford" if metric == "clifford_count" else metric
            expected = int(reference_row[f"{prefix}_{reference_metric}"])
            observed = int(row[metric])
            if abs(observed - expected) > tolerance:
                mismatches.append({
                    "benchmark": benchmark,
                    "method": row["method"],
                    "metric": metric,
                    "expected": expected,
                    "observed": observed,
                })
    return {
        "status": "no-data" if checked == 0 else ("passed" if not mismatches else "different"),
        "checked_values": checked,
        "mismatches": mismatches,
        "note": "Synthesis output can vary across software and hardware versions; inspect mismatches rather than treating them as a functional failure.",
    }
