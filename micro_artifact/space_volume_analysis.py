"""Evaluate the paper's fault-tolerant spacetime-volume experiment.

This follows the original ``ncf/NCF/Superstaq_test.py`` driver: load the
stored Clifford+T QASM for each method and estimate it with Infleqtion's
``resource-superstaq`` pipeline for one and ten T factories.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import platform
from pathlib import Path
from typing import Any

from ncfusion.metrics import metrics_from_qasm_file, write_json, write_records_csv
from ncfusion.spec import find_experiment, select_benchmarks

from .common import run_cli


STATUS = "available"

_ARTIFACT_ROOT = Path(__file__).resolve().parents[1]

# The original benchmark generator uses shorter directory names than the
# paper/configuration names used by the artifact harness.
_BENCHMARK_DIRS = {
    "Ising-2D-30": "IS-2D-30",
    "Ising-2D-60": "IS-2D-60",
    "Ising-3D-30": "IS-3D-30",
    "Ising-3D-60": "IS-3D-60",
    "Heisenberg-2D-30": "Hei-2D-30",
    "Heisenberg-2D-60": "Hei-2D-60",
    "Heisenberg-3D-30": "Hei-3D-30",
    "Heisenberg-3D-60": "Hei-3D-60",
}

_METHOD_FILES = {
    "gridsyn": "grid",
    "rustiq": "rustiq",
    "ncf-one": "ncf",
}

_RESOURCE_FIELDS = (
    "re_superstaq_phys_qubits",
    "re_superstaq_parallel_time_us",
    "re_superstaq_serial_time_us",
    "re_superstaq_volume",
    "re_superstaq_d",
    "re_superstaq_num_t_factories",
    "re_superstaq_primitive_moments",
    "re_superstaq_error",
)


def _normalise_resource_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep CSV columns stable when one estimator row reports an error."""

    return {field: metrics.get(field) for field in _RESOURCE_FIELDS}


def _check_resource_package() -> Any:
    try:
        return importlib.import_module("resource_estimation")
    except ImportError as error:
        raise RuntimeError(
            "Spacetime-volume evaluation requires Infleqtion/resource-superstaq. "
            "Install the pinned revision 717cbbfc62e558be3f2f9acb512e992d3cd43529 "
            "from https://github.com/Infleqtion/resource-superstaq.git, then rerun."
        ) from error


def run(
    output: Path | str = "micro_artifact/results/runs/space_volume_analysis",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
) -> dict[str, Any]:
    """Run Section 5.10 from the stored original-method QASM files."""

    del seed, gpu  # retained for the common artifact CLI contract
    output_path = Path(output)
    experiment = find_experiment("spacetime-volume")
    selected = select_benchmarks(experiment, benchmarks)
    chosen_methods = tuple(methods or experiment.methods)
    unknown = sorted(set(chosen_methods).difference(_METHOD_FILES))
    if unknown:
        known = ", ".join(_METHOD_FILES)
        raise ValueError(f"Unknown spacetime-volume method(s) {unknown}; choose from {known}")

    resource_estimation = _check_resource_package()
    from qiskit import QuantumCircuit
    from .resource_estimators import superstaq_estimate

    factory_counts = tuple(int(value) for value in experiment.settings["magic_state_factories"])
    jobs = []
    missing_files = []
    from .data import existing_qasm_path

    for benchmark in selected:
        for method in chosen_methods:
            qasm_path = existing_qasm_path(benchmark.name, method)
            if qasm_path is None:
                expected = Path("micro_artifact/circuits/single-qubit") / benchmark.name
                missing_files.append(str(expected / f"{benchmark.name}_{_METHOD_FILES[method]}_c+t.qasm"))
            else:
                jobs.append((benchmark, method, qasm_path))
    if missing_files:
        raise RuntimeError(
            "Spacetime-volume inputs are missing from this checkout: "
            + ", ".join(missing_files)
            + ". Supply the original Clifford+T QASM files before running the full experiment."
        )

    records: list[dict[str, Any]] = []
    for benchmark, method, qasm_path in jobs:
        qc = QuantumCircuit.from_qasm_file(str(qasm_path))
        circuit_metrics = metrics_from_qasm_file(qasm_path).as_dict()
        from .data import producer_record

        producer = producer_record(benchmark.name, method) or {}
        compilation_time = producer.get("compilation_time_seconds") or producer.get("runtime_seconds")
        if compilation_time is not None:
            compilation_time = float(compilation_time)
        for factory_count in factory_counts:
            estimate = superstaq_estimate(qc, num_t_factories=factory_count)
            records.append({
                "benchmark": benchmark.name,
                "method": method,
                "source_method": _METHOD_FILES[method],
                "magic_state_factories": factory_count,
                "qasm_path": str(qasm_path.relative_to(_ARTIFACT_ROOT)),
                "data_source": "existing_qasm",
                "compilation_time_seconds": compilation_time,
                "qasm_qubits": qc.num_qubits,
                **{f"qasm_{key}": value for key, value in circuit_metrics.items()},
                **_normalise_resource_metrics(estimate),
            })

    manifest = {
        "artifact_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": asdict(experiment),
        "benchmarks": [asdict(item) for item in selected],
        "methods": chosen_methods,
        "record_count": len(records),
        "python": platform.python_version(),
        "resource_superstaq": {
            "package": "Infleqtion/resource-superstaq",
            "import_name": "resource_estimation",
            "revision": "717cbbfc62e558be3f2f9acb512e992d3cd43529",
            "location": str(getattr(resource_estimation, "__file__", "")),
        },
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


if __name__ == "__main__":
    run_cli("space_volume_analysis", run)
