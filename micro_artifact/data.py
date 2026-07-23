"""Reusable artifact inputs and producer records.

The artifact keeps generated Clifford+T and Clifford+RZ QASM under
``micro_artifact/circuits``.  This module gives every evaluation the same
benchmark/method naming and reuse behavior.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
from typing import Any
import warnings

from ncfusion.metrics import metrics_from_qasm_file


ARTIFACT_ROOT = Path(__file__).resolve().parent
CIRCUIT_ROOT = ARTIFACT_ROOT / "circuits"
PRIMARY_CIRCUIT_ROOT = CIRCUIT_ROOT / "single-qubit"
RESULT_ROOT = ARTIFACT_ROOT / "results" / "runs"

BENCHMARK_DIRS = {
    "Ising-2D-30": "IS-2D-30",
    "Ising-2D-60": "IS-2D-60",
    "Ising-3D-30": "IS-3D-30",
    "Ising-3D-60": "IS-3D-60",
    "Heisenberg-2D-30": "Hei-2D-30",
    "Heisenberg-2D-60": "Hei-2D-60",
    "Heisenberg-3D-30": "Hei-3D-30",
    "Heisenberg-3D-60": "Hei-3D-60",
}

METHOD_STEMS = {
    "gridsyn": "grid",
    "rustiq": "rustiq",
    "phoenix": "phoenix",
    "ncf-one": "ncf",
    "ncf-two": "ncf-two",
}

PRODUCER_OUTPUTS = {
    "ncf-one": RESULT_ROOT / "single_qubit_result",
    "ncf-two": RESULT_ROOT / "two_qubit_result",
}


def benchmark_directory(benchmark: str) -> str:
    return BENCHMARK_DIRS.get(benchmark, benchmark)


def qasm_path(benchmark: str, method: str, *, synthesized: bool = True) -> Path:
    """Return the canonical stored QASM path for a benchmark/method."""

    directory = benchmark_directory(benchmark)
    stem = METHOD_STEMS[method]
    suffix = "c+t" if synthesized else "rz"
    if method in {"rustiq", "phoenix"} and not synthesized:
        suffix = "RZ"
    return PRIMARY_CIRCUIT_ROOT / directory / f"{directory}_{stem}_{suffix}.qasm"


def existing_qasm_path(benchmark: str, method: str, *, synthesized: bool = True) -> Path | None:
    """Find an existing QASM file, including the legacy two-qubit spelling."""

    directory = benchmark_directory(benchmark)
    stem = METHOD_STEMS[method]
    suffixes = ("c+t",) if synthesized else (
        ("RZ", "rz") if method in {"rustiq", "phoenix"} else ("rz",)
    )
    candidates = []
    for suffix in suffixes:
        filename = f"{directory}_{stem}_{suffix}.qasm"
        candidates.extend(
            (
                PRIMARY_CIRCUIT_ROOT / directory / filename,
                CIRCUIT_ROOT / directory / filename,
                CIRCUIT_ROOT / "two-qubit" / directory / filename,
            )
        )
    if method == "ncf-two" and synthesized:
        candidates.extend(
            path.with_name(path.name.replace("ncf-two", "ncf_two"))
            for path in tuple(candidates)
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def read_records(output_dir: Path) -> list[dict[str, str]]:
    path = output_dir / "metrics.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def producer_record(benchmark: str, method: str) -> dict[str, str] | None:
    rows = read_records(PRODUCER_OUTPUTS[method]) if method in PRODUCER_OUTPUTS else []
    for row in reversed(rows):
        if row.get("benchmark") == benchmark and row.get("method") == method:
            return row
    return None


def _is_clifford_rz(angle: object) -> bool:
    try:
        ratio = float(angle) / (math.pi / 2)
    except (TypeError, ValueError):
        return False
    return math.isclose(ratio, round(ratio), abs_tol=1e-8)


_QASM_GATE_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s+.+?;$")
_QASM_QUBIT_RE = re.compile(r"\[(\d+)\]")


def _qasm_gate_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("//", 1)[0].strip()
            match = _QASM_GATE_RE.match(line)
            if match:
                yield (
                    match.group(1).lower(),
                    match.group(2),
                    tuple(int(value) for value in _QASM_QUBIT_RE.findall(line)),
                )


def _count_rotations(path: Path, *, non_clifford_only: bool = False) -> int:
    count = 0
    for name, parameters, _ in _qasm_gate_lines(path):
        if name in {"u", "u3"}:
            count += 1
        elif name == "rz" and non_clifford_only:
            # The supplied NC-Fusion RZ files use U3 for generated units. If
            # a future file uses RZ directly, numeric angles can still be
            # classified without constructing a large Qiskit circuit.
            angle = (parameters or "").strip()
            try:
                is_clifford = _is_clifford_rz(float(angle))
            except ValueError:
                is_clifford = False
            if not is_clifford:
                count += 1
    return count


def qasm_gate_count(path: Path, gate: str) -> int:
    return sum(1 for name, _, _ in _qasm_gate_lines(path) if name == gate.lower())


def _streaming_t_depth(path: Path) -> int:
    """Mirror Qiskit's filtered depth without materializing huge QASM files."""

    object_depths: dict[int, int] = {}
    depth = 0
    for name, _, qubits in _qasm_gate_lines(path):
        new_depth = max((object_depths.get(qubit, 0) for qubit in qubits), default=0)
        if name == "t":
            new_depth += 1
        for qubit in qubits:
            object_depths[qubit] = new_depth
        depth = max(depth, new_depth)
    return depth


def exact_t_depth(path: Path) -> int:
    """Compute T depth with the Qiskit filter used by the paper scripts."""

    # QASM above this size can contain hundreds of thousands of gates; the
    # equivalent streaming calculation avoids materializing a large Qiskit
    # circuit while preserving the filtered-depth synchronization rules.
    if path.stat().st_size > 1_000_000:
        return _streaming_t_depth(path)

    from qiskit import QuantumCircuit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        circuit = QuantumCircuit.from_qasm_file(str(path))
        return int(circuit.depth(lambda gate: gate[0].name == "t"))


def producer_metadata(benchmark: str, method: str) -> dict[str, Any]:
    """Return reusable producer metrics, deriving missing legacy fields."""

    row = producer_record(benchmark, method) or {}
    metadata: dict[str, Any] = {
        "compilation_time_seconds": row.get("compilation_time_seconds") or row.get("runtime_seconds"),
        "runtime_seconds": row.get("runtime_seconds") or row.get("compilation_time_seconds"),
        "ncf_unitaries_generated": row.get("ncf_unitaries_generated"),
        "original_rz_gate_count": row.get("original_rz_gate_count"),
    }
    if method == "ncf-one":
        original_path = existing_qasm_path(benchmark, "gridsyn", synthesized=False)
        ncf_path = existing_qasm_path(benchmark, "ncf-one", synthesized=False)
        if original_path is not None:
            if metadata["original_rz_gate_count"] is None:
                metadata["original_rz_gate_count"] = qasm_gate_count(original_path, "rz")
            metadata["rz_qasm_path"] = str(original_path.relative_to(ARTIFACT_ROOT.parent))
        if ncf_path is not None:
            if metadata["ncf_unitaries_generated"] is None:
                metadata["ncf_unitaries_generated"] = _count_rotations(
                    ncf_path, non_clifford_only=True
                )
            metadata["ncf_rz_qasm_path"] = str(ncf_path.relative_to(ARTIFACT_ROOT.parent))
    return metadata


def reusable_record(
    benchmark_spec: Any,
    method: str,
    *,
    use_exact_t_depth: bool = False,
) -> dict[str, Any] | None:
    """Build a runner-compatible record from an existing synthesized QASM."""

    path = existing_qasm_path(benchmark_spec.name, method)
    if path is None:
        return None
    metrics = metrics_from_qasm_file(path).as_dict()
    record: dict[str, Any] = {
        **metrics,
        "benchmark": benchmark_spec.name,
        "method": method,
        "qubits": benchmark_spec.qubits,
        "qasm_path": str(path.relative_to(ARTIFACT_ROOT.parent)),
        "data_source": "existing_qasm",
    }
    record.update(producer_metadata(benchmark_spec.name, method))
    if use_exact_t_depth:
        record["t_depth"] = exact_t_depth(path)
    return record
