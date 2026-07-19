"""Metrics and lightweight QASM parsing for artifact outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
from typing import Iterable


_QUBIT_RE = re.compile(r"\[(\d+)\]")
_GATE_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s+(.+?);$")
_T_GATES = frozenset({"t", "tdg"})
_HEADER_GATES = frozenset({"openqasm", "include", "qreg", "creg", "barrier", "measure", "reset"})


@dataclass(frozen=True)
class CircuitMetrics:
    t_count: int
    t_depth: int
    clifford_count: int
    gate_count: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _operations(qasm: str) -> Iterable[tuple[str, tuple[int, ...]]]:
    for raw_line in qasm.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith("//"):
            continue
        match = _GATE_RE.match(line)
        if not match:
            continue
        name, operands = match.groups()
        name = name.lower()
        if name in _HEADER_GATES:
            continue
        qubits = tuple(int(value) for value in _QUBIT_RE.findall(operands))
        yield name, qubits


def metrics_from_qasm(qasm: str) -> CircuitMetrics:
    t_count = 0
    clifford_count = 0
    gate_count = 0
    t_layers: dict[int, int] = {}
    t_depth = 0

    for name, qubits in _operations(qasm):
        gate_count += 1
        if name in _T_GATES:
            t_count += 1
            layer = 1 + max((t_layers.get(qubit, 0) for qubit in qubits), default=0)
            for qubit in qubits:
                t_layers[qubit] = layer
            t_depth = max(t_depth, layer)
        else:
            clifford_count += 1

    return CircuitMetrics(t_count, t_depth, clifford_count, gate_count)


def metrics_from_qasm_file(path: Path) -> CircuitMetrics:
    return metrics_from_qasm(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: Iterable[dict[str, object]]) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

