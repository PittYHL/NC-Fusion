"""Metrics and lightweight QASM parsing for artifact outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CircuitMetrics:
    t_count: int
    t_depth: int
    clifford_count: int
    gate_count: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _metrics_from_circuit(circuit) -> CircuitMetrics:
    """Use the one canonical Qiskit definition for every circuit metric."""

    operations = list(circuit.data)
    t_count = sum(item[0].name == "t" or item[0].name == "tdg" for item in operations)
    t_depth = int(
        circuit.depth(lambda gate: gate[0].name == "t" or gate[0].name == "tdg")
    )
    return CircuitMetrics(
        t_count=int(t_count),
        t_depth=t_depth,
        clifford_count=len(operations) - int(t_count),
        gate_count=len(operations),
    )


def metrics_from_qasm(qasm: str) -> CircuitMetrics:
    from qiskit import QuantumCircuit

    return _metrics_from_circuit(QuantumCircuit.from_qasm_str(qasm))


def metrics_from_qasm_file(path: Path) -> CircuitMetrics:
    from qiskit import QuantumCircuit

    return _metrics_from_circuit(QuantumCircuit.from_qasm_file(str(path)))


def read_records_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV result file, returning an empty list when it is absent."""

    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_records(
    existing: Iterable[dict[str, object]],
    updates: Iterable[dict[str, object]],
    key_fields: Sequence[str],
) -> list[dict[str, object]]:
    """Merge result rows by configuration key.

    Existing rows are retained in their original order. A matching update
    replaces that row; a new key is appended. This gives artifact runs the
    requested append/replace behavior without accumulating duplicate rows.
    """

    merged = [dict(row) for row in existing]
    positions: dict[tuple[str, ...], int] = {}

    def key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(
            "" if row.get(field) in (None, "") else str(row.get(field))
            for field in key_fields
        )

    for index, row in enumerate(merged):
        positions[key(row)] = index
    for update in updates:
        row = dict(update)
        row_key = key(row)
        if row_key in positions:
            merged[positions[row_key]] = row
        else:
            positions[row_key] = len(merged)
            merged.append(row)
    return merged


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: Iterable[dict[str, object]]) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
