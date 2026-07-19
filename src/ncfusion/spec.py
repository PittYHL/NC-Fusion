"""Paper experiment specifications and reference metadata.

This module intentionally uses only the Python standard library so that
``list`` and ``smoke`` remain usable before the full quantum stack is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ARTIFACT_ROOT / "configs" / "paper.json"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    family: str
    structure: str
    qubits: int
    pauli_terms: int
    two_qubit_supported: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    section: str
    description: str
    benchmarks: tuple[str, ...]
    methods: tuple[str, ...]
    settings: dict[str, object]


def _load_config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def benchmarks() -> tuple[BenchmarkSpec, ...]:
    config = _load_config()
    rows = config["benchmarks"]
    return tuple(
        BenchmarkSpec(
            name=row["name"],
            family=row["family"],
            structure=row["structure"],
            qubits=int(row["qubits"]),
            pauli_terms=int(row["pauli_terms"]),
            two_qubit_supported=bool(row.get("two_qubit_supported", True)),
        )
        for row in rows
    )


def experiments() -> tuple[ExperimentSpec, ...]:
    config = _load_config()
    return tuple(
        ExperimentSpec(
            name=row["name"],
            section=row["section"],
            description=row["description"],
            benchmarks=tuple(row["benchmarks"]),
            methods=tuple(row["methods"]),
            settings=dict(row.get("settings", {})),
        )
        for row in config["experiments"]
    )


def find_benchmark(name: str) -> BenchmarkSpec:
    for item in benchmarks():
        if item.name.lower() == name.lower():
            return item
    known = ", ".join(item.name for item in benchmarks())
    raise KeyError(f"Unknown benchmark {name!r}; choose one of: {known}")


def find_experiment(name: str) -> ExperimentSpec:
    for item in experiments():
        if item.name.lower() == name.lower():
            return item
    known = ", ".join(item.name for item in experiments())
    raise KeyError(f"Unknown experiment {name!r}; choose one of: {known}")


def select_benchmarks(spec: ExperimentSpec, requested: Iterable[str] | None = None) -> tuple[BenchmarkSpec, ...]:
    names = set(requested or spec.benchmarks)
    selected = tuple(find_benchmark(name) for name in spec.benchmarks if name in names)
    missing = names.difference(item.name for item in selected)
    if missing:
        raise KeyError(f"Benchmarks {sorted(missing)} are not part of experiment {spec.name!r}")
    return selected

