"""Filesystem paths owned by the artifact layer."""

from __future__ import annotations

from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "micro_artifact"
ARTIFACT_CIRCUITS = ARTIFACT_ROOT / "circuits" / "single-qubit"

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


def circuit_path(benchmark: str, suffix: str) -> Path:
    """Return a generated QASM path under ``micro_artifact/circuits``."""

    directory_name = BENCHMARK_DIRS.get(benchmark, benchmark)
    directory = ARTIFACT_CIRCUITS / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{directory_name}_{suffix}.qasm"
