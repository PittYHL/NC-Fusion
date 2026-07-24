"""Shared circuit construction for the Section 5.7 evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_METHODS = ("gridsyn", "ncf-one")


def generated_method_circuit_paths(
    output: Path | str,
    benchmark: str,
    method: str,
    trotter_steps: int,
) -> tuple[Path, Path]:
    """Return step-specific RZ and Clifford+T paths for generated circuits."""

    directory = Path(output) / "circuits" / benchmark
    stem = f"{benchmark}_{method}_trotter_steps_{int(trotter_steps)}"
    return directory / f"{stem}_rz.qasm", directory / f"{stem}_c+t.qasm"


def save_generated_method_circuits(
    output: Path | str,
    benchmark: str,
    method: str,
    trotter_steps: int,
    rz_qc: Any,
    clifford_t_qc: Any,
) -> tuple[Path, Path]:
    """Save both generated circuit forms with the Trotter step in the name."""

    from qiskit import qasm2

    rz_path, clifford_t_path = generated_method_circuit_paths(
        output, benchmark, method, trotter_steps
    )
    rz_path.parent.mkdir(parents=True, exist_ok=True)
    rz_path.write_text(qasm2.dumps(rz_qc), encoding="utf-8")
    clifford_t_path.write_text(qasm2.dumps(clifford_t_qc), encoding="utf-8")
    return rz_path, clifford_t_path


def load_generated_method_circuits(
    output: Path | str,
    benchmark: str,
    method: str,
    trotter_steps: int,
) -> tuple[Any, Any, float | None] | None:
    """Load a previously generated pair for one exact Trotter step count."""

    rz_path, clifford_t_path = generated_method_circuit_paths(
        output, benchmark, method, trotter_steps
    )
    if not rz_path.is_file() or not clifford_t_path.is_file():
        return None

    from qiskit import QuantumCircuit

    compilation_time: float | None = None
    metrics_path = Path(output) / "metrics.csv"
    if metrics_path.is_file():
        import csv

        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("benchmark") == benchmark
                    and row.get("method") == method
                    and row.get("trotter_steps") == str(int(trotter_steps))
                ):
                    value = row.get("compilation_time_seconds")
                    if value not in (None, ""):
                        compilation_time = float(value)
                    break
    return (
        QuantumCircuit.from_qasm_file(str(rz_path)),
        QuantumCircuit.from_qasm_file(str(clifford_t_path)),
        compilation_time,
    )


def load_existing_method_circuits(benchmark: str, method: str) -> tuple[Any, Any, float | None] | None:
    """Load stored ``(RZ, Clifford+T, compilation_time)`` data when present."""

    from .data import existing_qasm_path, producer_metadata

    rz_path = existing_qasm_path(benchmark, method, synthesized=False)
    clifford_t_path = existing_qasm_path(benchmark, method, synthesized=True)
    if rz_path is None or clifford_t_path is None:
        return None
    from qiskit import QuantumCircuit

    metadata = producer_metadata(benchmark, method)
    compilation_time = metadata.get("compilation_time_seconds")
    if compilation_time in (None, ""):
        compilation_time = None
    return (
        QuantumCircuit.from_qasm_file(str(rz_path)),
        QuantumCircuit.from_qasm_file(str(clifford_t_path)),
        float(compilation_time) if compilation_time is not None else None,
    )


def validate_methods(methods: list[str] | None) -> tuple[str, ...]:
    selected = tuple(methods or SUPPORTED_METHODS)
    unknown = [method for method in selected if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            "error evaluations support only gridsyn and ncf-one; "
            f"received {', '.join(unknown)}"
        )
    return tuple(dict.fromkeys(selected))


def compile_method(
    hamiltonian: Any,
    method: str,
    *,
    synthesize: bool,
    error_threshold: float,
    t_budget: int,
    gpu: int,
    trotter_steps: int,
    evolution_time: float,
    window: int | None,
    pauli_order_seed: int | None,
) -> tuple[Any, Any]:
    """Return ``(rz_qc, clifford_t_qc)`` for one method.

    The first circuit is always the unsynthesized rotation circuit. The second
    is the synthesized Clifford+T circuit when ``synthesize=True``.
    """

    if method == "gridsyn":
        from baseline import baseline_circuit

        return baseline_circuit(
            hamiltonian,
            1,
            error_threshold=error_threshold,
            gpu=gpu,
            Trotter_steps=trotter_steps,
            evolution_time=evolution_time,
            rustiq=False,
            GRIDSYNTH=True,
            t_budget=t_budget,
            synthesize=synthesize,
            benchmark=None,
            method="grid",
        )

    if method == "ncf-one":
        from ncfusion import NC_Fusion

        return NC_Fusion(
            hamiltonian,
            budget=1,
            window=window,
            error_threshold=error_threshold,
            t_budget=t_budget,
            gpu=gpu,
            trotter_steps=trotter_steps,
            evolution_time=evolution_time,
            synthesize=synthesize,
            pauli_order_seed=pauli_order_seed,
        )

    raise ValueError(f"unsupported error-evaluation method: {method}")
