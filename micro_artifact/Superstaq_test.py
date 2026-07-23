"""Small direct driver matching the original ``Superstaq_test.py`` format."""

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit

from .resource_estimators import superstaq_estimate


def main(benchmark_name: str = "IS-3D-30") -> None:
    circuit_root = Path(__file__).resolve().parent / "circuits" / benchmark_name
    files = [
        circuit_root / f"{benchmark_name}_grid_c+t.qasm",
        circuit_root / f"{benchmark_name}_rustiq_c+t.qasm",
        circuit_root / f"{benchmark_name}_ncf_c+t.qasm",
    ]
    methods = ["grid", "rustiq", "ncf"]
    for factory_count in (1, 10):
        print(f"With {factory_count} T factories")
        for file, method in zip(files, methods):
            qc = QuantumCircuit.from_qasm_file(str(file))
            metrics = superstaq_estimate(qc, num_t_factories=factory_count)
            print(f"{method}: {metrics}")
        if factory_count == 1:
            print("--------------------------------")


if __name__ == "__main__":
    main()
