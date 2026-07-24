"""Shared circuit and external-tool helpers for method comparison."""

from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any


def operation_parts(item: Any) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    operation = item.operation if hasattr(item, "operation") else item[0]
    qargs = tuple(item.qubits if hasattr(item, "qubits") else item[1])
    cargs = tuple(item.clbits if hasattr(item, "clbits") else item[2])
    return operation, qargs, cargs


def circuit_metrics(circuit: Any) -> dict[str, int]:
    """Return comparable metrics, including RZ count for unsynthesized stages."""

    operations = list(circuit.data)
    t_names = {"t", "tdg"}
    t_count = 0
    rz_count = 0
    for item in operations:
        operation, qargs, _ = operation_parts(item)
        name = operation.name.lower()
        if name == "rz":
            rz_count += 1
        if name not in t_names:
            continue
        t_count += 1

    # Match ncf/NCF/tzap_test.py: both T and inverse-T operations contribute
    # to the T-depth filter.
    t_depth = int(
        circuit.depth(lambda item: item[0].name == "t" or item[0].name == "tdg")
    )

    return {
        "t_count": int(t_count),
        "t_depth": int(t_depth),
        "clifford_count": int(len(operations) - t_count),
        "gate_count": int(len(operations)),
        "rz_count": int(rz_count),
    }


def write_qasm(circuit: Any, path: str | Path) -> Path:
    from qiskit import qasm2

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(qasm2.dumps(circuit), encoding="utf-8")
    return destination


def synthesize_rz(circuit: Any, error_threshold: float) -> Any:
    """Synthesize every RZ in ``circuit`` with the paper's GridSynth plugin."""

    from qiskit import QuantumCircuit
    from qiskit_gridsynth_plugin.decompose import clifford_t_transpile

    if error_threshold <= 0:
        raise ValueError("error_threshold must be positive")

    synthesized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for item in circuit.data:
        instruction, qargs, cargs = operation_parts(item)
        name = instruction.name.lower()
        q_indices = [circuit.find_bit(qubit).index for qubit in qargs]
        c_indices = [circuit.find_bit(bit).index for bit in cargs]
        target_qargs = [synthesized.qubits[index] for index in q_indices]
        target_cargs = [synthesized.clbits[index] for index in c_indices]

        if name != "rz":
            if name == "swap":
                synthesized.cx(q_indices[0], q_indices[1])
                synthesized.cx(q_indices[1], q_indices[0])
                synthesized.cx(q_indices[0], q_indices[1])
            else:
                synthesized.append(instruction.copy(), target_qargs, target_cargs)
            continue

        one_qubit = QuantumCircuit(1)
        one_qubit.rz(float(instruction.params[0]), 0)
        replacement = clifford_t_transpile(one_qubit, epsilon=error_threshold)
        for replacement_item in replacement.data:
            replacement_instruction = (
                replacement_item.operation
                if hasattr(replacement_item, "operation")
                else replacement_item[0]
            )
            synthesized.append(
                replacement_instruction.copy(),
                [synthesized.qubits[q_indices[0]]],
                [],
            )

    return synthesized


def run_tzap(
    circuit: Any,
    input_path: str | Path,
    output_path: str | Path,
    executable: str = "tzap",
) -> tuple[Any, float, str]:
    """Run tzap on a Qiskit circuit and load its OpenQASM output."""

    from qiskit import qasm2

    input_file = write_qasm(circuit, input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(executable)
    if not command:
        raise ValueError("tzap executable cannot be empty")
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        raise RuntimeError(
            f"Could not find tzap executable {command[0]!r}. Build/install "
            "https://github.com/qqq-wisc/tzap and pass --tzap-bin or set TZAP_BIN."
        )

    started = time.perf_counter()
    completed = subprocess.run(
        command + [str(input_file), "-o", str(output_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"tzap failed with exit code {completed.returncode}: {details}")
    if not output_file.exists():
        raise RuntimeError(f"tzap completed without writing {output_file}")
    return qasm2.loads(output_file.read_text(encoding="utf-8")), elapsed, " ".join(command)
