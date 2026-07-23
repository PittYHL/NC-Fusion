import subprocess
import os
import shutil
from qiskit import QuantumCircuit
from pathlib import Path
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
import numpy as np
from paths import ARTIFACT_ROOT


_DEFAULT_SYNTHETIQ_OUTPUT = ARTIFACT_ROOT / "synthetiq_out"
_DEFAULT_UNITARY = ARTIFACT_ROOT / "unitary.txt"

def run_docker_with_volumes():
    """Start Synthetiq using evaluator-provided portable host paths."""
    image = os.environ.get("SYNTHETIQ_IMAGE", "synthetiq")
    container = os.environ.get("SYNTHETIQ_CONTAINER", "synthetiq_container")
    input_file = Path(
        os.environ.get("SYNTHETIQ_INPUT_FILE", str(_DEFAULT_UNITARY))
    ).resolve()
    output_dir = Path(
        os.environ.get("SYNTHETIQ_OUTPUT_DIR", str(_DEFAULT_SYNTHETIQ_OUTPUT))
    ).resolve()
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.touch(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "docker", "run", "-d",  # <- run detached (non-interactive)
        "-v", f"{input_file}:/usr/synthetiq/data/input/unitary.txt",
        "-v", f"{output_dir}:/usr/synthetiq/data/output",
        "--name", container,
        image,
        "sleep", "infinity"  # Keeps the container alive for exec later
    ], check=True)

def execute_main_in_docker(threshold, input_filename = "unitary.txt"):
    container = os.environ.get("SYNTHETIQ_CONTAINER", "synthetiq_container")
    height = os.environ.get("SYNTHETIQ_QUBITS", "12")
    t_budget = os.environ.get("SYNTHETIQ_T_BUDGET", "16000")
    subprocess.run([
        "docker", "exec", container,
        "./bin/main", input_filename,
        "-h", height,
        "-t", t_budget,
        # "-c", "20",
        "-eps", str(threshold)
    ],
    # stdout=subprocess.DEVNULL
    )

def clear_unitary_output():
    unitary_path = Path(
        os.environ.get("SYNTHETIQ_OUTPUT_DIR", str(_DEFAULT_SYNTHETIQ_OUTPUT))
    ) / "unitary"

    if not unitary_path.exists():
        print(f"Directory not found: {unitary_path}")
        return

    for entry_path in unitary_path.iterdir():
        try:
            if entry_path.is_file() or entry_path.is_symlink():
                os.remove(entry_path)
            elif entry_path.is_dir():
                shutil.rmtree(entry_path)
        except Exception as e:
            print(f"Failed to delete {entry_path}: {e}")

def load_all_qasm_circuits():
    output_dir = Path(
        os.environ.get("SYNTHETIQ_OUTPUT_DIR", str(_DEFAULT_SYNTHETIQ_OUTPUT))
    )
    unitary_dir = output_dir / "unitary"


    if not unitary_dir.exists():
        print(f"Directory not found: {unitary_dir}")
        return {}

    circuits = []

    for file in unitary_dir.glob("*.qasm"):
        try:
            qc = QuantumCircuit.from_qasm_file(str(file))
            # qc = transpile(qc, basis_gates=target_basis, optimization_level=1)
            circuits.append(qc)
            # print(f"✅ Loaded circuit: {file.name}")
        except (QiskitError, OSError) as e:
            print(f"❌ Failed to load {file.name}: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error with {file.name}: {e}")

    return circuits

def write_unitary_to_file(circuit: QuantumCircuit, output_file=_DEFAULT_UNITARY):
    # Compute the unitary matrix
    unitary = Operator(circuit).data  # complex numpy array
    dim = unitary.shape[0]
    n_qubits = int(np.log2(dim))  # Number of qubits

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = ARTIFACT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("matrix\n")
        f.write(f"{n_qubits}\n")

        # Write each matrix row in (real,imag) format
        for row in unitary:
            line = ' '.join(f"({v.real:.6f},{v.imag:.6f})" for v in row)
            f.write(line + "\n")

        # Write the padding block of 1s with dynamic size
        for _ in range(dim):
            f.write("1 " * dim + "\n")

def count_t_like_gates(circuit: QuantumCircuit) -> int:
    """Count T and Tdg gates."""
    return sum(1 for instr, _, _ in circuit.data if instr.name.lower() in {"t", "tdg"})

def count_total_gates(circuit: QuantumCircuit) -> int:
    return len(circuit.data)

def select_least_T_circuit(circuits: list[QuantumCircuit]) -> QuantumCircuit:
    if not circuits:
        raise ValueError("Circuit list is empty.")

    # Sort by T+Tdg gate count, then by total gate count
    sorted_circuits = sorted(
        circuits,
        key=lambda c: (count_t_like_gates(c), count_total_gates(c))
    )

    return sorted_circuits[0]
