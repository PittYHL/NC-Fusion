"""PyZX adapter used by the optimizer-comparison artifact.

This is the cleaned equivalent of the available legacy ``pyzx_path.py``
script.  The artifact workflow passes it the GridSynth Clifford+T circuit;
PyZX may introduce RZ phases while extracting a reduced circuit, so only
exact eighth-turn phases are converted back to Clifford+T gates, matching the
legacy script.
"""

from __future__ import annotations

import math
from typing import Any


def _remove_extracted_swaps(qasm: str) -> str:
    """Match 2025summer/pyzx_path.py, which drops extracted swap lines."""

    return "\n".join(
        line for line in qasm.splitlines() if not line.lstrip().startswith("swap ")
    )


def convert_pyzx_rz_to_clifford_t(circuit: Any) -> Any:
    """Convert only exact eighth-turn RZ phases, as in pyzx_path.py."""

    from qiskit import QuantumCircuit

    converted = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    eighth_turn = math.pi / 4
    for item in circuit.data:
        instruction = item.operation if hasattr(item, "operation") else item[0]
        qargs = item.qubits if hasattr(item, "qubits") else item[1]
        cargs = item.clbits if hasattr(item, "clbits") else item[2]
        q_indices = [circuit.find_bit(qubit).index for qubit in qargs]
        c_indices = [circuit.find_bit(bit).index for bit in cargs]

        if instruction.name.lower() != "rz":
            converted.append(instruction.copy(), q_indices, c_indices)
            continue

        angle = float(instruction.params[0]) % (2 * math.pi)
        k = round(angle / eighth_turn) % 8
        if not math.isclose(angle, k * eighth_turn, abs_tol=1e-9):
            converted.append(instruction.copy(), q_indices, c_indices)
            continue

        qubit = q_indices[0]
        if k == 1:
            converted.t(qubit)
        elif k == 2:
            converted.s(qubit)
        elif k == 3:
            converted.s(qubit)
            converted.t(qubit)
        elif k == 4:
            converted.z(qubit)
        elif k == 5:
            converted.z(qubit)
            converted.t(qubit)
        elif k == 6:
            converted.sdg(qubit)
        elif k == 7:
            converted.tdg(qubit)

    return converted


def _load_extracted_without_swaps(qasm: str) -> Any:
    from qiskit import qasm2

    return qasm2.loads(_remove_extracted_swaps(qasm))


def optimize(circuit: Any) -> tuple[Any, dict[str, Any]]:
    """Run PyZX full reduction and return ``(qiskit_circuit, stats)``."""

    from qiskit import qasm2
    import pyzx as zx

    zx_circuit = zx.Circuit.from_qasm(qasm2.dumps(circuit))
    graph = zx_circuit.to_graph()
    zx.full_reduce(graph)
    extracted = zx.extract_circuit(graph)
    optimized = _load_extracted_without_swaps(extracted.to_qasm())

    try:
        stats = dict(extracted.stats())
    except (AttributeError, TypeError, ValueError):
        stats = {}
    return optimized, stats
