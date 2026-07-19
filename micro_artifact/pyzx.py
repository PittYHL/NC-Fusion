"""PyZX adapter used by the optimizer-comparison artifact.

This is the cleaned equivalent of the available legacy ``pyzx_path.py``
script.  The artifact workflow passes it the GridSynth Clifford+T circuit;
PyZX may introduce RZ phases while extracting a reduced circuit, so the
caller resynthesizes those phases before counting T gates.
"""

from __future__ import annotations

from typing import Any


def _expand_swaps(circuit: Any) -> Any:
    from qiskit import QuantumCircuit

    expanded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for item in circuit.data:
        instruction = item.operation if hasattr(item, "operation") else item[0]
        qargs = item.qubits if hasattr(item, "qubits") else item[1]
        cargs = item.clbits if hasattr(item, "clbits") else item[2]
        indices = [circuit.find_bit(qubit).index for qubit in qargs]
        classical = [circuit.find_bit(bit).index for bit in cargs]
        if instruction.name.lower() == "swap":
            expanded.cx(indices[0], indices[1])
            expanded.cx(indices[1], indices[0])
            expanded.cx(indices[0], indices[1])
        else:
            expanded.append(
                instruction.copy(),
                [expanded.qubits[index] for index in indices],
                [expanded.clbits[index] for index in classical],
            )
    return expanded


def optimize(circuit: Any) -> tuple[Any, dict[str, Any]]:
    """Run PyZX full reduction and return ``(qiskit_circuit, stats)``."""

    from qiskit import qasm2
    import pyzx as zx

    zx_circuit = zx.Circuit.from_qasm(qasm2.dumps(circuit))
    graph = zx_circuit.to_graph()
    zx.full_reduce(graph)
    extracted = zx.extract_circuit(graph)
    optimized = qasm2.loads(extracted.to_qasm())
    optimized = _expand_swaps(optimized)

    try:
        stats = dict(extracted.stats())
    except (AttributeError, TypeError, ValueError):
        stats = {}
    return optimized, stats
