"""Translate Qiskit circuits to and from the T-Optimizer IR.

The upstream T-Optimizer repository does not expose a Qiskit adapter.  Its
``.qc`` format is deliberately small, so this module keeps the translation in
one place and avoids importing the optional optimizer at module import time.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import os
import sys
from typing import Any


_QISKIT_TO_TOPTIMIZER = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "S*",
    "t": "T",
    "tdg": "T*",
    "cx": "CX",
    "cnot": "CX",
    "cz": "CZ",
    "swap": "SWAP",
}


def qiskit_to_toptimizer(qc: Any, path: str | Path) -> Path:
    """Write ``qc`` in the ``.qc`` format consumed by T-Optimizer.

    T-Optimizer works on Clifford+T circuits.  Rotation gates, measurements,
    barriers, and conditionals are rejected instead of being silently
    dropped.  The caller should synthesize RZ gates before using this helper.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    qubits = [f"q{i}" for i in range(qc.num_qubits)]

    with destination.open("w", encoding="utf-8") as handle:
        handle.write(".v " + " ".join(qubits) + "\n")
        handle.write(".i " + " ".join(qubits) + "\n")
        handle.write(".o " + " ".join(qubits) + "\n\n")
        handle.write("BEGIN\n")

        for item in qc.data:
            instruction = item.operation if hasattr(item, "operation") else item[0]
            qargs = item.qubits if hasattr(item, "qubits") else item[1]
            name = instruction.name.lower()
            try:
                optimizer_name = _QISKIT_TO_TOPTIMIZER[name]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported gate for T-Optimizer translation: {name!r}. "
                    "Synthesize RZ gates and remove measurements/barriers first."
                ) from error

            indices = [qc.find_bit(qubit).index for qubit in qargs]
            targets = " ".join(qubits[index] for index in indices)
            handle.write(f"{optimizer_name} {targets}\n")

        handle.write("END\n")

    return destination


def _optimizer_root(root: str | Path | None) -> Path:
    configured = root or os.environ.get("T_OPTIMIZER_ROOT")
    if not configured:
        raise RuntimeError(
            "T-Optimizer is not configured. Clone "
            "https://github.com/iqubit-org/T-Optimizer and pass "
            "--t-optimizer-root /path/to/T-Optimizer (or set "
            "T_OPTIMIZER_ROOT)."
        )

    path = Path(configured).expanduser().resolve()
    if path.name == "optimize" and not (path / "optimize").exists():
        path = path.parent
    if not (path / "optimize").exists():
        raise RuntimeError(
            f"T-Optimizer root {path} does not contain an optimize/ package. "
            "Pass the root of the cloned repository."
        )
    return path


def _append_gate(qc: Any, name: str, targets: tuple[int, ...]) -> None:
    """Append one T-Optimizer gate to a Qiskit circuit."""

    upper = name.upper()
    if upper in {"H", "X", "Y", "Z"} and len(targets) == 1:
        getattr(qc, upper.lower())(targets[0])
    elif upper == "T" and len(targets) == 1:
        qc.t(targets[0])
    elif upper == "T*" and len(targets) == 1:
        qc.tdg(targets[0])
    elif upper in {"P", "S"} and len(targets) == 1:
        qc.s(targets[0])
    elif upper in {"P*", "S*"} and len(targets) == 1:
        qc.sdg(targets[0])
    elif upper in {"CX", "CNOT"} and len(targets) == 2:
        qc.cx(targets[0], targets[1])
    elif upper == "CZ" and len(targets) == 2:
        qc.cz(targets[0], targets[1])
    elif upper == "SWAP" and len(targets) == 2:
        qc.swap(targets[0], targets[1])
    elif upper in {"CCZ", "Z"} and len(targets) == 3:
        qc.ccz(targets[0], targets[1], targets[2])
    elif upper == "TOF" and len(targets) == 3:
        qc.ccx(targets[0], targets[1], targets[2])
    elif upper == "TOF" and len(targets) == 1:
        # This is the spelling emitted by T-Optimizer for a one-qubit X in
        # some of its legacy benchmark files.
        qc.x(targets[0])
    else:
        raise ValueError(f"Unsupported gate in T-Optimizer output: {name} {targets}")


def optimizer_to_qiskit(circuit: dict[str, Any]) -> Any:
    """Convert a parsed T-Optimizer circuit dictionary to Qiskit."""

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(len(circuit["qubits"]))
    for name, targets in circuit.get("gates", []):
        _append_gate(qc, name, tuple(targets))
    return qc


def optimize_with_toptimizer(
    qc: Any,
    root: str | Path | None,
    ir_path: str | Path,
) -> Any:
    """Run the upstream T-Optimizer duplicate-removal pass on ``qc``.

    This follows the upstream ``optimize.benchmark`` flow exactly:
    ``from_QC_file``, ``get_T_paulis``, ``remove_duplicates`` and finally
    ``remove_T_gates``.  The optimized circuit is returned as a Qiskit
    circuit so the artifact runner can use one metrics implementation for all
    methods.
    """

    optimizer_path = _optimizer_root(root)
    ir_file = qiskit_to_toptimizer(qc, ir_path)
    path_text = str(optimizer_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

    try:
        optimizer = importlib.import_module("optimize.T_optimizer")
    except Exception as error:  # pragma: no cover - depends on external repo
        raise RuntimeError(
            "T-Optimizer could not be imported. Install its QuaEC, gmpy2, "
            "Cython, and pyximport dependencies in the environment used for "
            f"the artifact run ({optimizer_path})."
        ) from error

    with ir_file.open(encoding="utf-8") as handle:
        parsed = optimizer.from_QC_file(handle)
    t_paulis, clifford = optimizer.get_T_paulis(parsed)
    kept, duplicates = optimizer.remove_duplicates(t_paulis, clifford, indices=True)
    optimized = optimizer.remove_T_gates(parsed, kept, duplicates)
    return optimizer_to_qiskit(optimized)
