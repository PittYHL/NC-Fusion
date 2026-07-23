"""Public NC-Fusion compilation API."""

from __future__ import annotations

from typing import Any

from .legacy import _grouped_ncf_inputs


def NC_Fusion(
    hamiltonian: Any,
    *,
    budget: int = 1,
    window: int | None = None,
    error_threshold: float = 0.001,
    t_budget: int = 60,
    gpu: int = 0,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    synthesize: bool = True,
    use_gridsynth: bool = False,
    fix_error_threshold: bool = False,
    pauli_order_seed: int | None = None,
) -> tuple[Any, Any]:
    """Compile a Hamiltonian with NC-Fusion.

    Parameters
    ----------
    hamiltonian:
        A Qiskit ``SparsePauliOp``-compatible Hamiltonian. It must provide
        ``paulis.to_labels()``, ``coeffs``, and ``num_qubits``.
    budget:
        Number of qubits retained by each fused group: ``1`` for single-qubit
        NC-Fusion or ``2`` for two-qubit NC-Fusion.
    window:
        Pauli-grouping window. Defaults to 4 for budget 1 and 128 otherwise.
    error_threshold:
        Total synthesis error target passed to the single-qubit synthesizer.
    t_budget:
        Per-rotation Trasyn T-budget.
    gpu:
        Trasyn GPU flag. Use ``0`` for CPU synthesis and ``1`` to enable GPU
        synthesis. Trasyn currently does not expose a GPU-device index here.
    trotter_steps:
        Number of repeated Trotter steps in each returned circuit.
    evolution_time:
        Total Hamiltonian evolution time.
    synthesize:
        If true, return both the compiled rotation circuit and its
        Clifford+T circuit. If false, skip synthesis and return ``None`` for
        the Clifford+T circuit.
    use_gridsynth:
        Use Qiskit's Clifford+T GridSynth decomposition for rotations instead
        of Trasyn.
    fix_error_threshold:
        Apply ``error_threshold`` independently to each rotation instead of
        normalizing the total error across all rotations.
    pauli_order_seed:
        Optional deterministic seed for shuffling Pauli terms before grouping.

    Returns
    -------
    tuple[qiskit.QuantumCircuit, qiskit.QuantumCircuit | None]
        ``(compiled_qc, clifford_t_qc)``. The first circuit retains RZ/U3
        rotations; the second contains the Clifford+T approximation when
        ``synthesize=True``.
    """
    if budget not in (1, 2):
        raise ValueError("budget must be 1 (single-qubit) or 2 (two-qubit)")
    if window is None:
        window = 4 if budget == 1 else 128
    if window < 1:
        raise ValueError("window must be positive")
    if error_threshold <= 0:
        raise ValueError("error_threshold must be positive")
    if t_budget < 1:
        raise ValueError("t_budget must be positive")
    if gpu not in (0, 1):
        raise ValueError("gpu must be 0 (CPU) or 1 (GPU)")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be positive")
    if not hasattr(hamiltonian, "paulis") or not hasattr(hamiltonian, "coeffs"):
        raise TypeError("hamiltonian must be a Qiskit SparsePauliOp-compatible object")
    if len(hamiltonian.paulis) == 0:
        raise ValueError("hamiltonian must contain at least one Pauli term")

    (new_paulis, commute_paulis, circuits), pauli_count = _grouped_ncf_inputs(
        hamiltonian,
        budget,
        int(window),
        pauli_order_seed,
    )

    # Import optional quantum/synthesis dependencies only when compilation is
    # requested. This keeps `import ncfusion` dependency-light.
    from compressor import compressor_circuit

    compiled_qc, clifford_t_qc = compressor_circuit(
        new_paulis,
        commute_paulis,
        circuits,
        error_threshold,
        budget,
        hamiltonian.num_qubits,
        gpu=int(gpu == 1),
        num_paulis=pauli_count,
        fix_error_threshold=int(fix_error_threshold),
        rz=int(use_gridsynth),
        gridsyn=use_gridsynth,
        trotter_steps=trotter_steps,
        evolution_time=evolution_time,
        synthesize=synthesize,
        t_budget=t_budget,
    )
    return compiled_qc, clifford_t_qc if synthesize else None
