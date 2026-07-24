from commuting_graph import commute_check

import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, TGate, XGate, YGate, ZGate, PauliEvolutionGate
from qiskit.compiler import transpile
from qiskit.qasm2 import dump, load
from qiskit.quantum_info import state_fidelity, Statevector, Operator
from qiskit.synthesis.evolution import MatrixExponential
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesSimpleCommutation
from qiskit_aer import AerSimulator, StatevectorSimulator, UnitarySimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import SparsePauliOp

from numpy.typing import NDArray
import time
from typing import Dict, Tuple

logical_errors = [1e-6, 1e-5, 1e-4]
# Use CPU for Aer simulators; GPU not supported on this system.
gpu = 0

# Cache exact evolution results keyed by (id(hamiltonian), time) to avoid
# recomputing expensive matrix exponentials across repeated calls.
_exact_evolution_cache: Dict[Tuple[int, float], Dict[str, object]] = {}

def trotter_pauli(pauli_list):
    trotter_pauli_list = []
    trotter_coeffs = []
    for pauli, coeff in pauli_list.items():
        trotter_pauli_list.append([pauli])
        trotter_coeffs.append([coeff])
    return trotter_pauli_list, trotter_coeffs

def trotter_pauli_group(group, pauli_list):
    trotter_pauli_list = []
    trotter_coeffs = []
    for sub in group:
        pauli_strings = sub[1] + sub[2] + sub[3] + sub['dependent']
        trotter_pauli_list.append(pauli_strings)
        if sub['commute'] != []:
            for pauli in sub['commute']:
                trotter_pauli_list.append([pauli])
                
    for paulis in trotter_pauli_list:
        coeffs = []
        for pauli in paulis:
            coeffs.append(pauli_list[pauli])
        trotter_coeffs.append(coeffs)
    return trotter_pauli_list, trotter_coeffs

def trotter_error(paulis_list, coeffs_list, time, steps):
    factor = pow(time, 2) / (2 * steps)
    constant = pow(time, 3) / pow(steps, 2)
    error = 0
    non_commute_pairs = 0
    for i in range(len(paulis_list)):
        pauli_strings = paulis_list[i]  
        coeffs = coeffs_list[i]
        latter_pauli_string = []
        latter_coeff = []
        for k in range(i + 1, len(paulis_list)):
                latter_pauli_string = latter_pauli_string + paulis_list[k]
                latter_coeff = latter_coeff + coeffs_list[k]
        for j in range(len(pauli_strings)):
            pauli_string = pauli_strings[j]
            coeff = coeffs[j]
            for k in range(len(latter_pauli_string)):
                if commute_check(pauli_string, latter_pauli_string[k]) == False:
                    non_commute_pairs += 1
                    error = error + abs(2 * coeff * latter_coeff[k])
    error = error * factor + constant
    return error, non_commute_pairs


def trotter_statevector_fidelity(hamiltonian, trotter_steps: int, time: float = 1.0) -> float:
    """
    Numerically compare the exact evolution under `hamiltonian` with a first-order
    Trotterized evolution using `trotter_steps` steps, via state-vector fidelity.

    Args:
        hamiltonian: Qiskit SparsePauliOp-style Hamiltonian.
        trotter_steps: Number of Trotter steps to use (must be >= 1).
        time: Total evolution time.

    Returns:
        The state fidelity between the exact and Trotter-evolved statevectors.
    """
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1.")

    num_qubits = hamiltonian.num_qubits

    # Include synthesis method in cache key since it changes the exact unitary.
    cache_key = (id(hamiltonian), float(time), "MatrixExponential")
    cached = _exact_evolution_cache.get(cache_key)
    if cached is None:
        cached = {}
        _exact_evolution_cache[cache_key] = cached

    # Exact evolution gate and circuit U = exp(-i H t)
    # Important: by default, PauliEvolutionGate uses a product-formula synthesis
    # (typically Lie-Trotter with reps=1). For an *exact* reference unitary exp(-i H t),
    # use MatrixExponential synthesis.
    exact_gate = PauliEvolutionGate(hamiltonian, time=time, synthesis=MatrixExponential())
    exact_qc = QuantumCircuit(num_qubits)
    exact_qc.append(exact_gate, exact_qc.qubits)

    # First-order Trotterized circuit, decomposed to the target basis.
    trotter_qc = QuantumCircuit(num_qubits)
    dt = time / trotter_steps
    pauli_labels = hamiltonian.paulis.to_labels()
    coeffs = hamiltonian.coeffs
    target_basis = ["cx", "h", "s", "rz", "sdg"]

    for _ in range(trotter_steps):
        for label, coeff in zip(pauli_labels, coeffs):
            if np.isclose(coeff, 0.0):
                continue
            term_op = hamiltonian.__class__.from_list([(label, coeff)])
            # Build a small circuit for this term and transpile it to the target basis.
            term_qc = QuantumCircuit(num_qubits)
            term_qc.append(PauliEvolutionGate(term_op, time=dt), term_qc.qubits)
            term_qc = transpile(term_qc, basis_gates=target_basis, optimization_level=1)
            trotter_qc = trotter_qc.compose(term_qc)

    # Use analytic statevector simulation from qiskit.quantum_info to avoid
    # backend support issues for PauliEvolutionGate.
    exact_sv = cached.get("exact_sv")
    if exact_sv is None:
        exact_sv = Statevector.from_instruction(exact_qc)
        cached["exact_sv"] = exact_sv
    trotter_sv = Statevector.from_instruction(trotter_qc)
    fidelity = state_fidelity(exact_sv, trotter_sv)

    # Compute unitary distance. For the "exact" unitary, use Operator on the
    # PauliEvolutionGate directly (full 2^n x 2^n matrix exponential; expensive).

    # For the trotterized circuit, all operations are transpiled to a supported basis,
    # so Aer can efficiently extract the unitary.

    print("Trotter statevector fidelity:", fidelity)
    return fidelity

def trace(mat1: NDArray[np.complex128], mat2: NDArray[np.complex128]) -> float:
    return min(np.abs(np.trace(mat1 @ mat2.conj().T) / mat1.shape[0]), 1)

def distance(mat1: NDArray[np.complex128], mat2: NDArray[np.complex128]) -> float:
    return np.sqrt(1 - trace(mat1, mat2) ** 2)

def density_matrix_error(qc, ft_qc):
    noisy_fidelity, fidelity = None, None
    if qc.num_qubits <= 27:
        simulator = StatevectorSimulator(device="GPU" if gpu else "CPU")
        ideal_sv = simulator.run(qc).result().get_statevector()
        fidelity = state_fidelity(
            ideal_sv,
            simulator.run(ft_qc).result().get_statevector(),
        )
    if qc.num_qubits <= 12:
        if logical_errors:
            noisy_fidelity = {}
        for logical_error in logical_errors:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(logical_error, 1), ["s", "h", "sdg"]
            )
            noise_model.add_all_qubit_quantum_error(depolarizing_error(logical_error * 10, 1), ["t", "tdg"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(logical_error, 2), ["cx"])
            noisy_qc = ft_qc.copy()
            noisy_qc.save_density_matrix()
            noisy_op = (
                AerSimulator(
                    method="density_matrix",
                    noise_model=noise_model,
                    device="GPU" if gpu else "CPU",
                )
                .run(noisy_qc)
                .result()
                .data(0)["density_matrix"]
            )
            noisy_fidelity[logical_error] = state_fidelity(noisy_op, ideal_sv)
    print('fidelity: ', fidelity)
    print('noisy_fidelity: ', noisy_fidelity)

def density_matrix_error_from_hamiltonian(hamiltonian, ft_qc, time: float, gpu: bool = False, logical_errors=None, ideal_sv = None):
    """
    hamiltonian: qubit operator (typically SparsePauliOp) representing H
    ft_qc: trotter circuit approximating exp(-i H time) applied to |0...0> (and any prep inside ft_qc)
    time: evolution time t
    """
    if logical_errors is None:
        logical_errors = []

    noisy_fidelity, fidelity = None, None
    num_qubits = hamiltonian.num_qubits
    # -------------------------
    # 1) EXACT statevector via expm(-i H t)
    # -------------------------
    # Build |psi0> = |0...0> (matches Aer default if ft_qc has no prep)
    if ideal_sv is None:
        psi0 = Statevector.from_label("0" * num_qubits)

        # Dense matrix for H and exact evolution
        # WARNING: exponential memory/time
        if num_qubits <= 12:
            H_mat = Operator(hamiltonian).data
            U_exact = expm(-1j * H_mat * time)
            ideal_sv = Statevector(U_exact @ psi0.data)
        else:
            H = hamiltonian
            if isinstance(H, SparsePauliOp):
                H_mat = H.to_matrix(sparse=True)   # sparse 2^n × 2^n
            else:
                # Fallback (still may densify!)
                H_mat = Operator(H).data
            psi0_vec = psi0.data 
            ideal_vec = expm_multiply((-1j) * H_mat * time, psi0_vec)
            ideal_sv = Statevector(ideal_vec)
    if num_qubits <= 30:
        simulator = StatevectorSimulator(device="GPU" if gpu else "CPU")

        ft_sv = simulator.run(ft_qc).result().get_statevector()

        fidelity = state_fidelity(ideal_sv, ft_sv)

    # ---- Noisy fidelity via density-matrix simulation of ft_qc ----
    if num_qubits <= 12 and logical_errors:
        noisy_fidelity = {}

        for logical_error in logical_errors:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(logical_error, 1), ["h"]
            )
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(2 * logical_error, 1), ["s", "sdg"]
            )
            # keep your original, but note: your target basis earlier didn't include t/tdg
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(6.3*2*logical_error + 4.4e-8, 1), ["t", "tdg"]
              #  depolarizing_error(121*logical_error + 3.5e-8, 1), ["t", "tdg"]
            )
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(3*logical_error, 2), ["cx"]
            )

            noisy_qc = ft_qc.copy()
            noisy_qc.save_density_matrix()

            noisy_dm = (
                AerSimulator(
                    method="density_matrix",
                    noise_model=noise_model,
                    device="GPU" if gpu else "CPU",
                )
                .run(noisy_qc)
                .result()
                .data(0)["density_matrix"]
            )

            # fidelity between noisy density matrix and ideal pure state
            noisy_fidelity[logical_error] = state_fidelity(noisy_dm, ideal_sv)

    print("fidelity:", fidelity)
    if noisy_fidelity is not None:
        print("noisy_fidelity:", noisy_fidelity)
    return fidelity, noisy_fidelity

def trotter_operator_norm_error(H, qc, t, cache_file="U_exact.npy"):
    """
    Compute operator norm error || exp(-i H t) - U_circ ||_2.
    Cache U_exact to disk to avoid recomputing.

    Parameters
    ----------
    H : SparsePauliOp or numpy.ndarray
        Hamiltonian.
    qc : QuantumCircuit
        Circuit implementing approximate evolution.
    t : float
        Evolution time.
    cache_file : str
        File used to store/load U_exact.

    Returns
    -------
    float
        Spectral norm error.
    """

    # Convert Hamiltonian to matrix
    if isinstance(H, SparsePauliOp):
        H_mat = H.to_matrix()
    else:
        H_mat = H

    # Load cached U_exact if available
    if os.path.exists(cache_file):
        print("Loading cached U_exact...")
        U_exact = np.load(cache_file)
    else:
        print("Computing U_exact...")
        U_exact = expm(-1j * H_mat * t)
        np.save(cache_file, U_exact)

    # Circuit unitary.  Aer avoids the very slow Python-side gate-by-gate
    # construction of a dense Operator for large Clifford+T circuits.  Keep
    # the original implementation as a fallback for environments without
    # qiskit-aer.
    try:
        from qiskit_aer import AerSimulator

        unitary_qc = qc.copy()
        unitary_qc.save_unitary()
        result = AerSimulator(method="unitary").run(unitary_qc).result()
        U_circ = np.asarray(result.get_unitary(unitary_qc))
    except ImportError:
        U_circ = Operator(qc).data

    # Remove global phase
    phase = np.angle(np.trace(U_exact.conj().T @ U_circ))
    U_circ = U_circ * np.exp(-1j * phase)

    # Operator norm.  For the large LiH operators, ARPACK computes the
    # largest singular value directly without the prohibitively expensive
    # full dense SVD.  This is still the spectral norm of the same dense
    # difference matrix; use the exact dense fallback for smaller operators
    # or when SciPy's iterative solver is unavailable.
    difference = U_exact - U_circ
    if max(difference.shape) >= 2048:
        try:
            from scipy.sparse.linalg import svds

            singular_values = svds(
                difference,
                k=1,
                which="LM",
                return_singular_vectors=False,
                tol=1e-10,
                maxiter=2000,
            )
            error = float(np.max(singular_values))
        except (ImportError, RuntimeError, ValueError):
            error = np.linalg.norm(difference, ord=2)
    else:
        error = np.linalg.norm(difference, ord=2)
    print('trotter operator norm error: ', error)

    return error
