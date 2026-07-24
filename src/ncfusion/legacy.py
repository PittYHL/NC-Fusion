"""Adapter from the public CLI to the original research modules.

No research module is imported at module import time.  This prevents the
legacy scripts' optional dependencies and import-time side effects from
breaking artifact discovery or the smoke test.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "legacy"
for import_root in (LEGACY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _load_qiskit_stack():
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.hamiltonians import HeisenbergModel, IsingModel
    from qiskit_nature.second_q.hamiltonians.lattices import HyperCubicLattice, LineLattice, SquareLattice
    from qiskit_nature.second_q.hamiltonians.lattices.boundary_condition import BoundaryCondition
    from qiskit_nature.second_q.mappers import JordanWignerMapper, LogarithmicMapper
    from qiskit.quantum_info import SparsePauliOp

    return {
        "QuantumCircuit": QuantumCircuit,
        "transpile": transpile,
        "PauliEvolutionGate": PauliEvolutionGate,
        "PySCFDriver": PySCFDriver,
        "HeisenbergModel": HeisenbergModel,
        "IsingModel": IsingModel,
        "HyperCubicLattice": HyperCubicLattice,
        "LineLattice": LineLattice,
        "SquareLattice": SquareLattice,
        "BoundaryCondition": BoundaryCondition,
        "JordanWignerMapper": JordanWignerMapper,
        "LogarithmicMapper": LogarithmicMapper,
        "SparsePauliOp": SparsePauliOp,
    }


def _molecule(name: str, stack: dict[str, Any]):
    driver_cls = stack["PySCFDriver"]
    mapper = stack["JordanWignerMapper"]()
    geometries = {
        "H2": "H 0 0 0; H 0 0 0.735",
        "LiH": "Li 0 0 0; H 0 0 1.5",
        "H2O": "O 0 0 0; H 0.755 0.588 0; H -0.755 0.588 0",
        "N2": "N 0 0 0; N 0 0 1.097",
        "H2S": "S 0 0 0; H 0.7586 0 0.5846; H -0.7586 0 0.5846",
        "CO2": "O 0 0 -1.160; C 0 0 0; O 0 0 1.160",
        "MgO": "Mg 0 0 0; O 0 0 1.749",
        "NaCl": "Na 0 0 0; Cl 0 0 2.361",
    }
    driver = driver_cls(atom=geometries[name], basis="sto3g", charge=0, spin=0)
    problem = driver.run()
    return mapper.map(problem.hamiltonian.second_q_op())


def _lattice_benchmark(spec, stack: dict[str, Any]):
    boundary = stack["BoundaryCondition"].OPEN
    if spec.family == "Ising":
        model_cls = stack["IsingModel"]
    else:
        model_cls = stack["HeisenbergModel"]

    if spec.structure.startswith("2D"):
        rows, cols = (5, 6) if spec.qubits == 30 else (6, 10)
        lattice = stack["SquareLattice"](rows=rows, cols=cols, boundary_condition=boundary)
    else:
        factors = {30: (2, 3, 5), 60: (3, 4, 5)}
        lattice = stack["HyperCubicLattice"](size=factors[spec.qubits], boundary_condition=boundary)

    if spec.family == "Ising":
        model = model_cls(lattice.uniform_parameters(uniform_interaction=-1.0, uniform_onsite_potential=-0.5))
    else:
        model = model_cls(lattice, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    return stack["LogarithmicMapper"]().map(model.second_q_op())


def _small_hamiltonian(spec, stack: dict[str, Any]):
    if spec.name == "H2":
        return _molecule("H2", stack)
    if spec.name == "LiH-reduced":
        # The reduced-qubit workload in the paper is a small Hamiltonian used
        # by the optimizer comparison.  Keep its construction explicit and
        # local rather than depending on a serialized binary matrix.
        return stack["SparsePauliOp"].from_list([
            ("IIII", -0.042), ("ZIII", 0.177), ("IZII", 0.177),
            ("IIZI", -0.242), ("IIIZ", -0.242), ("ZZII", 0.122),
            ("IIZZ", 0.122), ("XZXI", 0.167), ("IXZX", 0.167),
        ])
    lattice = stack["LineLattice"](num_nodes=spec.qubits, boundary_condition=stack["BoundaryCondition"].OPEN)
    model = stack["HeisenbergModel"](lattice, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    return stack["LogarithmicMapper"]().map(model.second_q_op())


def build_hamiltonian(spec):
    stack = _load_qiskit_stack()
    if spec.family == "Molecule":
        return _molecule(spec.name, stack)
    if spec.family == "Hamiltonian":
        return _small_hamiltonian(spec, stack)
    return _lattice_benchmark(spec, stack)


def _metrics(circuit) -> dict[str, int]:
    operations = list(circuit.data)
    t_names = {"t", "tdg"}
    t_count = sum(item.operation.name in t_names for item in operations)
    clifford_count = len(operations) - t_count
    # Qiskit's filtered depth is the convention used by the original scripts.
    t_depth = int(
        circuit.depth(lambda item: item[0].name == "t" or item[0].name == "tdg")
    )
    return {
        "t_count": int(t_count),
        "t_depth": t_depth,
        "clifford_count": int(clifford_count),
        "gate_count": len(operations),
    }


def _grouped_ncf_inputs(hamiltonian, budget: int, window: int, order_seed: int | None = None):
    import copy
    import numpy as np
    from circuit_generation_greedy import greedy_circuit_generation, new_paulis_transform
    from grouping import grouping, permute_keys_after_weight_sort
    from reorder import reorder_pauli_groups

    pauli_strings = list(hamiltonian.paulis.to_labels())
    coeffs = hamiltonian.coeffs
    identity = "I" * hamiltonian.num_qubits
    non_identity = [index for index, label in enumerate(pauli_strings) if label != identity]
    pauli_strings = [pauli_strings[index] for index in non_identity]
    coeffs = coeffs[non_identity]
    if order_seed is not None:
        # ``2025summer/random_order.py`` uses the legacy NumPy global RNG and
        # ``np.random.permutation``.  Keep that behavior for the artifact
        # random-order evaluation so a recorded seed describes the same
        # Pauli-order construction as the source script.
        np.random.seed(order_seed)
        order = np.random.permutation(len(pauli_strings))
        pauli_strings = [pauli_strings[index] for index in order]
        coeffs = coeffs[order]
    pauli_list = {label: coeffs[index].real for index, label in enumerate(pauli_strings)}
    # The primary NC-Fusion path in main_alg.py passes only ``group`` to the
    # compressor. ``no_commute_group`` is reserved for the component-ablation
    # variants; including it here doubles the reported unitary count.
    group, _no_commute_group = grouping(pauli_strings, budget, window, use_window=True)
    new_paulis, commute_paulis, circuits = [], [], []
    for subgroup in group:
        permuted = permute_keys_after_weight_sort(copy.deepcopy(subgroup))
        transformed, circuit, signs = greedy_circuit_generation(permuted[0])
        new, commute = new_paulis_transform(pauli_list, transformed, permuted[0], signs)
        new_paulis.append(new)
        commute_paulis.append(commute)
        circuits.append(circuit)
    return reorder_pauli_groups(new_paulis, commute_paulis, circuits), len(pauli_strings)


def _single_qubit_ncf(hamiltonian, spec, settings, gpu: int, *, save_qasm: bool):
    from compressor import compressor_circuit, ncf_unitaries_generated

    (new_paulis, commute_paulis, circuits), pauli_count = _grouped_ncf_inputs(
        hamiltonian,
        1,
        int(settings.get("single_window", 4)),
        settings.get("pauli_order_seed"),
    )
    rz_qc, synthesized = compressor_circuit(
        new_paulis,
        commute_paulis,
        circuits,
        float(settings.get("synthesis_error", 0.001)),
        1,
        hamiltonian.num_qubits,
        gpu=gpu,
        num_paulis=pauli_count,
        trotter_steps=int(settings.get("trotter_steps", 1)),
        evolution_time=float(settings.get("evolution_time", 1.0)),
        synthesize=True,
        benchmark=spec.name if save_qasm else None,
        fix_error_threshold=0,
        t_budget=int(settings.get("t_budget", 60)),
    )
    return rz_qc, synthesized, ncf_unitaries_generated(new_paulis, commute_paulis)


def _two_qubit_ncf(hamiltonian, spec, settings, *, save_qasm: bool):
    from compressor import synthetiq_compressor

    (new_paulis, commute_paulis, circuits), pauli_count = _grouped_ncf_inputs(
        hamiltonian, 2, int(settings.get("two_window", 128)), settings.get("pauli_order_seed")
    )
    unitary_count = sum(1 for item in new_paulis if item)
    rz_count = sum(len(item) for item in commute_paulis)
    circuit = synthetiq_compressor(
        new_paulis,
        commute_paulis,
        circuits,
        float(settings.get("two_qubit_error", settings.get("synthesis_error", 0.12))),
        2,
        hamiltonian.num_qubits,
        num_paulis=pauli_count,
        normalized=int(settings.get("ncf_two_normalized", 1)),
    )
    if circuit is None:
        raise RuntimeError("Synthetiq did not return a circuit")
    if save_qasm:
        from qiskit import qasm2
        from paths import circuit_path

        circuit_path(spec.name, "ncf-two_c+t").write_text(
            qasm2.dumps(circuit), encoding="utf-8"
        )
    return circuit, unitary_count, rz_count


def _count_gates(circuit, names: set[str]) -> int:
    return sum(
        1
        for item in circuit.data
        if (item.operation if hasattr(item, "operation") else item[0]).name.lower() in names
    )


def _original_rz_gate_count(hamiltonian) -> int:
    """Match ``main_alg.py:159``: count non-identity Pauli terms."""

    identity = "I" * hamiltonian.num_qubits
    return sum(label != identity for label in hamiltonian.paulis.to_labels())


def run_benchmark(
    spec,
    method: str,
    settings: dict[str, object],
    gpu: int,
    *,
    save_qasm: bool = True,
) -> dict[str, object]:
    from baseline import baseline_circuit
    from paths import circuit_path

    start = time.perf_counter()
    hamiltonian = build_hamiltonian(spec)
    effective_settings = dict(settings)
    if effective_settings.get("window") == "full":
        window_key = "single_window" if method == "ncf-one" else "two_window"
        effective_settings[window_key] = len(hamiltonian.paulis)
    common = {
        "Trotter_steps": int(effective_settings.get("trotter_steps", 1)),
        "evolution_time": float(effective_settings.get("evolution_time", 1.0)),
        "synthesize": True,
        "gpu": gpu,
        "error_threshold": float(effective_settings.get("synthesis_error", 0.001)),
    }
    original_rz_qc = None
    ncf_rz_qc = None
    ncf_unitary_count = None
    ncf_rz_count = None
    if method == "gridsyn":
        rz_qc, circuit = baseline_circuit(
            hamiltonian, 1, GRIDSYNTH=True, rustiq=False,
            benchmark=spec.name if save_qasm else None, method="grid", **common
        )
        original_rz_qc = rz_qc
    elif method == "rustiq":
        rz_qc, circuit = baseline_circuit(
            hamiltonian, 1, GRIDSYNTH=False, rustiq=True,
            benchmark=spec.name if save_qasm else None, method="rustiq", **common
        )
        original_rz_qc = rz_qc
    elif method == "ncf-one":
        original_rz_qc, _ = baseline_circuit(
            hamiltonian, 1, GRIDSYNTH=True, rustiq=False,
            benchmark=None, method="grid", **{**common, "synthesize": False}
        )
        ncf_rz_qc, circuit, ncf_unitary_count = _single_qubit_ncf(
            hamiltonian, spec, effective_settings, gpu, save_qasm=save_qasm
        )
        if save_qasm and original_rz_qc is not None:
            from qiskit import qasm2

            circuit_path(spec.name, "grid_rz").write_text(
                qasm2.dumps(original_rz_qc), encoding="utf-8"
            )
    elif method == "ncf-two":
        original_rz_qc, _ = baseline_circuit(
            hamiltonian, 1, GRIDSYNTH=True, rustiq=False,
            benchmark=None, method="grid", **{**common, "synthesize": False}
        )
        circuit, ncf_unitary_count, ncf_rz_count = _two_qubit_ncf(
            hamiltonian, spec, effective_settings, save_qasm=save_qasm
        )
        if save_qasm and original_rz_qc is not None:
            from qiskit import qasm2

            circuit_path(spec.name, "grid_rz").write_text(
                qasm2.dumps(original_rz_qc), encoding="utf-8"
            )
    elif method == "phoenix":
        from baseline import phoenix_baseline_circuit
        labels = list(hamiltonian.paulis.to_labels())
        _, circuit = phoenix_baseline_circuit(hamiltonian, labels, hamiltonian.coeffs, **common)
    else:
        raise ValueError(
            f"Method {method!r} is not available in the adapter. "
            "Use gridsyn, rustiq, ncf-one, or phoenix; ncf-two requires the Synthetiq container."
        )

    metrics = _metrics(circuit)
    metrics.update({"benchmark": spec.name, "method": method, "qubits": spec.qubits})
    runtime = round(time.perf_counter() - start, 4)
    metrics["runtime_seconds"] = runtime
    metrics["compilation_time_seconds"] = runtime
    metrics["data_source"] = "generated"
    if original_rz_qc is not None:
        metrics["original_rz_gate_count"] = _original_rz_gate_count(hamiltonian)
    if ncf_unitary_count is not None:
        metrics["ncf_unitaries_generated"] = ncf_unitary_count
    if method == "ncf-two" and ncf_rz_count is not None:
        metrics["ncf_rz_generated"] = ncf_rz_count
    if save_qasm:
        suffixes = {"gridsyn": "grid_c+t", "rustiq": "rustiq_c+t", "ncf-one": "ncf_c+t", "ncf-two": "ncf-two_c+t"}
        if method in suffixes:
            metrics["qasm_path"] = str(circuit_path(spec.name, suffixes[method]))
            if original_rz_qc is not None:
                metrics["rz_qasm_path"] = str(circuit_path(spec.name, "grid_rz"))
    return metrics
