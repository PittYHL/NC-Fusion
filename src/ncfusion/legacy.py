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
    t_depth = int(circuit.depth(lambda item: item[0].name in t_names))
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
    if order_seed is not None:
        order = np.random.default_rng(order_seed).permutation(len(pauli_strings))
        pauli_strings = [pauli_strings[index] for index in order]
        coeffs = coeffs[order]
    pauli_list = {label: coeffs[index].real for index, label in enumerate(pauli_strings)}
    group, no_commute_group = grouping(pauli_strings, budget, window, use_window=True)
    new_paulis, commute_paulis, circuits = [], [], []
    for subgroup in group + no_commute_group:
        permuted = permute_keys_after_weight_sort(copy.deepcopy(subgroup))
        transformed, circuit, signs = greedy_circuit_generation(permuted[0])
        new, commute = new_paulis_transform(pauli_list, transformed, permuted[0], signs)
        new_paulis.append(new)
        commute_paulis.append(commute)
        circuits.append(circuit)
    return reorder_pauli_groups(new_paulis, commute_paulis, circuits), len(pauli_strings)


def _single_qubit_ncf(hamiltonian, spec, settings, gpu: int):
    from compressor import compressor_circuit

    (new_paulis, commute_paulis, circuits), pauli_count = _grouped_ncf_inputs(
        hamiltonian,
        1,
        int(settings.get("single_window", 4)),
        settings.get("pauli_order_seed"),
    )
    _, synthesized = compressor_circuit(
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
        benchmark=spec.name,
        fix_error_threshold=0,
        t_budget=int(settings.get("t_budget", 60)),
    )
    return synthesized


def _two_qubit_ncf(hamiltonian, settings):
    from compressor import synthetiq_compressor

    (new_paulis, commute_paulis, circuits), pauli_count = _grouped_ncf_inputs(
        hamiltonian, 2, int(settings.get("two_window", 128)), settings.get("pauli_order_seed")
    )
    circuit = synthetiq_compressor(
        new_paulis,
        commute_paulis,
        circuits,
        float(settings.get("two_qubit_error", 0.12)),
        2,
        hamiltonian.num_qubits,
        num_paulis=pauli_count,
        normalized=1,
    )
    if circuit is None:
        raise RuntimeError("Synthetiq did not return a circuit")
    return circuit


def run_benchmark(spec, method: str, settings: dict[str, object], gpu: int) -> dict[str, object]:
    from baseline import baseline_circuit

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
    if method == "gridsyn":
        _, circuit = baseline_circuit(hamiltonian, 1, GRIDSYNTH=True, rustiq=False, benchmark=spec.name, method="grid", **common)
    elif method == "rustiq":
        _, circuit = baseline_circuit(hamiltonian, 1, GRIDSYNTH=False, rustiq=True, benchmark=spec.name, method="rustiq", **common)
    elif method == "ncf-one":
        circuit = _single_qubit_ncf(hamiltonian, spec, effective_settings, gpu)
    elif method == "ncf-two":
        circuit = _two_qubit_ncf(hamiltonian, effective_settings)
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
    metrics["runtime_seconds"] = round(time.perf_counter() - start, 4)
    return metrics
