from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit import transpile
import trasyn
from qiskit.qasm2 import dumps
import time
import copy
import subprocess
import numpy as np

from phoenix import Hamiltonian
from phoenix.compiler import compile_hamiltonian_simulation
from qiskit.quantum_info import Clifford


# Patch trasyn.utils.get_available_memory to support multiple GPUs (use first GPU's free memory)
import trasyn.utils
_original_get_available_memory = trasyn.utils.get_available_memory

def _get_available_memory_multi_gpu(gpu: bool = False):
    """Wrapper that handles nvidia-smi output with multiple GPUs (one line per GPU)."""
    if not gpu:
        return _original_get_available_memory(False)
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"]
    ).decode().strip()
    # Multiple GPUs: one line per GPU; take first line to avoid "too many values to unpack"
    first_line = out.split("\n")[0].strip()
    parts = first_line.split()
    memsize = int(parts[0])
    unit = parts[1] if len(parts) > 1 else "MiB"
    if unit == "MiB":
        memsize *= 1024**2
    elif unit == "GiB":
        memsize *= 1024**3
    return memsize

trasyn.utils.get_available_memory = _get_available_memory_multi_gpu
from qiskit_gridsynth_plugin.decompose import clifford_t_transpile
from compressor import rewrite_clifford_rz_u3_gates, normalized_error
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

def is_clifford_angle(theta):
    """Returns True if angle is a multiple of pi/2."""
    theta = float(theta) % (2 * np.pi)
    multiples = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    return any(np.isclose(theta, m, atol=1e-8) for m in multiples)

def snap_clifford_angle(theta, tol=1e-5):
    """Return (is_clifford, snapped_theta) where snapped_theta is the nearest
    multiple of pi/2 if within tol, otherwise the original theta.
    """
    theta = float(theta)
    half_pi = np.pi / 2
    k = round(theta / half_pi)
    snapped = k * half_pi
    if np.isclose(theta, snapped, atol=tol):
        return True, snapped
    return False, theta

def baseline_circuit(hamiltonian, budget, error_threshold, gpu = 0, Trotter_steps = 1, evolution_time = 1, rustiq = False, GRIDSYNTH = False, t_budget = 60, use_trotter = False, synthesize = True, benchmark = None, method = None):
    # t_budget = 60
    dt = evolution_time / Trotter_steps
    target_basis = ['cx', 'h', 's', 'rz', 'sdg']
    num_qubits = hamiltonian.num_qubits
    evolution_gate = PauliEvolutionGate(hamiltonian, time=dt)
    Trotter_qc = QuantumCircuit(num_qubits)
    synthesized_qc = QuantumCircuit(num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.append(evolution_gate, qc.qubits)
    # qc = transpile(qc, basis_gates=target_basis)
    # qc.draw(output='mpl')
    # plt.show()
    if budget == 1 and rustiq == False and GRIDSYNTH == False:  #use trasyn
        qc = transpile(qc, basis_gates=target_basis, optimization_level=1)
        qc = rewrite_clifford_rz_u3_gates(qc)
        if not synthesize:
            for _ in range(Trotter_steps):
                Trotter_qc.compose(qc, inplace=True)
            return Trotter_qc, Trotter_qc
        num_rotations = 0
        new_qc = QuantumCircuit(num_qubits)
        for instruction, qargs, cargs in qc.data:
            if instruction.name == 'h':
                new_qc.h(qargs[0])
            elif instruction.name == 's':
                new_qc.s(qargs[0])
            elif instruction.name == 'sdg':
                new_qc.sdg(qargs[0])
            elif instruction.name == 'rz':
                num_rotations += 1
                para = instruction.params
                seq, _, _ = trasyn.synthesize(para[0], t_budget, error_threshold=error_threshold, gate_set = 'tsh', gpu = gpu)
                for gate in seq:
                    if gate == 'h':
                        new_qc.h(qargs[0])
                    elif gate == 's':
                        new_qc.s(qargs[0])
                    elif gate == 't':
                        new_qc.t(qargs[0])
            elif instruction.name == 'cx':
                new_qc.cx(qargs[0], qargs[1])
        print("Number of rotations:", num_rotations)

    elif budget == 1 and rustiq == True and GRIDSYNTH == False:  #use rustiq
        hls_config = HLSConfig(PauliEvolution=[
            ("rustiq", {"optimize_count": True, "preserve_order": False})
        ])

        target_basis = ['cx', 'h', 's', 'rz', 'sdg', 'u3']
        # target_basis = ['cx', 'h', 's', 'sdg', 'u3']
        qc = transpile(qc, hls_config=hls_config, basis_gates=target_basis, optimization_level=1)
        if benchmark is not None:
            with open("circuits/" + benchmark + "_" + "rustiq" + "_RZ.qasm", "w") as f:
                f.write(dumps(qc))

        if not synthesize:
            for _ in range(Trotter_steps):
                Trotter_qc.compose(qc, inplace=True)
            return Trotter_qc, Trotter_qc
        # num_paulis = len(hamiltonian) - 1
        num_paulis = len(hamiltonian)
        copy_qc = copy.deepcopy(qc)
        copy_qc = rewrite_clifford_rz_u3_gates(copy_qc)
        normal_error = normalized_error(copy_qc, error_threshold, num_paulis)
        print("Normal error:", normal_error)
        num_rotations = 0
        new_qc = QuantumCircuit(num_qubits)
        for instruction, qargs, cargs in qc.data:
            if instruction.name == 'h':
                new_qc.h(qargs[0])
            elif instruction.name == 's':
                new_qc.s(qargs[0])
            elif instruction.name == 'sdg':
                new_qc.sdg(qargs[0])
            # elif instruction.name == 'rz':
            #     num_rotations += 1
            #     para = instruction.params
            #     is_cliff, snapped_theta = snap_clifford_angle(para[0])
            #     if is_cliff:
            #         temp_circ = QuantumCircuit(1)
            #         temp_circ.rz(snapped_theta, 0)
            #         cl = Clifford(temp_circ)
            #         temp_circ = cl.to_circuit()
            #         temp_circ = transpile(temp_circ, basis_gates=['h', 's', 'sdg'], optimization_level=1)
            #         for new_instruction, _, _ in temp_circ.data:
            #             if new_instruction.name == 'h':
            #                 new_qc.h(qargs[0])
            #             elif new_instruction.name == 's':
            #                 new_qc.s(qargs[0])
            #             elif new_instruction.name == 'sdg':
            #                 new_qc.sdg(qargs[0])
            #     else:
            #         seq, _, _ = trasyn.synthesize(para[0], t_budget, error_threshold=normal_error, gate_set = 'tsh', gpu = gpu)
            #         thresholds.append(err)
            #         found.append(seq)
            #         for gate in seq:
            #             if gate == 'h':
            #                 new_qc.h(qargs[0])
            #             elif gate == 's':
            #                 new_qc.s(qargs[0])
            #             elif gate == 't':
            #                 new_qc.t(qargs[0])
            elif instruction.name == 'rz':
                num_rotations += 1
                para = instruction.params
                temp_circ = QuantumCircuit(1)
                temp_circ.rz(para[0], 0)
                decomposed = clifford_t_transpile(temp_circ, epsilon=error_threshold)
                for new_instruction, _, _ in decomposed.data:
                    if new_instruction.name == 'h':
                        new_qc.h(qargs[0])
                    elif new_instruction.name == 's':
                        new_qc.s(qargs[0])
                    elif new_instruction.name == 't':
                        new_qc.t(qargs[0])
            elif instruction.name == 'u3':
                num_rotations += 1
                para = instruction.params
                is0, snapped0 = snap_clifford_angle(para[0])
                is1, snapped1 = snap_clifford_angle(para[1])
                is2, snapped2 = snap_clifford_angle(para[2])
                if is0 and is1 and is2:
                    # region agent log
                    try:
                        import json as _agent_json, time as _agent_time
                        with open("/afs/cs.pitt.edu/usr0/yil392/Project/2025summer/.cursor/debug-477e0e.log", "a") as _agent_f:
                            _agent_f.write(_agent_json.dumps({
                                "sessionId": "477e0e",
                                "runId": "post-fix-v1",
                                "hypothesisId": "H1",
                                "location": "baseline.py:225",
                                "message": "Clifford-classified u3 gate with snapping before Clifford() call",
                                "data": {
                                    "original_params": [float(para[0]), float(para[1]), float(para[2])],
                                    "snapped_params": [float(snapped0), float(snapped1), float(snapped2)],
                                    "is_clifford_flags": [bool(is0), bool(is1), bool(is2)]
                                },
                                "timestamp": int(_agent_time.time() * 1000),
                            }) + "\n")
                    except Exception:
                        pass
                    # endregion agent log
                    temp_circ = QuantumCircuit(1)
                    temp_circ.u(snapped0, snapped1, snapped2, 0)
                    cl = Clifford(temp_circ)
                    temp_circ = cl.to_circuit()
                    temp_circ = transpile(temp_circ, basis_gates=['h', 's', 'sdg'], optimization_level=1)
                    for new_instruction, _, _ in temp_circ.data:
                        if new_instruction.name == 'h':
                            new_qc.h(qargs[0])
                        elif new_instruction.name == 's':
                            new_qc.s(qargs[0])
                        elif new_instruction.name == 'sdg':
                            new_qc.sdg(qargs[0])
                else:
                    seq, _, _ = trasyn.synthesize(para, t_budget, error_threshold=normal_error, gate_set='tsh', gpu=gpu)
                    for gate in seq:
                        if gate == 'h':
                            new_qc.h(qargs[0])
                        elif gate == 's':
                            new_qc.s(qargs[0])
                        elif gate == 't':
                            new_qc.t(qargs[0])
            elif instruction.name == 'cx':
                new_qc.cx(qargs[0], qargs[1])
        print("Number of rotations:", num_rotations)

    elif budget == 1 and rustiq == True and GRIDSYNTH == True:  #use rustiq
        hls_config = HLSConfig(PauliEvolution=[
            ("rustiq", {"optimize_count": True, "preserve_order": False})
        ])

        target_basis = ['cx', 'h', 's', 'rz', 'sdg']
        # target_basis = ['cx', 'h', 's', 'sdg', 'u3']
        qc = transpile(qc, hls_config=hls_config, basis_gates=target_basis, optimization_level=1)
        if not synthesize:
            for _ in range(Trotter_steps):
                Trotter_qc.compose(qc, inplace=True)
            return Trotter_qc, Trotter_qc
        # num_paulis = len(hamiltonian) - 1
        num_paulis = len(hamiltonian)
        # qc = rewrite_clifford_rz_u3_gates(qc)
        normal_error = error_threshold
        print("Normal error:", normal_error)
        num_rotations = 0
        new_qc = QuantumCircuit(num_qubits)
        for instruction, qargs, cargs in qc.data:
            if instruction.name == 'h':
                new_qc.h(qargs[0])
            elif instruction.name == 's':
                new_qc.s(qargs[0])
            elif instruction.name == 'sdg':
                new_qc.sdg(qargs[0])
            elif instruction.name == 'rz':
                num_rotations += 1
                para = instruction.params
                temp_circ = QuantumCircuit(1)
                temp_circ.rz(para[0], 0)
                decomposed = clifford_t_transpile(temp_circ, epsilon=error_threshold)
                for new_instruction, _, _ in decomposed.data:
                    if new_instruction.name == 'h':
                        new_qc.h(qargs[0])
                    elif new_instruction.name == 's':
                        new_qc.s(qargs[0])
                    elif new_instruction.name == 't':
                        new_qc.t(qargs[0])
                    elif new_instruction.name == 'x':
                        new_qc.x(qargs[0])
                    else:
                        print(new_instruction.name)
            elif instruction.name == 'cx':
                new_qc.cx(qargs[0], qargs[1])
        print("Number of rotations:", num_rotations)

    elif budget == 1 and rustiq == False and GRIDSYNTH == 1:  #use grid syn
        target_basis = ['cx', 'h', 's', 'rz', 'sdg']
        qc = transpile(qc, basis_gates=target_basis, optimization_level=1)
        if not synthesize:
            for _ in range(Trotter_steps):
                Trotter_qc.compose(qc, inplace=True)
            return Trotter_qc, Trotter_qc
        # qc = rewrite_clifford_rz_u3_gates(qc)
        # qc.draw(output="mpl")
        # plt.show()
        if benchmark is not None:
            with open("circuits/" + benchmark + "_" + "grid" + "_rz.qasm", "w") as f:
                f.write(dumps(qc))
        num_rotations = 0
        new_qc = QuantumCircuit(num_qubits)
        for instruction, qargs, cargs in qc.data:
            if instruction.name == 'h':
                new_qc.h(qargs[0])
            elif instruction.name == 's':
                new_qc.s(qargs[0])
            elif instruction.name == 'sdg':
                new_qc.sdg(qargs[0])
            elif instruction.name == 'rz':
                num_rotations += 1
                para = instruction.params
                temp_circ = QuantumCircuit(1)
                temp_circ.rz(para[0], 0)
                decomposed = clifford_t_transpile(temp_circ, epsilon=error_threshold)
                for new_instruction, _, _ in decomposed.data:
                    if new_instruction.name == 'h':
                        new_qc.h(qargs[0])
                    elif new_instruction.name == 's':
                        new_qc.s(qargs[0])
                    elif new_instruction.name == 't':
                        new_qc.t(qargs[0])
                    elif new_instruction.name == 'x':
                        new_qc.x(qargs[0])
                    else:
                        print(new_instruction.name)
            elif instruction.name == 'cx':
                new_qc.cx(qargs[0], qargs[1])
        print("Number of rotations:", num_rotations)

    elif budget == 1 and rustiq == False and GRIDSYNTH == 1 and use_trotter == True:  #use grid syn
        synth = LieTrotter(reps=Trotter_steps)
        evo_gate = PauliEvolutionGate(hamiltonian, time=evolution_time, synthesis=synth)
        qc = QuantumCircuit(hamiltonian.num_qubits)
        qc.append(evo_gate, range(hamiltonian.num_qubits))
        target_basis = ['cx', 'h', 's', 'rz', 'sdg']
        qc = transpile(qc, basis_gates=target_basis, optimization_level=1)
        if not synthesize:
            Trotter_qc.compose(qc, inplace=True)
            return Trotter_qc, Trotter_qc
        num_rotations = 0
        new_qc = QuantumCircuit(num_qubits)
        for instruction, qargs, cargs in qc.data:
            if instruction.name == 'h':
                new_qc.h(qargs[0])
            elif instruction.name == 's':
                new_qc.s(qargs[0])
            elif instruction.name == 'sdg':
                new_qc.sdg(qargs[0])
            elif instruction.name == 'rz':
                num_rotations += 1
                para = instruction.params
                temp_circ = QuantumCircuit(1)
                temp_circ.rz(para[0], 0)
                decomposed = clifford_t_transpile(temp_circ, epsilon=error_threshold)
                for new_instruction, _, _ in decomposed.data:
                    if new_instruction.name == 'h':
                        new_qc.h(qargs[0])
                    elif new_instruction.name == 's':
                        new_qc.s(qargs[0])
                    elif new_instruction.name == 't':
                        new_qc.t(qargs[0])
                    elif new_instruction.name == 'x':
                        new_qc.x(qargs[0])
                    else:
                        print(new_instruction.name)
            elif instruction.name == 'cx':
                new_qc.cx(qargs[0], qargs[1])
        print("Number of rotations:", num_rotations)
    
    if benchmark is not None:
        with open("circuits/" + benchmark + "_" + method + "_c+t.qasm", "w") as f:
            f.write(dumps(new_qc))
    t_count = sum(1 for instr, _, _ in new_qc.data if instr.name == 't')
    clifford_count = sum(1 for instr, _, _ in new_qc.data if instr.name != 't')
    print("T Count:", t_count)
    print("Clifford Count:", clifford_count)
    t_depth = new_qc.depth(lambda gate: gate[0].name == 't')
    print("T Depth:", t_depth)

    if use_trotter == False:
        for _ in range(Trotter_steps):
            Trotter_qc.compose(qc, inplace=True)
            synthesized_qc.compose(new_qc, inplace=True)
    else:
        Trotter_qc.compose(qc, inplace=True)
        synthesized_qc.compose(new_qc, inplace=True)
    
    # t_count = sum(1 for instr, _, _ in new_qc.data if instr.name == 't')
    # clifford_count = sum(1 for instr, _, _ in new_qc.data if instr.name != 't')
    # print("T Count:", t_count)
    # print("Clifford Count:", clifford_count)
    # t_depth = new_qc.depth(lambda gate: gate[0].name == 't')
    # print("T Depth:", t_depth)

    return Trotter_qc, synthesized_qc
    # print("Threshold:", thresholds)
    # new_qc.draw(output='mpl')
    # plt.show()

def phoenix_baseline_circuit(hamiltonian, pauli_strings, coeffs, error_threshold, gpu = 0, t_budget = 60, Trotter_steps = 1, evolution_time = 1, synthesize = True):
    dt = evolution_time / Trotter_steps
    num_paulis = len(hamiltonian)
    num_qubits = hamiltonian.num_qubits
    Trotter_qc = QuantumCircuit(num_qubits)
    synthesized_qc = QuantumCircuit(num_qubits)
    target_basis = ['cx', 'h', 's', 'rz', 'sdg', 'u3']
    hamiltonian = Hamiltonian(pauli_strings, coeffs)
    start_time = time.time()
    qc = compile_hamiltonian_simulation(hamiltonian, order_method='greedy', time=dt)
    end_time = time.time()
    print("Time taken to compile:", end_time - start_time)
    qc = transpile(qc, basis_gates=target_basis, optimization_level=1)
    if not synthesize:
        for _ in range(Trotter_steps):
            Trotter_qc.compose(qc, inplace=True)
        return Trotter_qc, Trotter_qc
    copy_qc = copy.deepcopy(qc)
    copy_qc = rewrite_clifford_rz_u3_gates(copy_qc)
    normal_error = normalized_error(copy_qc, error_threshold, num_paulis)
    print("Normal error:", normal_error)
    num_rotations = 0
    new_qc = QuantumCircuit(num_qubits)
    for instruction, qargs, cargs in qc.data:
        if instruction.name == 'h':
            new_qc.h(qargs[0])
        elif instruction.name == 's':
            new_qc.s(qargs[0])
        elif instruction.name == 'sdg':
            new_qc.sdg(qargs[0])
        elif instruction.name == 'rz':
            num_rotations += 1
            para = instruction.params
            temp_circ = QuantumCircuit(1)
            temp_circ.rz(para[0], 0)
            decomposed = clifford_t_transpile(temp_circ, epsilon=error_threshold)
            for new_instruction, _, _ in decomposed.data:
                if new_instruction.name == 'h':
                    new_qc.h(qargs[0])
                elif new_instruction.name == 's':
                    new_qc.s(qargs[0])
                elif new_instruction.name == 't':
                    new_qc.t(qargs[0])
        elif instruction.name == 'u3':
            num_rotations += 1
            para = instruction.params
            if (is_clifford_angle(para[0]) and
                is_clifford_angle(para[1]) and
                is_clifford_angle(para[2])):
                    temp_circ = QuantumCircuit(1)
                    temp_circ.u(para[0], para[1], para[2], 0)
                    cl = Clifford(temp_circ)
                    temp_circ = cl.to_circuit()
                    temp_circ = transpile(temp_circ, basis_gates=['h', 's', 'sdg'], optimization_level=1)
                    for new_instruction, _, _ in temp_circ.data:
                        if new_instruction.name == 'h':
                            new_qc.h(qargs[0])
                        elif new_instruction.name == 's':
                            new_qc.s(qargs[0])
                        elif new_instruction.name == 'sdg':
                            new_qc.sdg(qargs[0])
            else:
                seq, _, _ = trasyn.synthesize(para, t_budget, error_threshold=normal_error, gate_set='tsh', gpu=gpu)
                for gate in seq:
                    if gate == 'h':
                        new_qc.h(qargs[0])
                    elif gate == 's':
                        new_qc.s(qargs[0])
                    elif gate == 't':
                        new_qc.t(qargs[0])
        elif instruction.name == 'cx':
            new_qc.cx(qargs[0], qargs[1])
    print("Number of rotations:", num_rotations)
    t_count = sum(1 for instr, _, _ in new_qc.data if instr.name == 't')
    clifford_count = sum(1 for instr, _, _ in new_qc.data if instr.name != 't')
    print("T Count:", t_count)
    print("Clifford Count:", clifford_count)
    t_depth = new_qc.depth(lambda gate: gate[0].name == 't')
    print("T Depth:", t_depth)
    for _ in range(Trotter_steps):
        Trotter_qc.compose(qc, inplace=True)
        synthesized_qc.compose(new_qc, inplace=True)
    return Trotter_qc, synthesized_qc
