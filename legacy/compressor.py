from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
import copy
from docker import (
    clear_unitary_output,
    execute_main_in_docker,
    load_all_qasm_circuits,
    run_docker_with_volumes,
    select_least_T_circuit,
    write_unitary_to_file,
)
from qiskit.qasm2 import dumps

import numpy as np
import math
import time

def append_gate(qc, new_qc):
    for instruction, qargs, cargs in new_qc.data:
        qc.append(instruction, qargs, cargs)
    return qc


def insert_gate(qc, circuit, evolution_gate, commute_gate = None, target_basis = None):
    num_qubits = qc.num_qubits - 1
    #insert the gate before the new pauli
    new_qc = QuantumCircuit(qc.num_qubits)
    for gate in circuit:
        if gate['gate'] == 'H':
            new_qc.h(num_qubits - gate['qubits'][0])
        elif gate['gate'] == 'S':
            new_qc.s(num_qubits - gate['qubits'][0])
        elif gate['gate'] == 'CNOT':
            new_qc.cx(num_qubits - gate['qubits'][0], num_qubits - gate['qubits'][1])
        elif gate['gate'] == 'SWAP':
            new_qc.swap(num_qubits - gate['qubits'][0], num_qubits - gate['qubits'][1])
    # insert new pauli
    # qc.barrier()
    new_qc.append(evolution_gate, qc.qubits)
    if commute_gate != None:
        new_qc.append(commute_gate, qc.qubits)
    if target_basis != None:
        new_qc = transpile(new_qc, basis_gates=target_basis, optimization_level=1)
        # qc = append_gate(qc, new_qc)
        # qc.compose(new_qc, qubits=qc.qubits)
    else:
        qc.compose(new_qc, qubits=qc.qubits)
    # qc.barrier()
    # insert the gate after the new pauli
    reversed_circuit = copy.deepcopy(circuit)
    reversed_circuit.reverse()
    for gate in reversed_circuit:
        if gate['gate'] == 'H':
            new_qc.h(num_qubits - gate['qubits'][0])
        elif gate['gate'] == 'S':
            new_qc.sdg(num_qubits - gate['qubits'][0])
        elif gate['gate'] == 'CNOT':
            new_qc.cx(num_qubits - gate['qubits'][0], num_qubits - gate['qubits'][1])
        elif gate['gate'] == 'SWAP':
            new_qc.swap(num_qubits - gate['qubits'][0], num_qubits - gate['qubits'][1])
    # qc.barrier()
    qc = qc.compose(new_qc, qubits=qc.qubits)
    return qc

def insert_gate_synthetiq(qc, circuit, evolution_gate, commute_gate = None):
    #insert the gate before the new pauli
    target_basis = ['cx', 'h', 's', 'rz', 'sdg']
    for gate in circuit:
        if gate['gate'] == 'H':
            qc.h(gate['qubits'][0])
        elif gate['gate'] == 'S':
            qc.s(gate['qubits'][0])
        elif gate['gate'] == 'CNOT':
            qc.cx(gate['qubits'][0], gate['qubits'][1])
        elif gate['gate'] == 'SWAP':
            qc.swap(gate['qubits'][0], gate['qubits'][1])
    # insert new pauli
    # qc.barrier()
    qc = qc.compose(evolution_gate, qubits=qc.qubits)
    if commute_gate != None:
        new_qc = QuantumCircuit(qc.num_qubits)
        new_qc.append(commute_gate, qc.qubits)
        new_qc = transpile(new_qc, basis_gates=target_basis, optimization_level=1)
        qc = append_gate(qc, new_qc)
    # qc.barrier()
    # insert the gate after the new pauli
    reversed_circuit = copy.deepcopy(circuit)
    reversed_circuit.reverse()
    for gate in reversed_circuit:
        if gate['gate'] == 'H':
            qc.h(gate['qubits'][0])
        elif gate['gate'] == 'S':
            qc.sdg(gate['qubits'][0])
        elif gate['gate'] == 'CNOT':
            qc.cx(gate['qubits'][0], gate['qubits'][1])
        elif gate['gate'] == 'SWAP':
            qc.swap(gate['qubits'][0], gate['qubits'][1])
    # qc.barrier()
    return qc

def is_non_clifford_angle(x, tol=1e-8):
    """Return whether ``x`` is not an integer multiple of π/2."""
    ratio = x / (math.pi / 2)
    return abs(ratio - round(ratio)) > tol

def get_num_u(qc):
    num_u = 0
    for instruction, qargs, cargs in qc.data:
        if instruction.name == 'u3':
            num_u += 1
        elif instruction.name == 'rz' and is_non_clifford_angle(instruction.params[0]):
            num_u += 1
    return num_u

def normalized_error(qc, error_threshold, num_paulis):
    num_u = get_num_u(qc)
    print("num_u:", num_u)
    print("num_paulis:", num_paulis)
    return error_threshold * num_paulis / num_u


def is_clifford_angle(theta):
    """Returns True if angle is a multiple of pi/2."""
    theta = float(theta) % (2 * np.pi)
    multiples = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    return any(np.isclose(theta, m, atol=1e-8) for m in multiples)


def replace_rz_with_clifford(circuit, qubit, theta):
    """Replace RZ with equivalent Clifford if possible."""
    theta = float(theta) % (2 * np.pi)
    if np.isclose(theta, np.pi / 2, atol=1e-8):
        circuit.s(qubit)
    elif np.isclose(theta, np.pi, atol=1e-8):
        circuit.s(qubit)
        circuit.s(qubit)
    elif np.isclose(theta, 3 * np.pi / 2, atol=1e-8):
        circuit.sdg(qubit)


def replace_u3_with_clifford(circuit, qubit, theta, phi, lam):
    """Try replacing U3 with Clifford. Covers special cases."""
    theta = float(theta) % (2 * np.pi)
    phi = float(phi) % (2 * np.pi)
    lam = float(lam) % (2 * np.pi)

    if np.isclose(theta, 0, atol=1e-8):
        # u3(0, phi, lam) ≈ rz(phi + lam)
        total = (phi + lam) % (2 * np.pi)
        replace_rz_with_clifford(circuit, qubit, total)
    elif np.isclose(theta, np.pi, atol=1e-8) and np.isclose(phi, 0, atol=1e-8) and np.isclose(lam, 0, atol=1e-8):
        circuit.x(qubit)
    elif (np.isclose(theta, np.pi / 2, atol=1e-8) and
          np.isclose(phi, 0, atol=1e-8) and
          np.isclose(lam, np.pi, atol=1e-8)):
        # Equivalent to H
        circuit.h(qubit)
    else:
        import trasyn

        seq, _, _ = trasyn.synthesize([theta, phi, lam], 10, error_threshold=0.01, gate_set='tsh', gpu=0)
        qc = QuantumCircuit(1)
        for gate in seq:
            if gate == 'h':
                qc.h(0)
                circuit.h(qubit)
            elif gate == 's':
                qc.s(0)
                circuit.s(qubit)
            elif gate == 't':
                qc.t(0)
                circuit.t(qubit)
        qc = transpile(qc, basis_gates=['u3'], optimization_level=1)
        # print(qc)
        
        # qc = transpile(qc, basis_gates=['h', 's'], optimization_level=1)
        # circuit = circuit.compose(qc, qubits=[qubit])




def rewrite_clifford_rz_u3_gates(circ):
    new_circ = QuantumCircuit(*circ.qregs)

    for inst, qargs, cargs in circ.data:
        if inst.name == 'measure':
            continue
        if inst.name == 'rz':
            theta = inst.params[0]
            if is_clifford_angle(theta):
                replace_rz_with_clifford(new_circ, qargs[0], theta)
            else:
                new_circ.append(inst, qargs, cargs)
        elif inst.name == 'u3':
            theta, phi, lam = inst.params
            if (is_clifford_angle(theta) and
                    is_clifford_angle(phi) and
                    is_clifford_angle(lam)):
                # print(theta, phi, lam)
                replace_u3_with_clifford(new_circ, qargs[0], theta, phi, lam)
                # temp_circ = QuantumCircuit(1)
                # temp_circ.u(theta, phi, lam, 0)
                # cl = Clifford(temp_circ)
                # temp_circ = cl.to_circuit()
                # temp_circ = transpile(temp_circ, basis_gates=['h', 's', 'sdg'], optimization_level=1)
                # new_circ = new_circ.compose(temp_circ, qubits=[qargs[0]])
            else:
                new_circ.append(inst, qargs, cargs)
        else:
            new_circ.append(inst, qargs, cargs)
    return new_circ

def compressor_circuit(new_paulis, commute_paulis, circuits, error_threshold, budget, num_qubits, gpu = 0, num_paulis = 0, fix_error_threshold = 0, rz = 0, gridsyn = False, trotter_steps = 1, evolution_time = 1, synthesize = True, benchmark = None, t_budget = 60):
    if gridsyn:
        rz = 1
    dt = evolution_time / trotter_steps
    if rz:
        target_basis = ['cx', 'h', 's', 'rz', 'sdg']
    else:
        target_basis = ['cx', 'h', 's', 'u3', 'rz', 'sdg']
    baqit_basis = ['cx', 'h', 's', 'rz', 'sdg']
    Trotter_qc = QuantumCircuit(num_qubits)
    synthesized_qc = QuantumCircuit(num_qubits)
    qc = QuantumCircuit(num_qubits)
    if budget == 1:
        for i in range(len(new_paulis)):
            evolution_gate = QuantumCircuit(num_qubits)
            circuit = circuits[i]
            # if new_paulis[i] != {}:
            pauli_string = new_paulis[i] | commute_paulis[i]
            dict_keys_object = pauli_string.keys()
            # pauli_strs = list(dict_keys_object)
            pauli_strs = [pauli[::-1] for pauli in dict_keys_object]
            coeffs = list(pauli_string.values())
            hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strs, coeffs)))
            evolution_gate = PauliEvolutionGate(hamiltonian, time=dt)
            #commute
            # if commute_paulis[i] != {}:
            #     commute_string = commute_paulis[i]
            #     dict_keys_object = commute_string.keys()
            #     pauli_strs = list(dict_keys_object)
            #     coeffs = list(commute_string.values())
            #     hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strs, coeffs)))
            #     commute_gate = PauliEvolutionGate(hamiltonian, time=evolution_time)
            #     qc = insert_gate(qc, circuit, evolution_gate, target_basis, commute_gate)
            # else:
            qc = insert_gate(qc, circuit, evolution_gate, commute_gate = None, target_basis = target_basis)
            # print(qc)
            # qc.barrier()
        # qc = transpile(qc, basis_gates=target_basis, optimization_level=1)
        # density_matrix_error(original_qc, qc)

    elif budget == 2 or budget == 3:
        for i in range(len(new_paulis)):
            temp_qc = QuantumCircuit(num_qubits)
            pauli_string = new_paulis[i]
            circuit = circuits[i]
            dict_keys_object = pauli_string.keys()
            pauli_strs = list(dict_keys_object)
            coeffs = list(pauli_string.values())
            hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strs, coeffs)))
            evolution_gate = PauliEvolutionGate(hamiltonian, time=evolution_time)
            temp_qc.append(evolution_gate, temp_qc.qubits)
            temp_qc = transpile(temp_qc, basis_gates=baqit_basis, optimization_level=1)
            if commute_paulis[i] != {}:
                commute_string = commute_paulis[i]
                dict_keys_object = commute_string.keys()
                pauli_strs = list(dict_keys_object)
                coeffs = list(commute_string.values())
                hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strs, coeffs)))
                commute_gate = PauliEvolutionGate(hamiltonian, time=evolution_time)
                qc = insert_gate(qc, circuit, temp_qc, commute_gate, target_basis = baqit_basis)
            else:
                qc = insert_gate(qc, circuit, temp_qc)
        qc = transpile(qc, basis_gates=baqit_basis, optimization_level=1)
    # density_matrix_error(original_qc, qc)
    # qc = rewrite_clifford_rz_u3_gates(qc)
    # density_matrix_error(original_qc, qc)
    if benchmark is not None:
        with open("circuits/" + benchmark + "_" + "ncf" + "_rz.qasm", "w") as f:
            f.write(dumps(qc))
    if not synthesize:
        for _ in range(trotter_steps):
            Trotter_qc.compose(qc, inplace=True)
        return Trotter_qc, None
    import trasyn

    if gridsyn:
        from qiskit_gridsynth_plugin.decompose import clifford_t_transpile

    #normalize the error
    if num_paulis == 0:
        for pauli_string in new_paulis:
            num_paulis += len(pauli_string)
        for pauli_string in commute_paulis:
            num_paulis += len(pauli_string)
    num_rotations = 0
    for pauli_string in new_paulis:
        if pauli_string != {}:
            num_rotations = num_rotations + 1
    for pauli_string in commute_paulis:
        num_rotations += len(pauli_string)
    if fix_error_threshold == 0:
        normal_error = error_threshold * num_paulis / num_rotations
    else:
        normal_error = error_threshold
    print("Num rotation:", num_rotations)
    print("Normal Error:", normal_error)

    new_qc = QuantumCircuit(num_qubits)
    for instruction, qargs, cargs in qc.data:
        if instruction.name == 'h':
            new_qc.h(qargs[0])
        elif instruction.name == 's':
            new_qc.s(qargs[0])
        elif instruction.name == 'sdg':
            new_qc.sdg(qargs[0])
        elif instruction.name == 'tdg':
            new_qc.tdg(qargs[0])
        elif instruction.name == 'rz' and gridsyn == False:
            para = instruction.params
            seq, _, _ = trasyn.synthesize(para[0], t_budget, error_threshold=normal_error, gate_set='tsh', gpu=gpu)
            for gate in seq:
                if gate == 'h':
                    new_qc.h(qargs[0])
                elif gate == 's':
                    new_qc.s(qargs[0])
                elif gate == 't':
                    new_qc.t(qargs[0])
                elif gate == 'x':
                    new_qc.x(qargs[0])
        elif instruction.name == 'rz' and gridsyn == True:
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
        elif instruction.name == 'u3':
            para = instruction.params
            theta, phi, lam = instruction.params
            if (is_clifford_angle(theta) and
                    is_clifford_angle(phi) and
                    is_clifford_angle(lam)):
                new_qc.h(qargs[0])
                new_qc.s(qargs[0])
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
        else:
            print(instruction.name, qargs)

    # density_matrix_error(qc, new_qc)
    if benchmark is not None:
        with open("circuits/" + benchmark + "_" + "ncf" + "_c+t.qasm", "w") as f:
            f.write(dumps(new_qc))
    t_count = sum(1 for instr, _, _ in new_qc.data if instr.name == 't')
    clifford_count = sum(1 for instr, _, _ in new_qc.data if instr.name != 't')
    print("T Count:", t_count)
    print("Clifford Count:", clifford_count)
    t_depth = new_qc.depth(lambda gate: gate[0].name == 't')
    print("T Depth:", t_depth)
    # print(original_qc.global_phase)
    # print(qc.global_phase)
    # print(new_qc.global_phase)
    for i in range(trotter_steps):
        Trotter_qc.compose(qc, inplace=True)
        synthesized_qc.compose(new_qc, inplace=True)
    return Trotter_qc, synthesized_qc
    # print("Threshold:", thresholds)
    # new_qc.draw(output='mpl')
    # plt.show()


def compress_circuit(circuit: QuantumCircuit):
    # Step 1: Find all used qubit indices
    used_qubit_indices = sorted({
        circuit.qubits.index(q) for instr, qargs, _ in circuit.data for q in qargs
    })

    # Step 2: Build forward and reverse maps
    index_map = {old: new for new, old in enumerate(used_qubit_indices)}
    reverse_map = {new: old for old, new in index_map.items()}

    # Step 3: Build compressed circuit
    new_circuit = QuantumCircuit(len(used_qubit_indices))
    for instr, qargs, cargs in circuit.data:
        new_qargs = [new_circuit.qubits[index_map[circuit.qubits.index(q)]] for q in qargs]
        new_circuit.append(instr, new_qargs, cargs)

    return new_circuit, used_qubit_indices, reverse_map

def decompress_circuit(compressed, reverse_map, num_qubits):
    original_circuit = QuantumCircuit(num_qubits)

    for instr, qargs, cargs in compressed.data:
        mapped_qargs = [original_circuit.qubits[reverse_map[compressed.qubits.index(q)]] for q in qargs]
        original_circuit.append(instr, mapped_qargs, cargs)

    return original_circuit


def synthetiq_compressor(new_paulis, commute_paulis, circuits, error_threshold, budget, num_qubits, num_paulis = 0, normalized = 1):
    from qiskit_gridsynth_plugin.decompose import clifford_t_transpile

    clear_unitary_output()
    run_docker_with_volumes()
    evolution_time = 1
    # target_basis = ['cx', 'h', 's', 'u3', 'rz', 'sdg']
    target_basis = ['cx', 'h', 's', 'rz', 'sdg']
    qc = QuantumCircuit(num_qubits)
    num_rotations = 0
    if num_paulis == 0:
        for pauli_string in new_paulis:
            if pauli_string != {}:
                num_paulis += len(pauli_string)
        for pauli_string in commute_paulis:
            num_paulis += len(pauli_string)
    for pauli_string in new_paulis:
        num_rotations = num_rotations + 1
    for pauli_string in commute_paulis:
        num_rotations += len(pauli_string)
    if normalized:
        print('total number of rotations: ', num_rotations)
        error_threshold = error_threshold * num_paulis / num_rotations
        print('error threshold: ', error_threshold)
    if budget == 2 or budget == 3:
        for i in range(len(new_paulis)):
            temp_qc = QuantumCircuit(num_qubits)
            circuit = circuits[i]
            if new_paulis[i] != {}:
                pauli_string = new_paulis[i]
                dict_keys_object = pauli_string.keys()
                pauli_strs = list(dict_keys_object)
                coeffs = list(pauli_string.values())
                # print('====================================================')
                # print('coeffs: ', coeffs)
                hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strs, coeffs)))
                evolution_gate = PauliEvolutionGate(hamiltonian, time=evolution_time)
                temp_qc.append(evolution_gate, temp_qc.qubits)
                temp_qc = transpile(temp_qc, basis_gates=target_basis, optimization_level=1)
                if len(new_paulis[i]) > 1:
                    temp_qc, used_qubits, reverse_map = compress_circuit(temp_qc)

                    write_unitary_to_file(temp_qc, "unitary.txt")
                    time_start = time.time()
                    execute_main_in_docker(error_threshold)
                    time_end = time.time()
                    print(f"Time taken for execute_main_in_docker: {time_end - time_start} seconds")
                    temp_circuits = load_all_qasm_circuits()
                    clear_unitary_output()
                    temp_qc = select_least_T_circuit(temp_circuits)
                    temp_qc = decompress_circuit(temp_qc, reverse_map, num_qubits)

            # qc = qc.compose(temp_qc, qubits=used_qubits)
            if commute_paulis[i] != {}:
                commute_string = commute_paulis[i]
                dict_keys_object = commute_string.keys()
                pauli_strs = list(dict_keys_object)
                coeffs = list(commute_string.values())
                hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strs, coeffs)))
                commute_gate = PauliEvolutionGate(hamiltonian, time=evolution_time)
                qc = insert_gate_synthetiq(qc, circuit, temp_qc, commute_gate)
            else:
                qc = insert_gate_synthetiq(qc, circuit, temp_qc)
        # qc = transpile(qc, basis_gates=target_basis)
    # qc = transpile(qc, basis_gates=target_basis)
    # qc.draw(output='mpl')
    # plt.show()
    # qc = rewrite_clifford_rz_u3_gates(qc)


    new_qc = QuantumCircuit(num_qubits)
    for instruction, qargs, cargs in qc.data:
        if instruction.name == 'h':
            new_qc.h(qargs[0])
        elif instruction.name == 's':
            new_qc.s(qargs[0])
        elif instruction.name == 'sdg':
            new_qc.sdg(qargs[0])
        elif instruction.name == 't':
            new_qc.t(qargs[0])
        elif instruction.name == 'tdg':
            new_qc.tdg(qargs[0])
        elif instruction.name == 'rz':
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
            # print('====================================================')
#use synthetiq
            # temp_circ.rz(para[0], 0)
            # write_unitary_to_file(temp_circ, "unitary.txt")
            # execute_main_in_docker(error_threshold)
            # circuits = load_all_qasm_circuits()
            # clear_unitary_output()
            # circuit = select_least_T_circuit(circuits)
            # new_qc = new_qc.compose(circuit, qubits=[qargs[0]])
        elif instruction.name == 'cx':
            new_qc.cx(qargs[0], qargs[1])

    t_count = sum(1 for instr, _, _ in new_qc.data if instr.name in {'t', 'tdg'})
    clifford_count = sum(1 for instr, _, _ in new_qc.data if instr.name not in {'t', 'tdg'})
    t_depth = new_qc.depth(lambda gate: gate[0].name in {'t', 'tdg'})
    print("T Count:", t_count)
    print("Clifford Count:", clifford_count)
    print("T Depth:", t_depth)
    return new_qc
