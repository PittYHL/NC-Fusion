import numpy as np
import copy
from itertools import combinations

def pauli_to_xz_matrix(pauli_strings):
    x_list, z_list = [], []
    for p in pauli_strings:
        x, z = [], []
        for ch in p:
            if ch == 'I':
                x.append(0)
                z.append(0)
            elif ch == 'X':
                x.append(1)
                z.append(0)
            elif ch == 'Y':
                x.append(1)
                z.append(1)
            elif ch == 'Z':
                x.append(0)
                z.append(1)
        x_list.append(x)
        z_list.append(z)
    return np.array(x_list), np.array(z_list)

def xz_to_pauli(x, z):
    paulis = []
    for xi, zi in zip(x, z):
        p = ''
        for a, b in zip(xi, zi):
            if a == 0 and b == 0: p += 'I'
            elif a == 1 and b == 0: p += 'X'
            elif a == 0 and b == 1: p += 'Z'
            elif a == 1 and b == 1: p += 'Y'
        paulis.append(p)
    return paulis

def H_gate(x, z, q):
    x_mat = x.copy()
    z_mat = z.copy()
    x_mat[:, q], z_mat[:, q] = z_mat[:, q], x_mat[:, q].copy()
    return x_mat, z_mat

def S_gate(x, z, q):
    x, z = x.copy(), z.copy()
    z[:, q] ^= x[:, q]
    return x, z

def CNOT_gate(x, z, control, target):
    x, z = x.copy(), z.copy()
    x[:, target] ^= x[:, control]
    z[:, control] ^= z[:, target]
    return x, z

def SWAP_gate(x, z, i, j):
    x, z = x.copy(), z.copy()
    x[:, i], x[:, j] = x[:, j].copy(), x[:, i].copy()
    z[:, i], z[:, j] = z[:, j].copy(), z[:, i].copy()
    return x, z

def eliminate_x_terms(x, z, row, column):
    # x_row = x[row].copy()
    cnot_ops = []
    active = [i for i in range(0, column) if x[row][i] == 1]
    pivot = np.where(x[row][column:] == 1)[0][0] + column
    for target in active:
        x, z = CNOT_gate(x, z, pivot, target)
        cnot_ops.append((pivot, target))
    return x, z, cnot_ops

def eliminate_x_terms_greedy(x, z, row, column, fixed):
    """
    General method to eliminate 1s in x_row using CNOTs.
    Returns a list of (control, target) CNOTs.
    """
    # x_row = x[row].copy()
    cnot_ops = []
    active = [i for i in range(column, len(x[row])) if x[row][i] == 1]

    while len(active) > 1:
        pair_scores = []
        combine = combinations(active, 2)
        for a, b in combine:
            for control, target in [(a, b), (b, a)]:
                x_temp = x.copy()
                z_temp = z.copy()
                x_temp, z_temp = CNOT_gate(x_temp, z_temp, control, target)
                score = np.sum(x_temp) + np.sum(z_temp)
                pair_scores.append((score, (control, target)))

        # Sort and greedily select disjoint pairs
        pair_scores.sort()
        used = set()
        selected_pairs = []

        if fixed != -1:
            for _, (control, target) in pair_scores:
                if control == fixed:
                    selected_pairs.append((control, target))
                    used.add(control)
                    used.add(target)
                    break

        for _, (control, target) in pair_scores:
            if control not in used and target not in used:
                selected_pairs.append((control, target))
                used.add(control)
                used.add(target)

        # Apply selected pairs
        for control, target in selected_pairs:
            x, z = CNOT_gate(x, z, control, target)
            cnot_ops.append((control, target))

        active = [i for i in range(column, len(x[row])) if x[row][i] == 1]

    return x, z, cnot_ops

def eliminate_z_terms_greedy(x, z, row, column):
    """
    Eliminates 1s in a Z row using CNOT(control, target), which modifies Z[control] ^= Z[target].
    The goal is to reduce the number of 1s to at most 1.
    """
    cnot_ops = []
    active = [i for i in range(column, len(z[row])) if z[row][i] == 1]

    while len(active) > 1:
        pair_scores = []
        combine = combinations(active, 2)
        for a, b in combine:
            for control, target in [(a, b), (b, a)]:
                x_temp = x.copy()
                z_temp = z.copy()
                x_temp, z_temp = CNOT_gate(x_temp, z_temp, control, target)
                score = np.sum(x_temp) + np.sum(z_temp)
                pair_scores.append((score, (control, target)))

        # Sort and greedily select disjoint pairs
        pair_scores.sort()
        used = set()
        selected_pairs = []

        for _, (control, target) in pair_scores:
            if control not in used and target not in used:
                selected_pairs.append((control, target))
                used.add(control)
                used.add(target)

        # Apply selected pairs
        for control, target in selected_pairs:
            x, z = CNOT_gate(x, z, control, target)
            cnot_ops.append((control, target))

        active = [i for i in range(column, len(z[row])) if z[row][i] == 1]

    return x, z, cnot_ops

def single_qubit_xz(x, z, row, column, circuit):
    for i in range(column, len(x[row])):
        if x[row][i] == z[row][i] and z[row][i] == 1:
            x, z = S_gate(x, z, i)
            circuit.append({'gate': 'S', 'qubits': [i]})
        elif x[row][i] == 0 and z[row][i] == 1:
            x, z = H_gate(x, z, i)
            circuit.append({'gate': 'H', 'qubits': [i]})
    return x, z, circuit

def dependent_pauli_circuit(paulis, circuit):
    x, z = pauli_to_xz_matrix(paulis)
    for gate in circuit:
        if gate['gate'] == 'H':
            x, z = H_gate(x,z,gate['qubits'][0])
        elif gate['gate'] == 'S':
            x, z = S_gate(x,z,gate['qubits'][0])
        elif gate['gate'] == 'CNOT':
            x, z = CNOT_gate(x,z,gate['qubits'][0],gate['qubits'][1])
        elif gate['gate'] == 'SWAP':
            x, z = SWAP_gate(x,z,gate['qubits'][0],gate['qubits'][1])
    return x, z

def remove_SWAP(x, z, circuit, tracker):
    column_tracker = list(range(len(x[0])))
    circuit_copy = copy.deepcopy(circuit)
    for gate in circuit_copy:
        if gate['gate'] == 'SWAP':
            # circuit_copy.remove(gate)
            column_tracker[gate['qubits'][0]], column_tracker[gate['qubits'][1]] = column_tracker[gate['qubits'][1]], column_tracker[gate['qubits'][0]]
        else:
            for i in range(len(gate['qubits'])):
                gate['qubits'][i] = column_tracker[gate['qubits'][i]]
    remove_gate = []
    for gate in circuit_copy:
        if gate['gate'] == 'SWAP':
            remove_gate.append(gate)
    for gate in remove_gate:
        circuit_copy.remove(gate)
    x_copy = copy.deepcopy(x)
    z_copy = copy.deepcopy(z)
    for i in range(len(tracker)):
        if tracker[i] != i:
            x_copy[:, tracker[i]] = x[:, i]
            z_copy[:, tracker[i]] = z[:, i]
    return x_copy, z_copy, circuit_copy


def _apply_single_qubit_clifford_with_sign(paulis, signs, gate, qubit):
    """Update Pauli labels and signs under single-qubit H or S on one qubit."""
    for idx, p in enumerate(paulis):
        ch = p[qubit]
        # Convert to list for in-place edit
        row = list(p)
        if gate == 'H':
            if ch == 'X':
                row[qubit] = 'Z'
            elif ch == 'Z':
                row[qubit] = 'X'
            elif ch == 'Y':
                # H Y H = -Y
                signs[idx] *= -1
        elif gate == 'S':
            if ch == 'X':
                # S X S† = Y
                row[qubit] = 'Y'
            elif ch == 'Y':
                # S Y S† = -X
                signs[idx] *= -1
                row[qubit] = 'X'
        paulis[idx] = ''.join(row)
    return paulis, signs

CNOT_CONJ_TABLE = {
    ('I','I'): ('I','I', +1),
    ('I','X'): ('I','X', +1),
    ('I','Y'): ('Z','Y', +1),
    ('I','Z'): ('Z','Z', +1),

    ('X','I'): ('X','X', +1),
    ('X','X'): ('X','I', +1),
    ('X','Y'): ('Y','Z', +1),
    ('X','Z'): ('Y','Y', -1),

    ('Y','I'): ('Y','X', +1),
    ('Y','X'): ('Y','I', +1),
    ('Y','Y'): ('X','Z', -1),
    ('Y','Z'): ('X','Y', +1),

    ('Z','I'): ('Z','I', +1),
    ('Z','X'): ('Z','X', +1),
    ('Z','Y'): ('I','Y', +1),
    ('Z','Z'): ('I','Z', +1),
}

def _apply_cnot_with_sign(paulis, signs, control, target):
    # x, z = pauli_to_xz_matrix(paulis)
    # # signs are ±1 only; track flips
    # for i in range(len(paulis)):
    #     xc, zc = x[i, control], z[i, control]
    #     xt, zt = x[i, target], z[i, target]

    #     # phase flip condition (uses pre-update bits)
    #     if xc and zt and (xt ^ zc ^ 1):
    #         signs[i] *= -1

    # # now update the x/z bits (same as your CNOT_gate)
    # x[:, target] ^= x[:, control]
    # z[:, control] ^= z[:, target]

    # paulis[:] = xz_to_pauli(x, z)
    # return paulis, signs
    # signs:  list[int], ±1 for each Pauli string
    out = []
    for i, p in enumerate(paulis):
        pc, pt = p[control], p[target]
        qc, qt, s = CNOT_CONJ_TABLE[(pc, pt)]
        signs[i] *= s

        # replace the two characters
        p_list = list(p)
        p_list[control] = qc
        p_list[target]  = qt
        out.append("".join(p_list))

    return out, signs

def _apply_swap_with_sign(paulis, signs, q0, q1):
    """Update Pauli labels under SWAP(q0, q1). No sign change."""
    for idx, p in enumerate(paulis):
        row = list(p)
        row[q0], row[q1] = row[q1], row[q0]
        paulis[idx] = ''.join(row)
    return paulis, signs


def _compute_signs_from_circuit(paulis, circuit):
    """Given original Pauli labels and a Clifford circuit, track sign changes.

    Returns a list of ±1 factors, in the same order as `paulis`, describing
    how each Pauli term's coefficient should be flipped by the circuit.
    """
    tracked_paulis = list(paulis)
    signs = [1] * len(tracked_paulis)
    for gate in circuit:
        name = gate['gate']
        qubits = gate['qubits']
        if name == 'H' or name == 'S':
            tracked_paulis, signs = _apply_single_qubit_clifford_with_sign(
                tracked_paulis, signs, name, qubits[0]
            )
        elif name == 'CNOT':
            tracked_paulis, signs = _apply_cnot_with_sign(
                tracked_paulis, signs, qubits[0], qubits[1]
            )
        elif name == 'SWAP':
            tracked_paulis, signs = _apply_swap_with_sign(
                tracked_paulis, signs, qubits[0], qubits[1]
            )
    return signs

def generate_row(x, z, row, column, column_tracker, circuit):
    if (np.all(x[row] == 0)):  # if all Z
        num_indicies = sum(z[row])
        if num_indicies != 1:
            x, z, cnot_ops = eliminate_z_terms_greedy(x, z, row, column)
            for cx in cnot_ops:
                # x, z = CNOT_gate(x, z, cx[0], cx[1])
                circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        sub_array = z[row][column:]
        index = np.where(sub_array == 1)[0][0]
        index = index + column
        if index != column:
            x, z = SWAP_gate(x, z, column, index)
            circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
            column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        row += 1
    else:  # has x
        x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
        x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, -1)
        for cx in cnot_ops:
            # x, z = CNOT_gate(x, z, cx[0], cx[1])
            circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        sub_array = x[row][column:]
        index = np.where(sub_array == 1)[0][0]
        index = index + column
        if index != column:
            x, z = SWAP_gate(x, z, column, index)
            circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
            column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        row += 1
    column += 1
    return x, z, row, column, column_tracker, circuit

# def generate_commute_circuit(paulis):
#     num_qubits = len(paulis[0])
#     row = 0  # track the current row
#     column = 0  # untouched before this
#     column_tracker = list(range(len(paulis[0])))
#     x, z = pauli_to_xz_matrix(paulis)
#     x = np.array(x)
#     z = np.array(z)
#     circuit = []
#     for i in range(len(paulis)):
#         x, z, row, column, column_tracker, circuit = generate_row(x, z, row, column, column_tracker, circuit)
#     x, z, new_circuit = remove_SWAP(x, z, circuit, column_tracker)
#     final_paulis = xz_to_pauli(x, z)
#     reversed_circuit = copy.deepcopy(new_circuit)
#     reversed_circuit.reverse()
#     original_x, original_z = dependent_pauli_circuit(final_paulis, reversed_circuit)
#     original_paulis = xz_to_pauli(original_x, original_z)
#     if original_paulis != paulis:
#         raise TypeError("the original and produced paulis not match!")
#     return final_paulis, new_circuit

def generate_commute_circuit(x, z, circuit, commute_row, column_tracker, column, num_commute):
    original_commute_row = copy.deepcopy(commute_row)
    old_x = copy.deepcopy(x)
    old_z = copy.deepcopy(z)
    for i in range(num_commute):
        # if (np.all(x[commute_row] == 0)):
        #     num_indicies = sum(z[commute_row])
        #     if num_indicies != 1:
        #         x, z, cnot_ops = eliminate_z_terms_greedy(x, z, commute_row, 0)
        #         for cx in cnot_ops:
        #             # x, z = CNOT_gate(x, z, cx[0], cx[1])
        #             circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        #     index = np.where(z[commute_row] == 1)[0][0]
        #     if index != column:
        #         x, z = SWAP_gate(x, z, column, index)
        #         circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
        #         column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        # else:  # has x
        x, z, circuit = single_qubit_xz(x, z, commute_row, column, circuit)
        x, z, cnot_ops = eliminate_x_terms_greedy(x, z, commute_row, column, -1)
        for cx in cnot_ops:
            # x, z = CNOT_gate(x, z, cx[0], cx[1])
            circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        if (np.sum(x[commute_row]) != 1):
            x, z, cnot_ops = eliminate_x_terms(x, z, commute_row, column)
            for cx in cnot_ops:
                # x, z = CNOT_gate(x, z, cx[0], cx[1])
                circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        index = np.where(x[commute_row] == 1)[0][0]
        if index != column:
            x, z = SWAP_gate(x, z, column, index)
            circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
            column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        column += 1
        commute_row += 1
    if np.sum(x[original_commute_row:]) + np.sum(z[original_commute_row:]) != num_commute:
        raise TypeError("the number of commute qubits not match!")
    return x, z, circuit, column_tracker

def greedy_circuit_generation(group):
    # if group['commute'] != []:
    #     final_paulis, new_circuit = generate_commute_circuit(group['commute'])
    #     return final_paulis, new_circuit
    num_commute = len(group['commute'])
    commute_row = len(group[1]) + len(group[2]) + len(group[3]) + len(group['dependent']) #track when the commute row starts
    paulis = group[1] + group[2] + group[3] + group['dependent'] + group['commute']
    # paulis = group
    row = 0 #track the current row
    column = 0 #untouched before this
    column_tracker = list(range(len(paulis[0])))
    x, z = pauli_to_xz_matrix(paulis)
    x = np.array(x)
    z = np.array(z)
    circuit = []
    #if there is only one term then no circuit
    if len(group[1]) == 1 and len(group[2]) == 0 and len(group['commute']) == 0:
        signs = [1] * len(paulis)
        return paulis, circuit, signs
    #do the first row
    if len(group[1]) != 0:
        if (np.all(x[row] == 0)): #if all Z
            num_indicies = sum(z[row])
            if num_indicies != 1:
                x, z, cnot_ops = eliminate_z_terms_greedy(x, z, row, column)
                for cx in cnot_ops:
                    # x, z = CNOT_gate(x, z, cx[0], cx[1])
                    circuit.append({'gate':'CNOT', 'qubits':[cx[0], cx[1]]})
            index = np.where(z[row] == 1)[0][0]
            if index != column:
                x, z = SWAP_gate(x, z, column, index)
                circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
                column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
            row += 1
        else: #has x
            x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
            x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, -1)
            for cx in cnot_ops:
                # x, z = CNOT_gate(x, z, cx[0], cx[1])
                circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
            index = np.where(x[row] == 1)[0][0]
            if index != column:
                x, z = SWAP_gate(x, z, column, index)
                circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
                column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
            if len(group[1]) == 2:
                x, z = H_gate(x, z, column)
                circuit.append({'gate': 'H', 'qubits': [column]})
            row += 1

    if len(group[1]) == 2:
        # do the second row
        x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
        x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, 0)
        for cx in cnot_ops:
            # x, z = CNOT_gate(x, z, cx[0], cx[1])
            circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        index = np.where(x[row] == 1)[0][0]
        if index != column:
            x, z = SWAP_gate(x, z, column, index)
            circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
            column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        row += 1
    if len(group[1]) != 0:
        column += 1

    if len(group[2]) == 0:
        if num_commute != 0:
            x, z, circuit, column_tracker = generate_commute_circuit(
                x, z, circuit, commute_row, column_tracker, column, num_commute
            )
        x, z, new_circuit = remove_SWAP(x, z, circuit, column_tracker)
        final_paulis = xz_to_pauli(x, z)
        reversed_circuit = copy.deepcopy(new_circuit)
        reversed_circuit.reverse()
        original_x, original_z = dependent_pauli_circuit(final_paulis, reversed_circuit)
        original_paulis = xz_to_pauli(original_x, original_z)
        if original_paulis != paulis:
            raise TypeError("the original and produced paulis not match!")
        signs = _compute_signs_from_circuit(paulis, new_circuit)
        return final_paulis, new_circuit, signs


    if len(group[2]) != 0:
        if (np.all(x[row] == 0)):  # if all Z
            num_indicies = sum(z[row])
            if num_indicies != 1:
                x, z, cnot_ops = eliminate_z_terms_greedy(x, z, row, column)
                for cx in cnot_ops:
                    # x, z = CNOT_gate(x, z, cx[0], cx[1])
                    circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
            sub_array = z[row][column:]
            index = np.where(sub_array == 1)[0][0]
            index = index + column
            if index != column:
                x, z = SWAP_gate(x, z, column, index)
                circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
                column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
            row += 1
        else:  # has x
            x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
            x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, -1)
            for cx in cnot_ops:
                # x, z = CNOT_gate(x, z, cx[0], cx[1])
                circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
            sub_array = x[row][column:]
            index = np.where(sub_array == 1)[0][0]
            index = index + column
            if index != column:
                x, z = SWAP_gate(x, z, column, index)
                circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
                column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
            if len(group[2]) == 2:
                x, z = H_gate(x, z, column)
                circuit.append({'gate': 'H', 'qubits': [column]})
            row += 1

    if len(group[2]) == 2:
        x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
        x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, 1)
        for cx in cnot_ops:
            # x, z = CNOT_gate(x, z, cx[0], cx[1])
            circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        sub_array = x[row][column:]
        index = np.where(sub_array == 1)[0][0]
        index = index + column
        if index != column:
            x, z = SWAP_gate(x, z, column, index)
            circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
            column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        row += 1
    if len(group[2]) != 0:
        column += 1

    if len(group[3]) == 0:
        if num_commute != 0:
            x, z, circuit, column_tracker = generate_commute_circuit(
                x, z, circuit, commute_row, column_tracker, column, num_commute
            )
        x, z, new_circuit = remove_SWAP(x, z, circuit, column_tracker)
        final_paulis = xz_to_pauli(x, z)
        reversed_circuit = copy.deepcopy(new_circuit)
        reversed_circuit.reverse()
        original_x, original_z = dependent_pauli_circuit(final_paulis, reversed_circuit)
        original_paulis = xz_to_pauli(original_x, original_z)
        if original_paulis != paulis:
            raise TypeError("the original and produced paulis not match!")
        signs = _compute_signs_from_circuit(paulis, new_circuit)
        return final_paulis, new_circuit, signs

    if len(group[3]) != 0:
        if (np.all(x[row] == 0)):  # if all Z
            num_indicies = sum(z[row])
            if num_indicies != 1:
                x, z, cnot_ops = eliminate_z_terms_greedy(x, z, row, column)
                for cx in cnot_ops:
                    # x, z = CNOT_gate(x, z, cx[0], cx[1])
                    circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
            sub_array = z[row][column:]
            index = np.where(sub_array == 1)[0][0]
            index = index + column
            if index != column:
                x, z = SWAP_gate(x, z, column, index)
                circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
                column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
            row += 1
        else:  # has x
            x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
            x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, -1)
            for cx in cnot_ops:
                # x, z = CNOT_gate(x, z, cx[0], cx[1])
                circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
            sub_array = x[row][column:]
            index = np.where(sub_array == 1)[0][0]
            index = index + column
            if index != column:
                x, z = SWAP_gate(x, z, column, index)
                circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
                column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
            if len(group[3]) == 2:
                x, z = H_gate(x, z, column)
                circuit.append({'gate': 'H', 'qubits': [column]})
            row += 1

    if len(group[3]) == 2:
        x, z, circuit = single_qubit_xz(x, z, row, column, circuit)
        x, z, cnot_ops = eliminate_x_terms_greedy(x, z, row, column, 2)
        for cx in cnot_ops:
            # x, z = CNOT_gate(x, z, cx[0], cx[1])
            circuit.append({'gate': 'CNOT', 'qubits': [cx[0], cx[1]]})
        sub_array = x[row][column:]
        index = np.where(sub_array == 1)[0][0]
        index = index + column
        if index != column:
            x, z = SWAP_gate(x, z, column, index)
            circuit.append({'gate': 'SWAP', 'qubits': [column, index]})
            column_tracker[column], column_tracker[index] = column_tracker[index], column_tracker[column]
        row += 1
    if len(group[3]) != 0:
        column += 1
    if num_commute != 0:
        x, z, circuit, column_tracker = generate_commute_circuit(
            x, z, circuit, commute_row, column_tracker, column, num_commute
        )
    # x = np.concatenate((x, dependent_x), axis=0)
    # z = np.concatenate((z, dependent_z), axis=0)
    x, z, new_circuit = remove_SWAP(x, z, circuit, column_tracker)
    # dependent_x, dependent_z = dependent_pauli_circuit(paulis, new_circuit)
    final_paulis = xz_to_pauli(x, z)
    reversed_circuit = copy.deepcopy(new_circuit)
    reversed_circuit.reverse()
    original_x, original_z = dependent_pauli_circuit(final_paulis, reversed_circuit)
    original_paulis = xz_to_pauli(original_x, original_z)
    if original_paulis != paulis:
        raise TypeError("the original and produced paulis not match!")
    signs = _compute_signs_from_circuit(paulis, new_circuit)
    return final_paulis, new_circuit, signs

def new_paulis_transform(pauli_list, new_pauli, group, signs=None):
    old_paulis = group[1] + group[2] + group[3] + group['dependent']
    new_pauli_list = {}
    index = 0
    for i in range(len(old_paulis)):
        pauli_reversed = new_pauli[i]
        pauli_reversed = pauli_reversed[::-1]
        coeff = pauli_list[old_paulis[i]]
        if signs is not None:
            coeff = signs[i] * coeff
        new_pauli_list[pauli_reversed] = coeff
        index += 1
    commute_pauli_list = {}
    old_paulis = group['commute']
    for i in range(len(old_paulis)):
        pauli_reversed = new_pauli[i + index]
        pauli_reversed = pauli_reversed[::-1]
        coeff = pauli_list[old_paulis[i]]
        if signs is not None:
            coeff = signs[i + index] * coeff
        commute_pauli_list[pauli_reversed] = coeff
    return new_pauli_list, commute_pauli_list

def transfprmed_sub(sub):
    new_sub = {1: [], 2: [], 3: [], 'dependent': [], 'commute': []}
    g1 = sub[1]
    g2 = sub[2]
    g3 = sub[3]
    gd = sub['dependent']
    gm = sub['commute']
    for pauli in g1:
        new_sub[1].append(pauli[1])
    for pauli in g2:
        new_sub[2].append(pauli[1])
    for pauli in g3:
        new_sub[3].append(pauli[1])
    for pauli in gd:
        new_sub['dependent'].append(pauli[1])
    for pauli in gm:
        new_sub['commute'].append(pauli[1])
    return new_sub