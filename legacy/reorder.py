"""Reorder fused groups so independent groups can be scheduled together."""

def _active_qubits(pauli_strings):
    return {
        qubit
        for pauli in pauli_strings
        for qubit, symbol in enumerate(pauli)
        if symbol != "I"
    }


def reorder_pauli_groups(new_paulis, commute_paulis, circuits):
    """Group independent fused circuits while preserving each group's data."""
    if not (len(new_paulis) == len(commute_paulis) == len(circuits)):
        raise ValueError("new_paulis, commute_paulis, and circuits must have equal lengths")

    active_qubits = []
    for paulis, commuting, circuit in zip(new_paulis, commute_paulis, circuits):
        group_paulis = [pauli[::-1] for pauli in paulis]
        group_paulis.extend(pauli[::-1] for pauli in commuting)
        active = _active_qubits(group_paulis)
        active.update(qubit for gate in circuit for qubit in gate["qubits"])
        active_qubits.append(active)

    remaining = sorted(range(len(new_paulis)), key=lambda i: len(active_qubits[i]), reverse=True)
    reordered = []
    while remaining:
        batch = [remaining.pop(0)]
        used_qubits = set(active_qubits[batch[0]])
        for index in remaining[:]:
            if used_qubits.isdisjoint(active_qubits[index]):
                batch.append(index)
                used_qubits.update(active_qubits[index])

        remaining = [index for index in remaining if index not in batch]
        reordered.extend(batch)

    return (
        [new_paulis[i] for i in reordered],
        [commute_paulis[i] for i in reordered],
        [circuits[i] for i in reordered],
    )
