import networkx as nx


def pauli_to_symplectic(pstr):
    x, z = [], []
    for p in pstr:
        if p == 'I': x.append(0); z.append(0)
        elif p == 'X': x.append(1); z.append(0)
        elif p == 'Y': x.append(1); z.append(1)
        elif p == 'Z': x.append(0); z.append(1)
    return x + z

def commute_check(p1, p2):
    v1 = pauli_to_symplectic(p1)
    v2 = pauli_to_symplectic(p2)
    n = len(v1) // 2
    x1, z1 = v1[:n], v1[n:]
    x2, z2 = v2[:n], v2[n:]
    # Symplectic inner product
    dot = sum((x1[i] * z2[i] + x2[i] * z1[i]) % 2 for i in range(n))
    return dot % 2 == 0

def generate_commutation_graphs(pauli_strings):
    G_comm = nx.Graph()
    G_anticomm = nx.Graph()
    G_comm.add_nodes_from(pauli_strings)
    G_anticomm.add_nodes_from(pauli_strings)
    for i, left in enumerate(pauli_strings):
        for right in pauli_strings[i + 1:]:
            left_string = left[1] if isinstance(left, tuple) else left
            right_string = right[1] if isinstance(right, tuple) else right
            if commute_check(left_string, right_string):
                G_comm.add_edge(left, right)
            else:
                G_anticomm.add_edge(left, right)
    return G_comm, G_anticomm
