import numpy as np

def solve_linear_gf2(G, target):
    """
    Solve G.T @ x = target over GF(2).
    G: shape (r, 2n)
    target: shape (2n,)
    Returns binary vector x (shape r,) or None if no solution.
    """
    r, n = G.shape[0], G.shape[1]
    A = G.T.copy()  # shape (2n, r)
    b = target.copy()  # shape (2n,)

    # Augmented matrix [A | b]
    Ab = np.concatenate([A, b[:, np.newaxis]], axis=1).astype(np.uint8)

    num_rows, num_cols = Ab.shape
    rank = 0
    pivot_cols = []

    for col in range(r):
        pivot = np.argmax(Ab[rank:, col]) + rank
        if Ab[pivot, col] == 0:
            continue

        # Swap rows
        if pivot != rank:
            Ab[[rank, pivot]] = Ab[[pivot, rank]]

        pivot_cols.append(col)

        # Eliminate below and above
        for row in range(num_rows):
            if row != rank and Ab[row, col] == 1:
                Ab[row] ^= Ab[rank]

        rank += 1
        if rank == num_rows:
            break

    # Check for inconsistency
    for row in range(num_rows):
        if np.all(Ab[row, :-1] == 0) and Ab[row, -1] == 1:
            return None  # No solution

    # Back-substitution to extract one solution
    x = np.zeros(r, dtype=int)
    for i in range(rank - 1, -1, -1):
        pivot_col = pivot_cols[i]
        x[pivot_col] = Ab[i, -1]
        for j in range(pivot_col + 1, r):
            x[pivot_col] ^= (Ab[i, j] & x[j])

    return x


def pauli_to_binary(pauli):
    mapping = {'I': (0,0), 'X': (0,1), 'Y': (1,1), 'Z': (1,0)}
    n = len(pauli)
    z = np.zeros(n, dtype=int)
    x = np.zeros(n, dtype=int)
    for i, p in enumerate(pauli):
        z[i], x[i] = mapping[p]
    return np.concatenate([z, x])

def gaussian_elimination(matrix, row_order):
    m, n = matrix.shape
    rank = 0
    pivot_rows = []

    for col in range(n):
        candidates = np.where(matrix[rank:, col] == 1)[0]
        if len(candidates) == 0:
            continue
        pivot_row = rank + candidates[0]
        if pivot_row != rank:
            matrix[[rank, pivot_row]] = matrix[[pivot_row, rank]]
            row_order[[rank, pivot_row]] = row_order[[pivot_row, rank]]

        pivot_rows.append(row_order[rank])

        for r in range(m):
            if r != rank and matrix[r, col] == 1:
                matrix[r] ^= matrix[rank]

        rank += 1
        if rank == m:
            break

    return pivot_rows


def gaussian_elimination_with_forced_generators(matrix, row_order, must_include):
    """
    Perform Gaussian elimination while ensuring two specific Pauli strings (rows)
    are included in the generators.

    Args:
        matrix: binary numpy array of shape (m, n)
        row_order: numpy array indicating the original order of rows
        must_include: list or set of two original row indices to include as pivots

    Returns:
        pivot_rows: list of row indices (from original row_order) used as generators
    """
    m, n = matrix.shape
    rank = 0
    pivot_rows = []
    included = set()

    for col in range(n):
        candidates = np.where(matrix[rank:, col] == 1)[0]
        if len(candidates) == 0:
            continue

        # Find a pivot that is one of the must_include rows (if not yet included)
        forced_pivot = None
        for c in candidates:
            candidate_index = row_order[rank + c]
            if candidate_index in must_include and candidate_index not in included:
                forced_pivot = rank + c
                break
        if forced_pivot is None:
            # Fall back to normal pivoting
            forced_pivot = rank + candidates[0]

        if forced_pivot != rank:
            matrix[[rank, forced_pivot]] = matrix[[forced_pivot, rank]]
            row_order[[rank, forced_pivot]] = row_order[[forced_pivot, rank]]

        current_row_index = row_order[rank]
        pivot_rows.append(current_row_index)
        if current_row_index in must_include:
            included.add(current_row_index)

        for r in range(m):
            if r != rank and matrix[r, col] == 1:
                matrix[r] ^= matrix[rank]

        rank += 1
        if rank == m:
            break

    # Final check: ensure both must_include are present
    if not all(idx in pivot_rows for idx in must_include):
        raise ValueError(f"Could not include both required rows {must_include} as generators.")

    return pivot_rows

def analyze_pauli_dependencies(paulis, num_edges, use_window):
    binary = np.array([pauli_to_binary(p) for p in paulis])
    weights = np.sum(binary, axis=1)
    sorted_indices = np.argsort(weights)
    min_dependents = 10000

    matrix = binary[sorted_indices].copy()
    row_order = sorted_indices.copy()
    if not use_window:
        pivot_rows = gaussian_elimination(matrix, row_order)
    elif num_edges > 0:
        pivot_rows = gaussian_elimination_with_forced_generators(matrix, row_order, [0, 1])
    else:
        pivot_rows = gaussian_elimination(matrix, row_order)

    generator_indices = sorted(pivot_rows)
    generators = [paulis[i] for i in generator_indices]
    dependents = [paulis[i] for i in range(len(paulis)) if i not in generator_indices]

    # Build dependency counts and mapping
    gen_indices = [paulis.index(g) for g in generators]
    G = binary[gen_indices]

    dependent_to_generators = {}

    for d in dependents:
        d_idx = paulis.index(d)
        target = binary[d_idx]
        x = solve_linear_gf2(G, target)
        if x is None:
            dependent_to_generators[d] = ["(Could not solve)"]
            continue

        involved_gens = [generators[j] for j in range(len(generators)) if x[j] == 1]
        dependent_to_generators[d] = involved_gens
        if len(involved_gens) < min_dependents:
            min_dependents = len(involved_gens)
    return generators, dependents, dependent_to_generators, min_dependents

