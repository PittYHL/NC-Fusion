import networkx as nx
from itertools import combinations
import copy
from commuting_graph import commute_check, generate_commutation_graphs
from new_gaussian import analyze_pauli_dependencies

def choose_first_anticomm_pair(G_anticomm, ungroup, pauli_count_constraint):
    """
    Select the first anticommuting pair found in `ungroup` based on order,
    and prepend it with the first (pauli_count_constraint - 2) Pauli strings from `ungroup`,
    skipping duplicates.

    Parameters
    ----------
    G_anticomm : networkx.Graph
        Graph where nodes are Pauli strings and edges indicate anticommutation.
    ungroup : list of str
        Ordered list of Pauli strings.
    pauli_count_constraint : int
        Total number of Pauli strings to return.

    Returns
    -------
    list of str
        Subset of Pauli strings satisfying the constraint, or empty list if not found.
    """
    # if pauli_count_constraint < 2:
    #     raise ValueError("pauli_count_constraint must be at least 2.")

    ungroup_set = set(ungroup)
    anticomm_pair = None

    # Step 1: Find first anticomm pair from G_anticomm in the order of ungroup
    for u, v in G_anticomm.edges:
        if u in ungroup_set and v in ungroup_set:
            idx_u = ungroup.index(u)
            idx_v = ungroup.index(v)
            if idx_u < idx_v:
                anticomm_pair = (u, v)
            else:
                anticomm_pair = (v, u)
            break

    if anticomm_pair is None:
        return ungroup

    # Step 2: Start result with the anticomm pair
    result = [anticomm_pair[0], anticomm_pair[1]]
    used = set(result)

    # Step 3: Add prefix elements from ungroup, skipping those in the pair
    for p in ungroup:
        if p not in used:
            result.append(p)
            used.add(p)
        if len(result) == pauli_count_constraint:
            return result

    return result

def get_edges_between_nodes(G, node_list):
    """
    Return list of edges from graph G that connect pairs in node_list.
    """
    subgraph = G.subgraph(node_list)
    return list(subgraph.edges())

def grading(generators, dependent_to_generators, available): #grade all combinations of potential generators
    grade = 0
    all_combine = []  # used to generate all combinations of the generator
    for i in range(len(generators), 0, -1):
        all_combine = all_combine + list(combinations(generators, i))
    for value in dependent_to_generators.values():
        dependent = copy.deepcopy(value)
        dependent.sort()
        for combine in all_combine:
            set1 = set(combine)
            list1 = list(combine)
            list1.sort()
            # if len(list1[0]) == 2:
            #     for i in range(len(list1)):
            #         list1[i] = list1[i][1]
            if list1 == dependent:
                grade += 3
                break
            elif set1.issubset(set(dependent)) and available >= (len(dependent) - len(combine)): #need to set constrain
                grade += 1
                break
    return grade

def get_dependent(generators, dependent_to_generators, dependents):
    all_combine = []  #used to generate all combinations of the generator
    chosen_dependents = []
    generator_paulis = list(generators.keys())
    for i in range(2, len(generators) + 1):
        all_combine = all_combine + list(combinations(generator_paulis, i))
    for combine in all_combine:
        list1 = []
        for pauli in combine:
            list1.append(generators[pauli][0])
        list1.sort()
        for key, value in dependent_to_generators.items():
            value.sort()
            if list1 == value:
                chosen_dependents.append(key)
    for dependent in chosen_dependents:
        dependents.remove(dependent)
        dependent_to_generators.pop(dependent)
    return chosen_dependents

def can_group_nodes_all_paths_internal(G, node_list, cut_off):
    node_set = set(n for n in node_list if n in G.nodes)
    filtered_list = [n for n in node_list if n in G.nodes]

    for u in filtered_list:
        for v in filtered_list:
            if u == v:
                continue
            # Check all paths from u to v
            try:
                for path in nx.all_simple_paths(G, source=u, target=v, cutoff=cut_off):
                    if not all(n in node_set for n in path):
                        return False
            except nx.NetworkXNoPath:
                continue  # No path between u and v is fine
    return True

def can_extend_group_with_node(G, node_list, new_node, cut_off):  #same as the previous one, but adding new node
    # Ensure all nodes are in the graph
    node_list = [n for n in node_list if n in G]
    if new_node not in G:
        return True  # Skipping invalid new_node

    group = set(node_list + [new_node])

    for u in node_list:
        # check all paths from u to new_node
        try:
            for path in nx.all_simple_paths(G, source=u, target=new_node, cutoff=cut_off):
                if not all(n in group for n in path):
                    return False
        except nx.NetworkXNoPath:
            pass  # no path is fine

        # check all paths from new_node to u
        try:
            for path in nx.all_simple_paths(G, source=new_node, target=u, cutoff=cut_off):
                if not all(n in group for n in path):
                    return False
        except nx.NetworkXNoPath:
            pass  # no path is fine

    return True


# def connectivity_check(chosen_generators, combinine, dag, dependent_to_generators, index, pauli_index): #only check the connectivity of neighbour pauli strings
#     chosen = []
#     chosen_indexes = [index]
#     for pauli in chosen_generators.values():
#         chosen.append(pauli[0])
#     for pauli in combinine:
#         for key, value in pauli_index.items():
#             if value[0] == pauli:
#                 chosen_indexes.append(key)
#                 break
#     # chosen = chosen + list(combinine)
#     # for dependent in dependent_to_generators:
#     #     is_subset = set(dependent_to_generators[dependent]).issubset(set(chosen))
#     #     if is_subset:
#     #         chosen.append(dependent)
#     #         for key, value in pauli_index.items():
#     #             if value[0] == dependent:
#     #                 chosen_indexes.append(key)
#     #                 break  # Stop after finding the first match
#     if can_group_nodes_all_paths_internal(dag, chosen_indexes, len(chosen_indexes)):
#         return True, chosen_indexes
#     else:
#         return False, []

def connectivity_check(chosen_indexes, combinine, dag, pauli_index, cut_off = 0): #only check the connectivity of neighbour pauli strings
    for pauli in combinine:
        for key, value in pauli_index.items():
            if value[0] == pauli:
                chosen_indexes.append(key)
                break
    if cut_off == 0:
        cut_off = len(chosen_indexes)
    if can_group_nodes_all_paths_internal(dag, chosen_indexes, cut_off):
        return True, chosen_indexes
    else:
        return False, []

def get_available_pair(chosen_generators, generators, G_anticomm, G_comm, num_qubits, stage): #obtain the available pairs first
    if stage * 2 != len(chosen_generators):
        return []
    generators_transformed = {} #record each transformed generator
    available_pairs = []
    for generator in generators:
        first_term = '' #the first term of the transformed generator
        for i in range(stage):
            P1 = chosen_generators[i * 2 + 1]
            P2 = chosen_generators[i * 2 + 2]
            if first_term != '':
                P1_first_commute_relation = commute_check(P1[1][:i], first_term) #check if the first part is commute or not
                P2_first_commute_relation = commute_check(P2[1][:i], first_term)
            else:
                P1_first_commute_relation = 1
                P2_first_commute_relation = 1
            if P1_first_commute_relation and P2_first_commute_relation and G_comm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'I'
                continue
            if P1_first_commute_relation and P2_first_commute_relation and G_comm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'X'
                continue
            if P1_first_commute_relation and P2_first_commute_relation and G_anticomm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'Z'
                continue
            if P1_first_commute_relation and P2_first_commute_relation and G_anticomm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'Y'
                continue

            if P1_first_commute_relation and P2_first_commute_relation == 0 and G_comm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'X'
                continue
            if P1_first_commute_relation and P2_first_commute_relation == 0 and G_comm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'I'
                continue
            if P1_first_commute_relation and P2_first_commute_relation == 0 and G_anticomm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'Y'
                continue
            if P1_first_commute_relation and P2_first_commute_relation == 0 and G_anticomm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'Z'
                continue

            if P1_first_commute_relation == 0 and P2_first_commute_relation and G_comm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'Z'
                continue
            if P1_first_commute_relation == 0 and P2_first_commute_relation and G_comm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'Y'
                continue
            if P1_first_commute_relation == 0 and P2_first_commute_relation and G_anticomm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'I'
                continue
            if P1_first_commute_relation == 0 and P2_first_commute_relation and G_anticomm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'X'
                continue

            if P1_first_commute_relation == 0 and P2_first_commute_relation == 0 and G_comm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'Y'
                continue
            if P1_first_commute_relation == 0 and P2_first_commute_relation == 0 and G_comm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'Z'
                continue
            if P1_first_commute_relation == 0 and P2_first_commute_relation == 0 and G_anticomm.has_edge(P1[0], generator) and G_comm.has_edge(P2[0], generator):
                first_term = first_term + 'X'
                continue
            if P1_first_commute_relation == 0 and P2_first_commute_relation == 0 and G_anticomm.has_edge(P1[0],generator) and G_anticomm.has_edge(P2[0], generator):
                first_term = first_term + 'I'
                continue
        generators_transformed.update({generator:first_term})
        #check if any pairs exist
    all_combine = list(combinations(generators, 2))
    for combinine in all_combine:
        if commute_check(generators_transformed[combinine[0]], generators_transformed[combinine[1]]) and G_anticomm.has_edge(combinine[0], combinine[1]):
            available_pairs.append({combinine[0]:generators_transformed[combinine[0]] + 'X' + (num_qubits- stage - 1)*'I', combinine[1]:generators_transformed[combinine[1]] + 'Z' + (num_qubits- stage - 1)*'I'})
        elif commute_check(generators_transformed[combinine[1]], generators_transformed[combinine[0]]) == 0 and G_comm.has_edge(combinine[0], combinine[1]):
            available_pairs.append({combinine[0]:generators_transformed[combinine[0]] + 'X' + (num_qubits- stage - 1)*'I', combinine[1]:generators_transformed[combinine[1]] + 'Z' + (num_qubits- stage - 1)*'I'})
    return available_pairs, generators_transformed

def remove_selected(G_com, G_anticomm, current_generator, current_dependent):
    if 1 in current_generator:
        pauli = current_generator[1][0]
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    if 2 in current_generator:
        pauli =  current_generator[2][0]
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    if 3 in current_generator:
        pauli = current_generator[3][0]
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    if 4 in current_generator:
        pauli = current_generator[4][0]
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    if 5 in current_generator:
        pauli = current_generator[5][0]
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    if 6 in current_generator:
        pauli = current_generator[6][0]
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    for pauli in current_dependent:
        # G_com.remove_node(pauli)
        G_anticomm.remove_node(pauli)
    return G_com, G_anticomm

def generate_group(current_generator, current_dependent):
    group = {1: [], 2: [], 3: [], "dependent": current_dependent, "commute": []}
    for generator_key, group_key in ((1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)):
        if generator_key in current_generator:
            group[group_key].append(current_generator[generator_key][0])
    return group


def permute_keys_after_weight_sort(pauli_dict):

    keys = [1, 2, 3]
    key_permutations = list(permutations(keys))
    result = []

    # Step 1: Sort each group's strings by weight
    sorted_groups = {}
    for k in keys:
        sorted_groups[k] = sorted(pauli_dict[k], key=pauli_weight)
    sorted_groups['dependent'] = pauli_dict['dependent']
    sorted_groups['commute'] = pauli_dict['commute']
    return [sorted_groups]

def add_commute(group, generators, budget, G_comm, num_qubits, ungroup, dag=None):
    if (budget == 1 and len(group[1]) != 2) or (budget == 2 and len(group[2]) != 2) or (budget == 3 and len(group[3]) != 2):
        return group, ungroup
    to_be_removed = []
    for g in generators:  # generators that commute with current
        # g = generators[0]
        if budget + len(group['commute']) == num_qubits:
            break
        if budget == 1 and len(group[1]) == 2 and G_comm.has_edge(group[1][0], g) and G_comm.has_edge(group[1][1], g) and len(group['commute']) < num_qubits:
            if dag is not None and (nx.has_path(dag, group[1][0], g) or nx.has_path(dag, group[1][1], g) or
            nx.has_path(dag, g, group[1][0]) or nx.has_path(dag, g, group[1][1])):
                break
            group['commute'].append(g)
            to_be_removed.append(g)
        elif (budget == 2 and len(group[1]) == 2 and len(group[2]) == 2 and G_comm.has_edge(group[1][0], g) and G_comm.has_edge(group[1][1], g)
              and G_comm.has_edge(group[2][0], g) and G_comm.has_edge(group[2][1], g)) and len(group['commute']) < num_qubits - 1:
            if dag is not None and (nx.has_path(dag, group[1][0], g) or nx.has_path(dag, group[1][1], g) or nx.has_path(dag, g, group[1][0]) or nx.has_path(dag, g, group[1][1])
            or nx.has_path(dag, group[2][0], g) or nx.has_path(dag, group[2][1], g) or nx.has_path(dag, g, group[2][0]) or nx.has_path(dag, g, group[2][1])):
                break
            group['commute'].append(g)
            to_be_removed.append(g)
        elif ((budget == 2 and len(group[1]) == 2 and len(group[2]) == 2 and len(group[3]) == 2 and G_comm.has_edge(group[1][0], g) and G_comm.has_edge(group[1][1], g)
              and G_comm.has_edge(group[2][0], g) and G_comm.has_edge(group[2][1], g) and G_comm.has_edge(group[3][0], g) and G_comm.has_edge(group[3][1], g))
              and len(group['commute']) < num_qubits - 2):
            if dag is not None and (nx.has_path(dag, group[1][0], g) or nx.has_path(dag, group[1][1], g) or nx.has_path(dag, g, group[1][0]) or nx.has_path(dag, g, group[1][1])
            or nx.has_path(dag, group[2][0], g) or nx.has_path(dag, group[2][1], g) or nx.has_path(dag, g, group[2][0]) or nx.has_path(dag, g, group[2][1])
            or nx.has_path(dag, group[3][0], g) or nx.has_path(dag, group[3][1], g) or nx.has_path(dag, g, group[3][0]) or nx.has_path(dag, g, group[3][1])):
                break
            group['commute'].append(g)
            to_be_removed.append(g)
            # new_qubits = get_global_active_qubits(new_pauli)
            # if set(active_qubits).isdisjoint(new_qubits):
            #     added_paulis.append(g)
            #     added_paulis_transformed.append(new_pauli)
    for g in to_be_removed:
        generators.remove(g)
        ungroup.remove(g)
    return group, ungroup

def _pauli_sort_key(p):
            non_i_pos = tuple(i for i, c in enumerate(p) if c != 'I')
            return (-len(non_i_pos), non_i_pos, p)

def grouping(pauli_strings, budget, pauli_count_constraint = 64, use_window = False, sorted_group = False): #budget: number of qubits reduced to
    minimum_constraint = 2 ** (2 * budget)
    if pauli_count_constraint < minimum_constraint:
        print("Warning: The pauli_count_constraint is too low!")
    if sorted_group:
        pauli_strings = sorted(pauli_strings, key=_pauli_sort_key)

    ungroup = copy.deepcopy(pauli_strings)
    num_qubits = len(pauli_strings[0])
    group = [] #resulting grouping
    no_commute_group = []
    G_comm, G_anticomm = generate_commutation_graphs(pauli_strings)
    while ungroup:
        # print('group', len(group))
        num_edges = len(G_anticomm.edges)
        if not use_window:
            new_ungroup = ungroup
        elif len(ungroup) > pauli_count_constraint and num_edges >= 1:
            new_ungroup = choose_first_anticomm_pair(G_anticomm, ungroup, pauli_count_constraint)
        elif num_edges >= 1:
            new_ungroup = choose_first_anticomm_pair(G_anticomm, ungroup, len(ungroup))
        else:
            new_ungroup = ungroup
        generators, dependents, dependent_to_generators, min_dependents = analyze_pauli_dependencies(new_ungroup, num_edges, use_window)
        # print('gaussian finished')
        current_generator = {} #for the chosen generator
        current_dependent = [] #for the chosen dependent
        anti_edges = get_edges_between_nodes(G_anticomm, generators) #anti-commuting edges between

        #if no commuting groups, add the rest back to the exisiting groups
        if anti_edges == [] and min_dependents > budget:
            no_commute_group = copy.deepcopy(group)
            ungroup_copy = copy.deepcopy(ungroup)
            for i in range(len(group)):
                if len(ungroup) == 0:
                    break
                _, ungroup = add_commute(group[i], generators, budget, G_comm, num_qubits, ungroup)
                if ungroup:
                    generators, dependents, dependent_to_generators, min_dependents = analyze_pauli_dependencies(
                    ungroup, 0, use_window)
            commute = []
            while (len(ungroup_copy) > 0):
                commute.append(ungroup_copy.pop(0))
                current_group = {1:[], 2:[], 3:[], 'dependent':[], 'commute':commute}
                no_commute_group.append(current_group)
                commute = []
            
            if len(ungroup) == 0:
                return group, no_commute_group

        if len(ungroup) <= budget:
            #need to change!
            if len(ungroup) == 1:
                current_group = {1:[ungroup[0]], 2:[], 3:[], 'dependent':[], 'commute':[]}
            elif len(ungroup) == 2:
                current_group = {1: [ungroup[0]], 2: [ungroup[1]], 3: [], 'dependent': [], 'commute':[]}
            elif len(ungroup) == 3:
                current_group = {1: [ungroup[0]], 2: [ungroup[1]], 3: [ungroup[2]], 'dependent': [], 'commute': []}
            group.append(current_group)
            return group

        # for the first qubit P1 and P2
        if anti_edges != []:
            grades = [] #rate each edge
            for edge in anti_edges:
                rating = grading(edge, dependent_to_generators, (budget - 1) * 2)
                grades.append(rating)
            chosen_pair = anti_edges[grades.index(max(grades))] #choose the best pair in terms of grade
            # add chosen pair to the current generator
            current_generator.update({1:[chosen_pair[0], 'X'+(num_qubits-1)*'I']})
            current_generator.update({2:[chosen_pair[1], 'Z'+(num_qubits-1)*'I']})
            # extract the dependent terms
            chosen_dependents = get_dependent(current_generator, dependent_to_generators, dependents)
            ungroup.remove(chosen_pair[0])
            ungroup.remove(chosen_pair[1])
            generators.remove(chosen_pair[0])
            generators.remove(chosen_pair[1])
            if chosen_dependents!= []:
                for dependent in chosen_dependents:
                    ungroup.remove(dependent)
                    current_dependent.append(dependent)
        # elif budget == 1:
        #     for pauli in ungroup:
        #         group.append([pauli])
        #     return group
        else:  #no anticommuting exist in generators
            grades = []
            for pauli in generators:
                rating = grading([pauli], dependent_to_generators, (budget - 1))
                grades.append(rating)
            chosen_pauli = generators[grades.index(max(grades))]
            #add chosen generator
            current_generator.update({1: [chosen_pauli, 'X' + (num_qubits - 1) * 'I']})
            ungroup.remove(chosen_pauli)
            generators.remove(chosen_pauli)
        if budget == 1:
            G_comm, G_anticomm = remove_selected(G_comm, G_anticomm, current_generator, current_dependent)
            current_group = generate_group(current_generator, current_dependent)
            group.append(current_group)
            # greedy_pauli, greeedy_circuit = greedy_circuit_generation(current_group)
            # resulting_paulis.append(greedy_pauli)
            # circuits.append(greeedy_circuit)
            continue

        # for second qubit
        available_pairs = []
        if len(current_generator) == 2:
            available_pairs, generators_transformed = get_available_pair(current_generator, generators, G_anticomm, G_comm, num_qubits, 1)
        if available_pairs != []:
            grades = []
            for pair in available_pairs:
                rating = grading([current_generator[1][0]] + [current_generator[2][0]] + list(pair.keys()), dependent_to_generators, (budget - 2) * 2)
                grades.append(rating)
            chosen_pair = available_pairs[grades.index(max(grades))]
            keys = list(chosen_pair.keys())
            # add chosen pair to the current generator
            current_generator.update({3: [keys[0], chosen_pair[keys[0]]]})
            current_generator.update({4: [keys[1], chosen_pair[keys[1]]]})
            ungroup.remove(keys[0])
            ungroup.remove(keys[1])
            generators.remove(keys[0])
            generators.remove(keys[1])
        else:  #only one generator has been selected for qubit 1
            grades = []
            current_paulis = []
            for key, value in current_generator.items():
                current_paulis.append(value[0])
            for pauli in generators:
                rating = grading(current_paulis + [pauli], dependent_to_generators, (budget - 2))
                grades.append(rating)
            chosen_pauli = generators[grades.index(max(grades))]
            current_generator.update({3: [chosen_pauli, 'IX' + (num_qubits - 2) * 'I']})
            ungroup.remove(chosen_pauli)
            generators.remove(chosen_pauli)

        chosen_dependents = get_dependent(current_generator, dependent_to_generators, dependents)
        if chosen_dependents != []:
            while len(current_generator) + len(current_dependent) < pauli_count_constraint and chosen_dependents != []:
                dependent = chosen_dependents.pop(0)
                ungroup.remove(dependent)
                current_dependent.append(dependent)
        if budget == 2:
            G_comm, G_anticomm = remove_selected(G_comm, G_anticomm, current_generator, current_dependent)
            current_group = generate_group(current_generator, current_dependent)
            group.append(current_group)
            # greedy_pauli, greeedy_circuit = greedy_circuit_generation(current_group)
            # resulting_paulis.append(greedy_pauli)
            # circuits.append(greeedy_circuit)
            continue

        # for third qubit
        available_pairs = []
        if len(current_generator) == 4:  # still possible has pairs
            available_pairs, generators_transformed = get_available_pair(current_generator, generators, G_anticomm,
                                                      G_comm, num_qubits, 2)
        if available_pairs != []:
            grades = []
            for pair in available_pairs:
                rating = grading([current_generator[1][0]] + [current_generator[2][0]] + [current_generator[3][0]] + [current_generator[4][0]] + list(pair.keys()), dependent_to_generators, (budget - 3) * 2)
                grades.append(rating)
            chosen_pair = available_pairs[grades.index(max(grades))]
            keys = list(chosen_pair.keys())
            # add chosen pair to the current generator
            current_generator.update({5: [keys[0], chosen_pair[keys[0]]]})
            current_generator.update({6: [keys[1], chosen_pair[keys[1]]]})
            ungroup.remove(keys[0])
            ungroup.remove(keys[1])
            generators.remove(keys[0])
            generators.remove(keys[1])
        else:
            grades = []
            current_paulis = []
            for key, value in current_generator.items():
                current_paulis.append(value[0])
            for pauli in generators:
                rating = grading(current_paulis + [pauli], dependent_to_generators, (budget - 3))
                grades.append(rating)
            chosen_pauli = generators[grades.index(max(grades))]
            current_generator.update({5: [chosen_pauli, 'IIX' + (num_qubits - 2) * 'I']})
            ungroup.remove(chosen_pauli)
            generators.remove(chosen_pauli)
        chosen_dependents = get_dependent(current_generator, dependent_to_generators, dependents)
        if chosen_dependents != []:
            while len(current_generator) + len(current_dependent) < pauli_count_constraint and chosen_dependents != []:
                dependent = chosen_dependents.pop(0)
                ungroup.remove(dependent)
                current_dependent.append(dependent)
        current_group = generate_group(current_generator, current_dependent)

        group.append(current_group)
        # greedy_pauli, greeedy_circuit = greedy_circuit_generation(current_group)
        # resulting_paulis.append(greedy_pauli)
        # circuits.append(greeedy_circuit)
    return group, no_commute_group


def pauli_weight(pauli_string):
    """Calculate weight = number of non-'I' characters in the Pauli string."""
    weight = 0
    for c in pauli_string:
        if c == 'X' or c == 'Z':
            weight += 1
        elif c == 'Y':
            weight += 2
    return weight

