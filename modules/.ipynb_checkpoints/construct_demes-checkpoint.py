import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# def deme_distribution(weights, total_demes, min_demes_per_group):
#     """
#     Distributes demes based on a greedy algorithm where each group 
#     initially gets a minimum number of demes, and the rest are distributed 
#     based on their weights.
#     """
#     n = len(weights)
#     # Initialize deme distribution with the minimum demes per group
#     demes = [min_demes_per_group] * n  

#     # Adjust the total number of demes after initial allocation
#     adjusted_total_demes = total_demes - n * min_demes_per_group

#     # Check if the remaining demes are enough to distribute
#     if adjusted_total_demes < 0:
#         raise ValueError("Total demes not sufficient to give everyone the minimum required.")

#     # Distribute the remaining demes using the greedy approach
#     for _ in range(adjusted_total_demes):
#         ratios = [weights[i] / (demes[i]) for i in range(n)]
#         max_index = ratios.index(max(ratios))
#         demes[max_index] += 1

#     return demes


# def deme_distribution(weights_group, total_demes, min_demes_per_group, max_demes_per_group):
#     """
#     Distributes demes based on a greedy algorithm where each group 
#     initially gets a minimum number of demes, and the rest are distributed 
#     based on their weights, with a cap on the maximum demes per group.
#     """
#     n = len(weights_group)
#     # Initialize deme distribution with the minimum demes per group
#     demes = [min_demes_per_group] * n  

#     # Adjust the total number of demes after initial allocation
#     adjusted_total_demes = total_demes - n * min_demes_per_group

#     # Check if the remaining demes are enough to distribute
#     if adjusted_total_demes < 0:
#         raise ValueError("Total demes not sufficient to give everyone the minimum required.")

#     # Distribute the remaining demes using the greedy approach
#     for _ in range(adjusted_total_demes):
#         # Update ratios only for those who haven't reached the max limit
#         ratios = [weights_group[i] / (demes[i]) if demes[i] < max_demes_per_group else 0 for i in range(n)]
#         max_index = ratios.index(max(ratios))

#         # Check if the highest ratio is zero, indicating everyone is at the max limit
#         if ratios[max_index] == 0:
#             break

#         demes[max_index] += 1

#     return demes

def deme_distribution(weights_group, total_demes, min_demes_per_group, max_demes_per_group):
    """
    Distributes demes based on a greedy algorithm where each group 
    initially gets a specified minimum number of demes, and the rest are 
    distributed based on their weights, with a cap on the specified 
    maximum demes per group.
    """
    n = len(weights_group)

    if not (len(min_demes_per_group) == len(max_demes_per_group) == n):
        raise ValueError("Length of min_demes_per_group and max_demes_per_group lists must match the number of groups.")

    # Initialize deme distribution with the specified minimum demes per group
    demes = min_demes_per_group.copy()

    # Adjust the total number of demes after initial allocation
    adjusted_total_demes = total_demes - sum(min_demes_per_group)

    # Check if the remaining demes are enough to distribute
    if adjusted_total_demes < 0:
        raise ValueError("Total demes not sufficient to give every group the minimum required.")

    # Distribute the remaining demes using the greedy approach
    for _ in range(adjusted_total_demes):
        # Update ratios only for those who haven't reached the max limit
        ratios = [weights_group[i] / demes[i] if demes[i] < max_demes_per_group[i] else 0 for i in range(n)]
        max_index = ratios.index(max(ratios))

        # Check if the highest ratio is zero, indicating every group is at the max limit
        if ratios[max_index] == 0:
            break

        demes[max_index] += 1

    return demes


# Initialize groups with named identifiers
def initialize_groups_with_named_groups(sorted_nodes, n):
    """ Initialize groups with sequential names """
    return {'G{}'.format(i): [sorted_nodes[i]] for i in range(n)}

# Initialize the group weights for groups with named identifiers
def initialize_group_weights_with_named_groups(groups, node_weights):
    """ Initialize the group weights for groups with named identifiers """
    return {group_name: node_weights[nodes[0]] for group_name, nodes in groups.items()}

# Calculate shortest distances
def calculate_shortest_distances(graph, nodes):
    """ Calculate shortest path distances for each node """
    return {node: nx.single_source_shortest_path_length(graph, node) for node in nodes}

# Assign a node to a group
def assign_node_to_group(node, groups, group_weights, shortest_distances, node_weights):
    """ Assign a node to a group based on shortest distances and group weights """
    min_distance = float('inf')
    candidate_groups = []

    for group, members in groups.items():
        group_distance = min(shortest_distances[node][member] for member in members)
        if group_distance < min_distance:
            min_distance = group_distance
            candidate_groups = [group]
        elif group_distance == min_distance:
            candidate_groups.append(group)

    chosen_group = min(candidate_groups, key=lambda g: group_weights[g])
    groups[chosen_group].append(node)
    group_weights[chosen_group] += node_weights[node]

# Main function to run algorithm and visualize the network
def run_algorithm_and_visualize(adj_matrix, node_weights, n, visualize='y'):
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_matrix(adj_matrix)
    node_names=list(node_weights.keys())
    mapping = {i: node_names[i] for i in range(len(node_names))}
    G = nx.relabel_nodes(G, mapping)

    # Calculate shortest path distances
    nodes = list(node_weights.keys())
    shortest_distances = calculate_shortest_distances(G, nodes)

    # Sort nodes by weight and initialize groups
    sorted_nodes = sorted(nodes, key=lambda x: node_weights[x], reverse=True)
    groups = initialize_groups_with_named_groups(sorted_nodes, n)
    group_weights = initialize_group_weights_with_named_groups(groups, node_weights)

    # Assign remaining nodes to groups
    for node in sorted_nodes[n:]:
        assign_node_to_group(node, groups, group_weights, shortest_distances, node_weights)

    if visualize == 'y':
        # Visualization
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(G)
        color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
        if n > len(color_list):
            color_list.extend(['#'+np.random.randint(100000, 999999) for _ in range(n - len(color_list))])

        # Dynamic color mapping
        color_map = []
        for node in G.nodes():
            group_index = None
            for group_name, members in groups.items():
                if node in members:
                    group_index = int(group_name[1:])
                    break
            color_map.append(color_list[group_index] if group_index is not None else 'black')

        # Calculate node sizes, normalized by the maximum weight
        max_weight = max(node_weights.values())
        node_sizes = [700 * node_weights[node] / max_weight for node in G.nodes()]

        nx.draw(G, pos, node_color=color_map, node_size=node_sizes, with_labels=False, font_size=12)

        # Add node weights as labels
        node_labels = {node: f"{node}\n({node_weights[node]})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        plt.show()

    return groups


def replace_keys(original_dict, new_keys):

    if len(new_keys) != len(original_dict):
        raise ValueError("The number of new keys must match the number of keys in the original dictionary.")

    return {new_key: original_dict[old_key] for new_key, old_key in zip(new_keys, original_dict.keys())}


def invert_group_location_dict(group_location_dict):

    location_group_dict = {}
    for group, locations in group_location_dict.items():
        for location in locations:
            location_group_dict[location] = group
    return location_group_dict