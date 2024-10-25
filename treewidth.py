import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import treewidth_min_degree
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import numpy as np

# Define the graph representing the complex Sparse Neural Network
G = nx.Graph()
edges = [
    ('Input_1', 'Hidden_1'), ('Input_1', 'Hidden_2'),
    ('Input_2', 'Hidden_1'), ('Input_2', 'Hidden_3'),
    ('Input_3', 'Hidden_2'), ('Input_3', 'Hidden_3'),
    ('Input_4', 'Hidden_3'), ('Input_4', 'Hidden_4'),
    ('Hidden_1', 'Hidden_5'), ('Hidden_1', 'Hidden_6'),
    ('Hidden_2', 'Hidden_5'), ('Hidden_2', 'Hidden_7'),
    ('Hidden_3', 'Hidden_6'), ('Hidden_3', 'Hidden_7'),
    ('Hidden_4', 'Hidden_7'), ('Hidden_4', 'Hidden_8'),
    ('Hidden_5', 'Hidden_9'), ('Hidden_6', 'Hidden_9'),
    ('Hidden_7', 'Hidden_9'), ('Hidden_8', 'Hidden_9'),
    ('Hidden_9', 'Output_1'), ('Hidden_9', 'Output_2'),
    ('Hidden_9', 'Output_3')
]
G.add_edges_from(edges)

# Compute the treewidth and tree decomposition
treewidth, tree_decomposition = treewidth_min_degree(G)
print(f"Treewidth of sparse: {treewidth}")
print(f"Tree Decomposition: {tree_decomposition}")

# Generate a list of distinct colors
distinct_colors = list(mcolors.TABLEAU_COLORS.values())

# Ensure we have enough colors for the number of bags
if len(distinct_colors) < len(tree_decomposition):
    distinct_colors = list(mcolors.CSS4_COLORS.values())


# Function to adjust positions within each bag
def adjust_bag_positions(pos, tree_decomposition):
    scale_factor = 1.5  # Increase this factor to add more space between bags
    for i, bag in enumerate(tree_decomposition):
        bag_nodes = [node for node in pos if f"{i}_" in node]
        bag_center = sum([np.array(pos[node]) for node in bag_nodes]) / len(bag_nodes)
        for j, node in enumerate(bag_nodes):
            angle = 2 * np.pi * j / len(bag_nodes)
            pos[node] = bag_center + 0.3 * np.array([np.cos(angle), np.sin(angle)])  # Increase spacing within bags
        pos.update({node: scale_factor * np.array(position) for node, position in pos.items()})
    return pos


# Function to plot the tree decomposition with colored nodes and red circles for bags
def plot_tree_decomposition(tree_decomposition):
    T = nx.Graph()

    # Add nodes for the contents of each bag
    for i, bag in enumerate(tree_decomposition):
        bag_list = list(bag)
        for node in bag_list:
            T.add_node(f"{i}_{node}", label=node, color=distinct_colors[i % len(distinct_colors)])

    # Define the edges between bags based on tree decomposition
    for i in range(len(tree_decomposition) - 1):
        bag1, bag2 = list(tree_decomposition)[i], list(tree_decomposition)[i + 1]
        common_nodes = set(bag1).intersection(set(bag2))
        if common_nodes:
            node = list(common_nodes)[0]
            T.add_edge(f"{i}_{node}", f"{i + 1}_{node}")

    pos = nx.spring_layout(T, k=0.3)  # Initial layout with increased spacing
    pos = adjust_bag_positions(pos, tree_decomposition)  # Adjust positions within bags
    labels = nx.get_node_attributes(T, 'label')
    colors = [T.nodes[node]['color'] for node in T.nodes]

    plt.figure(figsize=(14, 10))
    nx.draw(T, pos, node_size=400, node_color=colors, font_size=10, font_weight='bold', edge_color='orange')
    nx.draw_networkx_labels(T, pos, labels, font_size=10, font_weight='bold', verticalalignment='bottom')

    # Draw ellipses around the nodes in each bag
    ax = plt.gca()
    for i, bag in enumerate(tree_decomposition):
        bag_nodes = [f"{i}_{node}" for node in bag]
        bag_pos = [pos[node] for node in bag_nodes]
        if bag_pos:  # Check if there are nodes to plot
            x_vals, y_vals = zip(*bag_pos)
            center_x, center_y = sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)
            width = (max(x_vals) - min(x_vals)) * 1.5  # Increased size of ellipses
            height = (max(y_vals) - min(y_vals)) * 1.5  # Increased size of ellipses
            ellipse = Ellipse((center_x, center_y), width, height, edgecolor='red', facecolor='none', lw=2, alpha=0.5)
            ax.add_patch(ellipse)

    plt.show()


# Plot the tree decomposition
plot_tree_decomposition(tree_decomposition)

# Define the layers
input_layer = ['Input_1', 'Input_2', 'Input_3', 'Input_4']
hidden_layer1 = ['Hidden_1', 'Hidden_2', 'Hidden_3', 'Hidden_4']
hidden_layer2 = ['Hidden_5', 'Hidden_6', 'Hidden_7', 'Hidden_8']
hidden_layer3 = ['Hidden_9']
output_layer = ['Output_1', 'Output_2', 'Output_3']


# Define the positions for a clear layout
def get_positions(layers):
    pos = {}
    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            pos[node] = (i, -j)
    return pos


# Create graphs for DNN and SNN
def create_dnn_graph():
    G = nx.DiGraph()
    G.add_nodes_from(input_layer + hidden_layer1 + hidden_layer2 + hidden_layer3 + output_layer)

    for input_node in input_layer:
        for hidden_node in hidden_layer1:
            G.add_edge(input_node, hidden_node)

    for hidden_node1 in hidden_layer1:
        for hidden_node2 in hidden_layer2:
            G.add_edge(hidden_node1, hidden_node2)

    for hidden_node2 in hidden_layer2:
        for hidden_node3 in hidden_layer3:
            G.add_edge(hidden_node2, hidden_node3)

    for hidden_node3 in hidden_layer3:
        for output_node in output_layer:
            G.add_edge(hidden_node3, output_node)

    return G


def create_snn_graph():
    G = nx.DiGraph()
    G.add_nodes_from(input_layer + hidden_layer1 + hidden_layer2 + hidden_layer3 + output_layer)

    edges = [
        ('Input_1', 'Hidden_1'), ('Input_1', 'Hidden_2'),
        ('Input_2', 'Hidden_1'), ('Input_2', 'Hidden_3'),
        ('Input_3', 'Hidden_2'), ('Input_3', 'Hidden_3'),
        ('Input_4', 'Hidden_3'), ('Input_4', 'Hidden_4'),
        ('Hidden_1', 'Hidden_5'), ('Hidden_1', 'Hidden_6'),
        ('Hidden_2', 'Hidden_5'), ('Hidden_2', 'Hidden_7'),
        ('Hidden_3', 'Hidden_6'), ('Hidden_3', 'Hidden_7'),
        ('Hidden_4', 'Hidden_7'), ('Hidden_4', 'Hidden_8'),
        ('Hidden_5', 'Hidden_9'), ('Hidden_6', 'Hidden_9'),
        ('Hidden_7', 'Hidden_9'), ('Hidden_8', 'Hidden_9'),
        ('Hidden_9', 'Output_1'), ('Hidden_9', 'Output_2'),
        ('Hidden_9', 'Output_3')
    ]

    G.add_edges_from(edges)

    return G


# Get positions for nodes
positions = get_positions([input_layer, hidden_layer1, hidden_layer2, hidden_layer3, output_layer])

# Create DNN and SNN graphs
dnn_graph = create_dnn_graph()
snn_graph = create_snn_graph()

# Plot the DNN
plt.figure(figsize=(14, 7))
plt.subplot(121)
nx.draw(dnn_graph, positions, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold',
        arrowsize=20, edge_color='lightgray')
plt.title('Deep Neural Network (DNN)')

# Plot the SNN
plt.subplot(122)
nx.draw(snn_graph, positions, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold',
        arrowsize=20, edge_color='lightgray')
plt.title('Sparse Neural Network (SNN)')

plt.show()
