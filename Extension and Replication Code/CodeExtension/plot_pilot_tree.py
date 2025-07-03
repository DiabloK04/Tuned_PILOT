import networkx as nx
import uuid
from networkx.drawing.nx_pydot import to_pydot
import os

"""
File: plot_pilot_tree.py
Author: Ivo Klazema
Description: This file creates a plot of the tree fitted by the PILOT model.
"""


def format_model(model_terms):
    """
    Format the total linear model for display in leaf nodes.
    """
    terms = []
    for feature_id, (coef, intercept) in sorted(model_terms.items()):
        terms.append(f"{coef:+.2f}*X{feature_id}")
    # Sum all intercepts across features for the global intercept
    total_intercept = sum(inter for _, inter in model_terms.values())
    terms.append(f"{total_intercept:+.2f}")
    return " + ".join(terms)


def visualize_tree_box_graphviz(model_tree, output_path="pilot_tree.png"):
    """
    Visualize a PILOT model tree using Graphviz with box-shaped nodes.
    Edge labels indicate the linear model added at each split, the leaf nodes show the final linear model.
    """
    G = nx.DiGraph()

    def add_nodes_edges(node, parent_id=None, model_terms=None, is_left=None, incoming_edge_label=None):
        # Initialize or copy cumulative model terms
        cum_terms = {} if model_terms is None else model_terms.copy()

        # Create a unique ID for this graph node
        current_id = str(uuid.uuid4())[:6]

        # Build the node label (show split rule for decision nodes)
        label = f"{node.node}"
        if node.node not in ["lin", "con"] and node.pivot:
            label += f"\nX{node.pivot[0]} <= {node.pivot[1]:.2f}"

        # If this is a leaf, append the full linear model
        if node.left is None and node.right is None:
            label += f"\n{format_model(cum_terms)}"

        # Add the node to the graph
        G.add_node(
            current_id,
            label=label,
            shape="box",
            style="filled",
            fillcolor="lightblue"
        )

        # Connect to the parent with the incoming edge label
        if parent_id:
            G.add_edge(parent_id, current_id)
            if incoming_edge_label:
                G[parent_id][current_id]["label"] = incoming_edge_label

        # Recurse on the left child, computing its edge boost
        if node.left:
            child_terms = cum_terms.copy()
            edge_label = ""
            if node.pivot and node.lm_l is not None:
                fid = node.pivot[0]
                coef, intercept = node.lm_l
                prev_coef, prev_intercept = child_terms.get(fid, (0.0, 0.0))
                child_terms[fid] = (prev_coef + coef, prev_intercept + intercept)
                edge_label = f"{coef:+.5f}*X{fid} {intercept:+.5f}_{coef > 0}"
            add_nodes_edges(
                node.left,
                parent_id=current_id,
                model_terms=child_terms,
                is_left=True,
                incoming_edge_label=edge_label
            )

        # Recurse on the right child
        if node.right:
            child_terms = cum_terms.copy()
            edge_label = ""
            if node.pivot and node.lm_r is not None:
                fid = node.pivot[0]
                coef, intercept = node.lm_r
                prev_coef, prev_intercept = child_terms.get(fid, (0.0, 0.0))
                child_terms[fid] = (prev_coef + coef, prev_intercept + intercept)
                edge_label = f"{coef:+.5f}*X{fid} {intercept:+.5f}"
            add_nodes_edges(
                node.right,
                parent_id=current_id,
                model_terms=child_terms,
                is_left=False,
                incoming_edge_label=edge_label
            )

    # Start recursion from the root
    add_nodes_edges(model_tree)

    # Export the graph to a PNG file
    pydot_graph = to_pydot(G)
    pydot_graph.write_png(output_path)
    print(f"Tree saved to: {os.path.abspath(output_path)}")
