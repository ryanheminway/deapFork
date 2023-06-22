# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:14:17 2023

@author: Ryan Heminway

This module provides graph visualization support for EANT individuals produced
using the DEAP extensions. 
"""
from eant import Genome, Node, InputNode, Neuron, Jumper

def graph_eant(genome, label_renaming_map=None):
    """
    Construct the graph of a genome. It returns in order a node list, an edge list, and a dictionary of the per node
    labels. The node are represented by numbers, the edges are tuples connecting two nodes (number), and the labels are
    values of a dictionary for which keys are the node numbers.

    :param genome: :class:`~deap.eant.Genome`
    :param label_renaming_map: dict, which maps the old name of a primitive (or a linking function)
        to a new one for better visualization. The default label for each node is just the name of the primitive
        placed on this node. For example, you may provide ``renamed_labels={'and_': 'and'}``.
    :return: A node list, an edge list, and a dictionary of labels.

    You can visualize a genome and export the tree visualization to an image file directly using the
    :func:`export_eant_tree` function.
    """
    nodes = []
    edges = []
    labels = {}
    input_id_lookup = {}
    
    if isinstance(genome, Genome):
        # Add neurons as nodes
        for n in genome.neuron_list:
            nodes.append(n.unique_id)
        # Add inputs as nodes
        input_id = max(nodes) + 1
        for i in genome.input_list:
            nodes.append(input_id)
            labels[input_id] = i.name
            input_id_lookup[i.name] = input_id
            input_id += 1
            
        # Calculate edges from genome
        i = 0
        while i < len(genome):
            node, weight = genome[i]
            # Only calculate edges at subnetworks
            k = i
            if isinstance(node, Neuron):
                for j in range(node.arity):
                    in_node, in_weight = genome[k + j + 1]
                    if isinstance(in_node, InputNode):
                        # Edge is (from_id, to_id, edge_label)
                        edges.append((input_id_lookup[in_node.name], node.unique_id, in_weight))
                    else: # in_node is neuron or jumper
                        if isinstance(in_node, Jumper) and in_node.recurrent:
                            in_weight = "REC: " + str(in_weight)
                        edges.append((in_node.unique_id, node.unique_id, in_weight))
                        # Skip over sub-networks, these aren't edges
                        if isinstance(in_node, Neuron):
                            # Traverse through sub-sub-network
                            constructed = False
                            # Track sub-genome by input/output differences
                            tracking_value = 0
                            while (not constructed):
                                sub_node, _ = genome[k + j + 1]
                                tracking_value += sub_node.tracking_value()
                                k += 1
                                if tracking_value == 1:
                                    k -= 1
                                    constructed = True
            i += 1
        
    else:
        raise TypeError('Only an argument of type Genome is acceptable. The provided '
                        'genome type is {}.'.format(type(genome)))
    # rename_labels labels
    if label_renaming_map is not None:
        for k, v in labels.items():
            if v in label_renaming_map:
                labels[k] = label_renaming_map[v]
    return nodes, edges, labels


def export_eant_tree(genome, label_renaming_map=None, file='tree.png'):
    """
    Construct the graph of a Genome containing Nodes and then export it to a *file*.

    :param genome: :class:`~deap.eant.Genome`, the genotype of an individual
    :param file: str, the file path to draw the expression tree, which may be a relative or absolute one.
        If no extension is included in *file*, then the default extension 'png' is used.

    .. note::
        This function currently depends on the :mod:`graphviz` module to render the tree. Please first install the
        `graphviz <https://pypi.org/project/graphviz/>`_ module before using this function.
        Alternatively, you can always obtain the raw graph data with the :func:`graph` function, then postprocess the
        data and render them with other tools as you want.
    """
    import graphviz as gv
    import os.path

    _, edges, labels = graph_eant(genome, label_renaming_map)
    file_name, ext = os.path.splitext(file)
    ext = ext.lstrip('.')
    g = gv.Digraph(format=ext)
    for name, label in labels.items():
        g.node(str(name), str(label))  # add node
    for name1, name2, label in edges:
        g.edge(str(name1), str(name2), label=str(""), dir="Forward")  # add edge # label = str(label)
    g.render(file_name)  
    
    
__all__ = ['export_eant_tree',]