"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import networkx as nx
import numpy as np
from typing import List

def apply_structural_plasticity(
    graph: nx.Graph,
    bdnf_levels: dict,
    growth_threshold: float = 0.8,
    pruning_threshold: float = 0.1,
    rewiring_prob: float = 0.1
):
    """
    Applies structural plasticity rules (growth, pruning, rewiring) to a graph.

    Parameters
    ----------
    graph : nx.Graph
        The graph to modify.
    bdnf_levels : dict
        A dictionary where keys are node pairs (u, v) and values are their
        BDNF proxy levels.
    growth_threshold : float, optional
        The BDNF level above which a new connection might form, by default 0.8.
    pruning_threshold : float, optional
        The weight below which an existing connection might be pruned, by default 0.1.
    rewiring_prob : float, optional
        The probability of rewiring a pruned connection, by default 0.1.
    """
    # Pruning
    for u, v, data in list(graph.edges(data=True)):
        if data.get('weight', 1.0) < pruning_threshold:
            graph.remove_edge(u, v)
            # Rewiring
            if np.random.rand() < rewiring_prob:
                # A simple rewiring strategy: connect u to a random other node
                other_nodes = list(set(graph.nodes()) - {u, v})
                if other_nodes:
                    new_neighbor = np.random.choice(other_nodes)
                    graph.add_edge(u, new_neighbor, weight=pruning_threshold)

    # Growth
    for (u, v), bdnf in bdnf_levels.items():
        if bdnf > growth_threshold and not graph.has_edge(u, v):
            graph.add_edge(u, v, weight=growth_threshold)
