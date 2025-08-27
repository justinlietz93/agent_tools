"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import networkx as nx
from typing import Dict

def calculate_graph_edit_distance(graph1: nx.Graph, graph2: nx.Graph) -> float:
    """
    Calculates the graph edit distance between two graphs.

    Note: This is a computationally expensive operation.
    
    Parameters
    ----------
    graph1 : nx.Graph
        The first graph.
    graph2 : nx.Graph
        The second graph.

    Returns
    -------
    float
        The graph edit distance between the two graphs.
    """
    return nx.graph_edit_distance(graph1, graph2)

def calculate_pagerank(graph: nx.Graph, alpha: float = 0.85) -> Dict[Any, float]:
    """
    Calculates the PageRank of the nodes in a graph.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph object.
    alpha : float, optional
        The damping parameter for PageRank, by default 0.85.

    Returns
    -------
    Dict[Any, float]
        A dictionary where keys are nodes and values are their PageRank scores.
    """
    return nx.pagerank(graph, alpha=alpha)
