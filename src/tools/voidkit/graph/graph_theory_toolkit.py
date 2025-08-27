"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import networkx as nx
from typing import Dict, Any, List
import numpy as np

def calculate_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calculates a set of standard metrics for a given graph.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph object.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the calculated graph metrics, including:
        - 'num_nodes': Number of nodes.
        - 'num_edges': Number of edges.
        - 'density': The density of the graph.
        - 'avg_degree': The average degree of the nodes.
        - 'avg_clustering_coefficient': The average clustering coefficient.
        - 'avg_shortest_path_length': The average shortest path length (if connected).
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")

    metrics = {}
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    metrics['density'] = nx.density(graph)
    
    degrees = [d for n, d in graph.degree()]
    if degrees:
        metrics['avg_degree'] = np.mean(degrees)
    else:
        metrics['avg_degree'] = 0

    metrics['avg_clustering_coefficient'] = nx.average_clustering(graph)

    if nx.is_connected(graph):
        metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)
    else:
        metrics['avg_shortest_path_length'] = float('inf') # Or handle disconnected components separately

    return metrics

def detect_communities(graph: nx.Graph, method: str = 'louvain') -> List[List[Any]]:
    """
    Detects communities in a graph using a specified algorithm.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph object.
    method : str, optional
        The community detection algorithm to use. Currently, only 'louvain' is supported.
        Default is 'louvain'.

    Returns
    -------
    List[List[Any]]
        A list of lists, where each inner list contains the nodes of a community.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")

    if method == 'louvain':
        # Note: The 'louvain' algorithm is in the 'community' package, which is a separate dependency.
        # For simplicity, we will use the greedy modularity maximization from NetworkX,
        # which is similar in spirit.
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(graph))
            return [list(c) for c in communities]
        except ImportError:
            raise ImportError("The 'louvain' method requires the 'python-louvain' or 'community' package, or use networkx's greedy_modularity_communities.")
    else:
        raise ValueError(f"Method '{method}' is not supported. Currently, only 'louvain' is available.")
