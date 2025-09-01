"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import networkx as nx
from typing import List

def coarse_grain_graph(
    graph: nx.Graph,
    partitions: List[List[int]]
) -> nx.Graph:
    """
    Coarse-grains a graph by merging nodes based on a given partition.

    Parameters
    ----------
    graph : nx.Graph
        The original graph.
    partitions : List[List[int]]
        A list of lists, where each inner list contains the nodes of a partition.

    Returns
    -------
    nx.Graph
        The coarse-grained graph.
    """
    coarse_graph = nx.Graph()
    
    # Add nodes to the coarse graph, one for each partition
    for i, partition in enumerate(partitions):
        coarse_graph.add_node(i, members=partition)
        
    # Add edges to the coarse graph
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            weight = 0
            for u in partitions[i]:
                for v in partitions[j]:
                    if graph.has_edge(u, v):
                        weight += graph[u][v].get('weight', 1.0)
            if weight > 0:
                coarse_graph.add_edge(i, j, weight=weight)
                
    return coarse_graph
