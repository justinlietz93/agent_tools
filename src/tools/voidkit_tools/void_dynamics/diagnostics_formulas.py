"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

# FUM_AdvancedMath.fum.diagnostics_formulas
#
# Provides the pure, canonical implementations of the mathematical formulas
# for the Introspection Probe's pathology detection and the ADC's adaptive scheduling.

import torch

def calculate_pathology_score(spike_rates: torch.Tensor, output_diversity: torch.Tensor) -> float:
    """
    Calculates the pathology score for a locus (subgraph) to identify
    inefficient, high-activity, low-output regions.

    Ref: Blueprint Rule 4.1
    Time Complexity: O(k) where k is the number of nodes in the locus.
    """
    return torch.mean(spike_rates * (1 - output_diversity)).item()

def calculate_graph_entropy(degree_distribution: torch.Tensor) -> float:
    """
    Calculates the entropy of the graph based on its degree distribution.
    Used by Introspection Probe for global health monitoring and ADC for scheduling.

    Ref: Blueprint Rule 4.1
    Time Complexity: O(N) where N is the number of nodes in the graph.
    """
    # Normalize the distribution to get probabilities
    p = degree_distribution / torch.sum(degree_distribution)
    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]
    return -torch.sum(p * torch.log(p)).item()

def calculate_cartography_time(graph_entropy: float, alpha: float, base_interval: int = 100000) -> int:
    """
    Calculates the timestep for the next scheduled ADC cartography event,
    based on the current graph entropy.

    Ref: Blueprint Rule 7
    Time Complexity: O(1)
    """
    t_territory = base_interval * torch.exp(torch.tensor(-alpha * graph_entropy))
    return int(t_territory)