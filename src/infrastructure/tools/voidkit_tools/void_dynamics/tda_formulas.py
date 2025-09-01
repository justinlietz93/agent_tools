"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

# FUM_AdvancedMath.fum.tda_formulas
#
# Blueprint-compliant TDA formulas implementing the EHTP three-stage pipeline
# with performance optimization and pathology scoring integration.

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import logging

# Import existing pathology score function from diagnostics_formulas
from .diagnostics_formulas import calculate_pathology_score

logger = logging.getLogger(__name__)

def ehtp_stage1_cohesion_check(adjacency_matrix: csr_matrix) -> Dict[str, any]:
    """
    EHTP Stage 1: Fast O(N+M) cohesion check for connectome fragmentation.
    
    Blueprint Reference: Rule 4 - EHTP Stage 1: Cohesion Check (CCC)
    Performance: O(N+M) - Linear in nodes and edges
    
    Args:
        adjacency_matrix: Sparse connectivity matrix of the network
        
    Returns:
        Dict containing:
        - n_components: Number of disconnected components
        - component_labels: Array mapping each node to its component
        - fragmented: Boolean indicating if network is fragmented
        - largest_component_size: Size of largest connected component
    """
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_time:
        start_time.record()
    
    n_components, component_labels = connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )
    
    # Calculate component statistics
    component_sizes = np.bincount(component_labels)
    largest_component_size = np.max(component_sizes)
    fragmented = n_components > 1
    
    if start_time:
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        logger.debug(f"EHTP Stage 1 timing: {elapsed_ms:.3f}ms for {adjacency_matrix.shape[0]} nodes")
    
    logger.info(f"EHTP Stage 1: {n_components} components, largest: {largest_component_size} nodes, fragmented: {fragmented}")
    
    return {
        'n_components': n_components,
        'component_labels': component_labels,
        'fragmented': fragmented,
        'largest_component_size': largest_component_size,
        'component_sizes': component_sizes
    }

def ehtp_stage2_locus_search(
    adjacency_matrix: csr_matrix,
    spike_rates: torch.Tensor,
    pathology_threshold: float = 0.7,
    max_locus_size: int = 100
) -> Optional[List[int]]:
    """
    EHTP Stage 2: Hierarchical locus search to identify suspect subgraphs.
    
    Blueprint Reference: Rule 4 - Pathology Score = Avg Firing Rate × (1 - Output Diversity)
    Performance: O(N × k) where k is average node degree
    
    Args:
        adjacency_matrix: Sparse connectivity matrix
        spike_rates: Tensor of firing rates for each neuron
        pathology_threshold: Minimum pathology score to trigger Stage 3
        max_locus_size: Maximum size of locus to analyze (performance guard)
        
    Returns:
        List of node indices forming the suspect locus, or None if no pathology found
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Performance guard: Skip if network too large without fragmentation
    if n_nodes > 2000:
        logger.warning(f"EHTP Stage 2: Network too large ({n_nodes} nodes), skipping locus search")
        return None
    
    # Calculate output diversity for each node using downstream connections
    output_diversity = torch.zeros(n_nodes)
    
    for node_idx in range(n_nodes):
        # Get downstream neighbors
        downstream_neighbors = adjacency_matrix[node_idx].nonzero()[1]
        
        if len(downstream_neighbors) > 1:
            # Calculate Shannon entropy of downstream firing rates
            downstream_rates = spike_rates[downstream_neighbors]
            
            # Normalize to probabilities (avoid division by zero)
            rate_sum = torch.sum(downstream_rates)
            if rate_sum > 1e-8:
                p = downstream_rates / rate_sum
                # Filter out zero probabilities
                p = p[p > 1e-8]
                if len(p) > 1:
                    entropy = -torch.sum(p * torch.log(p))
                    # Normalize entropy by maximum possible entropy
                    max_entropy = torch.log(torch.tensor(len(p), dtype=torch.float32))
                    output_diversity[node_idx] = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Calculate individual pathology scores for node selection
    pathology_scores = spike_rates * (1 - output_diversity)
    
    # Calculate overall pathology score using existing function for logging
    pathology_score_overall = calculate_pathology_score(spike_rates, output_diversity)
    
    # Find nodes with high pathology scores
    high_pathology_nodes = torch.where(pathology_scores > pathology_threshold)[0]
    
    if len(high_pathology_nodes) == 0:
        logger.debug("EHTP Stage 2: No pathological nodes found")
        return None
    
    # Build locus by including high pathology nodes and their immediate neighbors
    locus_nodes = set(high_pathology_nodes.tolist())
    
    # Add immediate neighbors to form cohesive locus
    for node_idx in high_pathology_nodes:
        neighbors = adjacency_matrix[node_idx].nonzero()[1]
        locus_nodes.update(neighbors.tolist())
        
        # Performance guard: Limit locus size
        if len(locus_nodes) > max_locus_size:
            break
    
    locus_list = sorted(list(locus_nodes))[:max_locus_size]
    
    avg_pathology = torch.mean(pathology_scores[high_pathology_nodes]).item()
    logger.info(f"EHTP Stage 2: Found locus with {len(locus_list)} nodes, avg pathology: {avg_pathology:.4f}, overall: {pathology_score_overall:.4f}")
    
    return locus_list

def ehtp_stage3_deep_tda(
    adjacency_matrix: csr_matrix,
    locus_nodes: List[int],
    b1_persistence_threshold: float = 0.1
) -> Dict[str, any]:
    """
    EHTP Stage 3: Deep TDA analysis on small suspect locus only.
    
    Blueprint Reference: Rule 4 - O(n³) TDA analysis where n << N
    Performance: O(n³) where n is locus size (typically < 100 nodes)
    
    Args:
        adjacency_matrix: Full network adjacency matrix
        locus_nodes: List of node indices forming the suspect locus
        b1_persistence_threshold: Minimum B1 persistence to indicate inefficient cycles
        
    Returns:
        Dict containing TDA results and repair recommendations
    """
    if len(locus_nodes) > 200:
        logger.warning(f"EHTP Stage 3: Locus too large ({len(locus_nodes)} nodes), truncating to 200")
        locus_nodes = locus_nodes[:200]
    
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_time:
        start_time.record()
    
    # Extract locus subgraph
    locus_adj = adjacency_matrix[np.ix_(locus_nodes, locus_nodes)]
    
    # Convert sparse to dense for TDA (only small locus, so acceptable)
    locus_dense = locus_adj.toarray().astype(np.float32)
    
    # Apply TDA using existing functions (but only on small locus)
    try:
        from ..tda.compute_persistent_homology import compute_persistent_homology
        from ..tda.calculate_tda_metrics import calculate_tda_metrics
        
        # Compute persistent homology on locus distance matrix
        ph_result = compute_persistent_homology(
            locus_dense, 
            max_dim=1, 
            is_distance_matrix=True
        )
        
        # Calculate TDA metrics
        tda_metrics = calculate_tda_metrics(ph_result['dgms'])
        
        # Determine if topological repair is needed
        total_b1_persistence = tda_metrics.get('total_b1_persistence', 0.0)
        inefficient_cycles = total_b1_persistence > b1_persistence_threshold
        
        repair_needed = inefficient_cycles
        repair_type = "topological_pruning" if inefficient_cycles else None
        
    except Exception as e:
        logger.error(f"EHTP Stage 3 TDA computation failed: {e}")
        # Fallback: recommend repair based on locus size
        repair_needed = len(locus_nodes) > 50
        repair_type = "size_reduction" if repair_needed else None
        total_b1_persistence = 0.0
    
    if start_time:
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        logger.debug(f"EHTP Stage 3 timing: {elapsed_ms:.3f}ms for {len(locus_nodes)} locus nodes")
    
    logger.info(f"EHTP Stage 3: B1 persistence: {total_b1_persistence:.6f}, repair needed: {repair_needed}")
    
    return {
        'locus_size': len(locus_nodes),
        'locus_nodes': locus_nodes,
        'total_b1_persistence': total_b1_persistence,
        'repair_needed': repair_needed,
        'repair_type': repair_type,
        'inefficient_cycles': inefficient_cycles
    }

def optimized_ehtp_pipeline(
    adjacency_matrix: csr_matrix,
    spike_rates: torch.Tensor,
    pathology_threshold: float = 0.7,
    b1_persistence_threshold: float = 0.1,
    max_locus_size: int = 100
) -> Dict[str, any]:
    """
    Complete EHTP pipeline optimized for performance following blueprint specifications.
    
    Blueprint Reference: Rule 4 - Three-stage EHTP diagnostic pipeline
    
    Performance Optimization:
    - Stage 1: O(N+M) only - always runs
    - Stage 2: O(N×k) only if Stage 1 passes - skipped for very large networks
    - Stage 3: O(n³) only on small loci where n << N - performance bounded
    
    Args:
        adjacency_matrix: Sparse connectivity matrix of the network
        spike_rates: Tensor of firing rates for each neuron
        pathology_threshold: Pathology score threshold for Stage 2
        b1_persistence_threshold: B1 persistence threshold for Stage 3
        max_locus_size: Maximum locus size for performance protection
        
    Returns:
        Complete EHTP analysis results with repair recommendations
    """
    total_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if total_start:
        total_start.record()
    
    n_nodes = adjacency_matrix.shape[0]
    logger.info(f"EHTP Pipeline: Starting analysis on {n_nodes} nodes, {adjacency_matrix.nnz} edges")
    
    # Stage 1: Always run cohesion check (fast O(N+M))
    stage1_results = ehtp_stage1_cohesion_check(adjacency_matrix)
    
    # Early termination for highly fragmented networks
    if stage1_results['n_components'] > 10:
        logger.warning(f"EHTP: Network highly fragmented ({stage1_results['n_components']} components), recommending connectivity repair")
        return {
            'stage1': stage1_results,
            'stage2': None,
            'stage3': None,
            'repair_needed': True,
            'repair_type': 'connectivity_repair',
            'early_termination': True
        }
    
    # Stage 2: Run locus search if network not too fragmented
    stage2_results = ehtp_stage2_locus_search(
        adjacency_matrix, 
        spike_rates, 
        pathology_threshold, 
        max_locus_size
    )
    
    stage3_results = None
    
    # Stage 3: Run expensive TDA only if suspect locus found
    if stage2_results is not None:
        stage3_results = ehtp_stage3_deep_tda(
            adjacency_matrix,
            stage2_results,
            b1_persistence_threshold
        )
    
    # Determine overall repair recommendation
    repair_needed = False
    repair_type = None
    
    if stage1_results['fragmented']:
        repair_needed = True
        repair_type = 'connectivity_repair'
    elif stage3_results and stage3_results['repair_needed']:
        repair_needed = True
        repair_type = stage3_results['repair_type']
    
    if total_start:
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()
        total_ms = total_start.elapsed_time(end_time)
        logger.info(f"EHTP Pipeline: Total time {total_ms:.3f}ms, repair needed: {repair_needed}")
    
    return {
        'stage1': stage1_results,
        'stage2': stage2_results,
        'stage3': stage3_results,
        'repair_needed': repair_needed,
        'repair_type': repair_type,
        'early_termination': False
    }