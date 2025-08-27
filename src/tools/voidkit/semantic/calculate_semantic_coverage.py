"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_semantic_coverage(
    input_embeddings: np.ndarray,
    primitive_set: np.ndarray
) -> float:
    """
    Calculates the semantic coverage of a set of primitives with respect to a
    set of input embeddings.

    C_sem = torch.mean(cosine_similarity(input_embeddings, primitive_set))

    Parameters
    ----------
    input_embeddings : np.ndarray
        A 2D numpy array of input embeddings.
    primitive_set : np.ndarray
        A 2D numpy array of primitive embeddings.

    Returns
    -------
    float
        The calculated semantic coverage.
    """
    # Calculate the cosine similarity between each input and each primitive
    similarity_matrix = cosine_similarity(input_embeddings, primitive_set)
    
    # For each input, find the similarity to the closest primitive
    max_similarities = np.max(similarity_matrix, axis=1)
    
    # The semantic coverage is the mean of these maximum similarities
    return np.mean(max_similarities)
