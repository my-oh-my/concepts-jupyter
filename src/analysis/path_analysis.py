"""
Core mathematical functions for Medoid Path Analysis and Partial Path Matching.
"""

from typing import Any, Dict, List

import numpy as np


def calculate_medoid_channel(input_matrix: np.ndarray, probability: float = 0.75) -> Dict[str, Any]:
    """
    Finds the Medoid path and calculating the probability channel.

    Args:
        input_matrix (np.ndarray): Shape (n_series, time_steps)
        probability (float): The coverage probability for the bands (e.g., 0.75 for 75%).

    Returns:
        dict: Contains medoid path, lower bound, upper bound, and the index of the medoid.
    """
    # Step 1: Find the Medoid (The Most Likely single path)
    # input_matrix shape: (n_series, time_steps)
    # We use broadcasting to compute differences: (N, 1, T) - (1, N, T) -> (N, N, T)
    diffs = input_matrix[:, np.newaxis, :] - input_matrix[np.newaxis, :, :]

    # Euclidean distance between each pair of paths
    dist_matrix = np.sqrt(np.sum(np.square(diffs), axis=2))

    # The Medoid minimizes the sum of distances to all other points
    medoid_idx = np.argmin(np.sum(dist_matrix, axis=1))
    most_likely_path = input_matrix[medoid_idx]

    # Step 2: Calculate the Statistical Channel (Probability Range)
    # We calculate the distribution at each time step t across all N series
    tail = (1.0 - probability) / 2.0
    lower_bound = np.percentile(input_matrix, tail * 100, axis=0)
    upper_bound = np.percentile(input_matrix, (1.0 - tail) * 100, axis=0)

    return {
        "medoid": most_likely_path,
        "lower": lower_bound,
        "upper": upper_bound,
        "index": int(medoid_idx),
    }


def find_nearest_neighbors(
    partial_path: np.ndarray, history_matrix: np.ndarray, n_neighbors: int = 1
) -> List[Dict[str, Any]]:
    """
    Finds the nearest neighbors in the history matrix for a given partial path.

    Args:
        partial_path (np.ndarray): Shape (k,) - The first k hours of a day.
        history_matrix (np.ndarray): Shape (N, 24) - The historical dataset.
        n_neighbors (int): Number of top matches to return.

    Returns:
        List[dict]: List of matches, each containing matched path, index, distance, and rank.
    """
    k = len(partial_path)
    if k == 0:
        raise ValueError("Partial path cannot be empty.")

    # Slice history to the same length as partial_path
    history_slice = history_matrix[:, :k]

    # Calculate Euclidean distance between partial_path and all historical segments
    # Shape: (N,)
    distances = np.sqrt(np.sum(np.square(history_slice - partial_path), axis=1))

    # Find the indices of the n smallest distances
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:n_neighbors]

    matches = []
    for rank, neighbor_idx in enumerate(top_indices):
        matches.append(
            {
                "matched_path": history_matrix[neighbor_idx],
                "index": int(neighbor_idx),
                "distance": float(distances[neighbor_idx]),
                "rank": rank + 1,
            }
        )

    return matches


def get_lookback_vector(matrix: np.ndarray, pivot_idx: int, days_back: int) -> Any:
    """Concatenates previous `days_back` full days preceding `pivot_idx`."""
    if days_back <= 0:
        return np.array([])
    start = pivot_idx - days_back
    if start < 0:
        return None  # Not enough history

    # Extract segments
    segments = []
    for i_seg in range(days_back):
        segments.append(matrix[start + i_seg])
    return np.concatenate(segments)


def get_lookforward_vector(matrix: np.ndarray, pivot_idx: int, days_forward: int) -> np.ndarray:
    """Concatenates `days_forward` full days following `pivot_idx`."""
    if days_forward <= 0:
        return np.array([])

    segments = []
    # Pivot index i is the 'Day 0'. We want indices (i+1) to (i+days_forward)
    for i_f in range(1, days_forward + 1):
        target_f_idx = pivot_idx + i_f
        if target_f_idx < len(matrix):
            segments.append(matrix[target_f_idx])
        else:
            # Not enough future data available
            break

    if not segments:
        return np.array([])
    return np.concatenate(segments)
