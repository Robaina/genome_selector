import numpy as np
from scipy.linalg import qr
from typing import Tuple


def deterministic_subset_selection(
    A: np.ndarray, k: int, f: float = 1.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the deterministic subset selection algorithm (Algorithm 1) from the paper.

    Args:
    A (np.ndarray): Input matrix of shape (m, n) where m >= n.
    k (int): Number of columns to select, where k <= rank(A).
    f (float): Tolerance parameter, f >= 1. Default is 1.01.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - The permutation matrix Pi as a 1D array of column indices.
        - The QR decomposition of the permuted matrix.

    Raises:
    ValueError: If the input matrix does not satisfy m >= n or k > rank(A).
    """
    m, n = A.shape
    if m < n:
        raise ValueError("Input matrix must have m >= n")

    # Compute initial QR decomposition with column pivoting
    Q, R, P = qr(A, mode="full", pivoting=True)

    if k > np.linalg.matrix_rank(R):
        raise ValueError("k must be <= rank(A)")

    while True:
        # Partition R into [R_k B_k; 0 C_k]
        R_k = R[:k, :k]
        B_k = R[:k, k:]
        C_k = R[k:, k:]

        # Check if any columns need to be swapped
        max_ratio = 0
        max_i, max_j = -1, -1

        for i in range(k):
            for j in range(n - k):
                R_k_inv_B_k_ij = np.linalg.solve(R_k, B_k[:, j])[i]
                C_k_j_norm = np.linalg.norm(C_k[:, j])
                R_k_inv_i_norm = np.linalg.norm(np.linalg.solve(R_k.T, np.eye(k)[:, i]))

                ratio = np.sqrt(R_k_inv_B_k_ij**2 + C_k_j_norm**2) * R_k_inv_i_norm

                if ratio > max_ratio:
                    max_ratio = ratio
                    max_i, max_j = i, j + k

        if max_ratio <= f:
            break

        # Permute columns i and j+k of R
        R[:, [max_i, max_j]] = R[:, [max_j, max_i]]
        P[[max_i, max_j]] = P[[max_j, max_i]]

        # Retriangularize R
        Q_ij, R = qr(R)
        Q = Q @ Q_ij

    return P, (Q, R)


def two_stage_deterministic_subset_selection(
    A: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the Two-Stage Deterministic Subset Selection algorithm (Algorithm 4) from the paper.

    Args:
    A (np.ndarray): Input matrix of shape (m, n).
    k (int): Number of columns to select, where k <= rank(A).

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - The permutation matrix Pi as a 1D array of column indices.
        - The selected columns of A.

    Raises:
    ValueError: If k > rank(A).
    """
    m, n = A.shape
    r = np.linalg.matrix_rank(A)

    if k > r:
        raise ValueError("k must be <= rank(A)")

    # Compute SVD to get right singular vectors
    _, _, Vt = np.linalg.svd(A, full_matrices=False)

    # Get the first k rows of V^T
    V_k_T = Vt[:k]

    # Compute column norms of V_k_T
    col_norms = np.linalg.norm(V_k_T, axis=0)

    # Sort columns by descending norm
    Pi_1 = np.argsort(-col_norms)

    # Select the first 4k columns
    A_4k = A[:, Pi_1[: 4 * k]]

    # Apply deterministic subset selection to A_4k
    Pi_2, _ = deterministic_subset_selection(A_4k, k)

    # Combine permutations
    Pi = np.concatenate([Pi_1[: 4 * k][Pi_2], Pi_1[4 * k :]])

    return Pi, A[:, Pi[:k]]


def weighted_two_stage_subset_selection(
    A: np.ndarray, k: int, weights: np.ndarray, alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements a weighted version of the Two-Stage Deterministic Subset Selection algorithm.

    Args:
    A (np.ndarray): Input matrix of shape (m, n).
    k (int): Number of columns to select, where k <= rank(A).
    weights (np.ndarray): Array of weights for each column, shape (n,).
    alpha (float): Balance factor between representation and weights.
                   0 <= alpha <= 1, where 0 is all weights, 1 is all representation.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - The permutation matrix Pi as a 1D array of column indices.
        - The selected columns of A.

    Raises:
    ValueError: If k > rank(A) or if weights are not the right shape.
    """
    m, n = A.shape
    r = np.linalg.matrix_rank(A)

    if k > r:
        raise ValueError("k must be <= rank(A)")
    if weights.shape != (n,):
        raise ValueError(
            "weights must be a 1D array with length equal to the number of columns in A"
        )

    # Normalize weights to [0, 1] range
    normalized_weights = (weights - np.min(weights)) / (
        np.max(weights) - np.min(weights)
    )

    # Compute SVD to get right singular vectors
    _, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Get the first k rows of V^T
    V_k_T = Vt[:k]

    # Compute column norms of V_k_T
    col_norms = np.linalg.norm(V_k_T, axis=0)

    # Normalize column norms to [0, 1] range
    normalized_norms = (col_norms - np.min(col_norms)) / (
        np.max(col_norms) - np.min(col_norms)
    )

    # Combine norms and weights
    combined_score = alpha * normalized_norms + (1 - alpha) * normalized_weights

    # Sort columns by descending combined score
    Pi_1 = np.argsort(-combined_score)

    # Select the first 4k columns
    A_4k = A[:, Pi_1[: 4 * k]]

    # Apply deterministic subset selection to A_4k
    Pi_2, _ = deterministic_subset_selection(A_4k, k)

    # Combine permutations
    Pi = np.concatenate([Pi_1[: 4 * k][Pi_2], Pi_1[4 * k :]])

    return Pi, A[:, Pi[:k]]


def weighted_deterministic_subset_selection(
    A: np.ndarray, k: int, weights: np.ndarray, f: float = 1.01, alpha: float = 0.5
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Implements a weighted version of the deterministic subset selection algorithm (Algorithm 1) from the paper.

    Args:
    A (np.ndarray): Input matrix of shape (m, n) where m >= n.
    k (int): Number of columns to select, where k <= rank(A).
    weights (np.ndarray): Array of weights for each column, shape (n,).
    f (float): Tolerance parameter, f >= 1. Default is 1.01.
    alpha (float): Balance factor between representation and weights.
                   0 <= alpha <= 1, where 0 is all weights, 1 is all representation.

    Returns:
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: A tuple containing:
        - The permutation matrix Pi as a 1D array of column indices.
        - The QR decomposition of the permuted matrix (Q, R).

    Raises:
    ValueError: If the input matrix does not satisfy m >= n or k > rank(A).
    """
    m, n = A.shape
    if m < n:
        raise ValueError("Input matrix must have m >= n")
    if weights.shape != (n,):
        raise ValueError(
            "weights must be a 1D array with length equal to the number of columns in A"
        )

    # Normalize weights to [0, 1] range
    normalized_weights = (weights - np.min(weights)) / (
        np.max(weights) - np.min(weights)
    )

    # Compute initial QR decomposition with column pivoting
    Q, R, P = qr(A, mode="full", pivoting=True)

    if k > np.linalg.matrix_rank(R):
        raise ValueError("k must be <= rank(A)")

    while True:
        # Partition R into [R_k B_k; 0 C_k]
        R_k = R[:k, :k]
        B_k = R[:k, k:]
        C_k = R[k:, k:]

        # Check if any columns need to be swapped
        max_combined_score = 0
        max_i, max_j = -1, -1

        for i in range(k):
            for j in range(n - k):
                R_k_inv_B_k_ij = np.linalg.solve(R_k, B_k[:, j])[i]
                C_k_j_norm = np.linalg.norm(C_k[:, j])
                R_k_inv_i_norm = np.linalg.norm(np.linalg.solve(R_k.T, np.eye(k)[:, i]))

                # Calculate the original ratio
                ratio = np.sqrt(R_k_inv_B_k_ij**2 + C_k_j_norm**2) * R_k_inv_i_norm

                # Normalize the ratio to [0, 1] range (approximately)
                normalized_ratio = min(ratio / f, 1)

                # Calculate the combined score
                combined_score = alpha * normalized_ratio + (1 - alpha) * (
                    normalized_weights[P[j + k]] - normalized_weights[P[i]]
                )

                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    max_i, max_j = i, j + k

        if max_combined_score <= f * alpha:
            break

        # Permute columns i and j+k of R
        R[:, [max_i, max_j]] = R[:, [max_j, max_i]]
        P[[max_i, max_j]] = P[[max_j, max_i]]

        # Retriangularize R
        Q_ij, R = qr(R)
        Q = Q @ Q_ij

    return P, (Q, R)
