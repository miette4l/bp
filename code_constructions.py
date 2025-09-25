"""
Code construction utilities for tests and BP decoding.
"""

import numpy as np
import random

# -----------------------------
# Repetition code
# -----------------------------
def make_repetition(n):
    """Return H matrix for n-bit repetition code (1 logical bit)."""
    H = np.zeros((n - 1, n), dtype=int)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


# -----------------------------
# Random regular LDPC
# -----------------------------
def make_random_regular_ldpc(m, n, row_weight, col_weight, max_passes=100):
    """
    Generate a random (m x n) LDPC parity check matrix with fixed row and column weights.
    """
    assert m * row_weight == n * col_weight, "Inconsistent dimensions"

    row_ones = np.repeat(np.arange(m), row_weight)
    col_ones = np.repeat(np.arange(n), col_weight)
    np.random.shuffle(col_ones)
    edges = list(zip(row_ones, col_ones))

    for _ in range(max_passes):
        used = set()
        duplicates = False
        for i, (r1, c1) in enumerate(edges):
            if (r1, c1) not in used:
                used.add((r1, c1))
                continue
            duplicates = True
            candidates = [
                j
                for j in range(len(edges))
                if j != i
                and (r1, edges[j][1]) not in used
                and (edges[j][0], c1) not in used
                and r1 != edges[j][0]
                and c1 != edges[j][1]
            ]
            if candidates:
                j = random.choice(candidates)
                r2, c2 = edges[j]
                edges[i], edges[j] = (r1, c2), (r2, c1)
                used.add(edges[i])
            else:
                used.add(edges[i])
        if not duplicates:
            break

    H = np.zeros((m, n), dtype=int)
    for r, c in edges:
        H[r, c] = 1

    # Sanity check
    assert np.all(np.count_nonzero(H, axis=1) == row_weight)
    assert np.all(np.count_nonzero(H, axis=0) == col_weight)
    return H


# -----------------------------
# Syndrome simulation - binary symmetric channel
# -----------------------------
def get_syndrome(H, p):
    """
    Generate a random error vector and compute its syndrome.

    Args:
        H (np.ndarray): Parity check matrix
        p (float): Physical bit-flip probability

    Returns:
        tuple: (syndrome, received) where received = codeword + error
    """
    n = H.shape[1]
    codeword = np.zeros(n, dtype=int)
    received = codeword.copy()
    flip_mask = np.random.rand(n) < p
    received[flip_mask] ^= 1
    syndrome = H @ received % 2
    return syndrome, received

if __name__ == "__main__":
    rep = make_repetition(3)
    print(rep)
    syndrome, received = get_syndrome(rep, 0.2)
    print(syndrome)