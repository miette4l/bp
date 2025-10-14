"""
Implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

Design:
    - message dictionaries giving graph structure
    - loops over precomputed neighbour lists which are numpy arrays
    - some vectorisation, not much
    - SciPy sparse H for H.dot(e_BP) but not used to iterate for message updates
    - not GPU-ready: all NumPy/SciPy
    - dual horizontal and vertical scanning
"""

import time

import numpy as np
import scipy.sparse as sp

from code_constructions import get_syndrome, make_random_regular_ldpc, make_repetition


class BPDecoder:
    def __init__(self, H: sp.csr_matrix, p: float):
        if p == 0:
            raise ValueError("p cannot be zero")

        self.H = H.astype(int)
        self.m, self.n = H.shape  # type: ignore
        self.p = p
        self.p_l = np.log((1 - self.p) / self.p)
        self.H_csr = sp.csr_matrix(H)

        self.edge_i, self.edge_j = H.nonzero()
        self.num_edges = len(self.edge_i)
        self.edges = np.zeros(self.num_edges)

        self.row2edges = [[] for _ in range(self.m)]
        self.col2edges = [[] for _ in range(self.n)]
        for idx, (r, c) in enumerate(zip(self.edge_i, self.edge_j)):
            self.row2edges[r].append(idx)
            self.col2edges[c].append(idx)

        # lists to NumPy arrays for speed
        self.row2edges = [np.array(lst, dtype=int) for lst in self.row2edges]
        self.col2edges = [np.array(lst, dtype=int) for lst in self.col2edges]

    def set_initial_llrs(self):
        return self.edges

    def data_to_parity(self, edges):
        incoming_sum = np.zeros(self.n)
        col2edges = self.col2edges  # local var for speed
        for j in range(self.n):
            incoming_sum[j] = edges[col2edges[j]].sum()

        new_edges = np.zeros_like(edges)
        for j in range(self.n):
            for idx in col2edges[j]:
                new_edges[idx] = incoming_sum[j] - edges[idx] + self.p_l
        return new_edges

    def parity_to_data(self, edges, syndrome, alpha):
        new_edges = np.zeros_like(edges)
        row2edges = self.row2edges  # local var for speed

        for i in range(self.m):
            edge_idxs = row2edges[i]
            msgs = edges[edge_idxs]

            parity_hole = (-1) ** syndrome[i]
            signs = np.sign(msgs)
            abs_vals = np.abs(msgs)

            total_sign = 1 - 2 * (np.count_nonzero(signs < 0) % 2)
            min_abs_vals = np.minimum.accumulate(abs_vals)
            reversed_min = np.minimum.accumulate(abs_vals[::-1])[::-1]

            for k, idx in enumerate(edge_idxs):
                if len(abs_vals) == 1:
                    min_abs = 0
                elif k == 0:
                    min_abs = reversed_min[1]
                elif k == len(abs_vals) - 1:
                    min_abs = min_abs_vals[-2]
                else:
                    min_abs = min(min_abs_vals[k - 1], reversed_min[k + 1])

                sign_prod = total_sign / signs[k]
                new_edges[idx] = parity_hole * alpha * sign_prod * min_abs

        return new_edges

    def run_bp(self, syndrome, max_iter=50):
        edges = self.set_initial_llrs()
        e_BP = np.zeros(self.n, dtype=int)
        soft = np.zeros(self.n)
        it = 0

        col2edges = self.col2edges
        H_csr = self.H_csr

        for it in range(1, max_iter + 1):
            alpha = 1 - 2 ** (-it)

            edges = self.data_to_parity(edges)
            edges = self.parity_to_data(edges, syndrome, alpha)

            for j in range(self.n):
                soft[j] = edges[col2edges[j]].sum() + self.p_l

            e_BP = (soft < 0).astype(int)

            if np.all(H_csr.dot(e_BP) % 2 == syndrome):  # type: ignore
                return True, e_BP, it, soft

        return False, e_BP, it, soft


if __name__ == "__main__":
    H = sp.csr_matrix(make_repetition(3333))
    p = 0.2
    syndrome, received = get_syndrome(H, p)
    decoder = BPDecoder(H, p)

    start = time.time()
    result, e_BP, it, soft = decoder.run_bp(syndrome, 33)
    end = time.time()

    print(result)
    print(f"Time taken: {end - start:.4f} s")
