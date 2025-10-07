"""
Initial implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

Design:
    - memory efficient sparse H gives huge speedup
    - memory speedup outweighs fact that this has loops/is not vectorised
    - loops over edges in a dict
"""

import numpy as np
import time
import scipy.sparse as sp

from bp_decoder import AbstractBPDecoder
from code_constructions import make_random_regular_ldpc, make_repetition, get_syndrome


class BPDecoder(AbstractBPDecoder):
    def __init__(self, H: sp.csr_matrix, p: float):
        """
        Args:
            H (csr_matrix): Parity-check matrix (sparse).
            p (float): Bit-flip probability.
        """
        super().__init__(H, p)
        self.H_csc = H.tocsc()
        self.m, self.n = H.shape

        # encode tanner graph adjacency
        # encode all adjacencies as pairs indexed by position in parallel arrays
        self.edge_i = H.nonzero()[0]  # row (parity) index of each nonzero in H
        self.edge_j = H.nonzero()[1]  # col (data) index of each nonzero in H
        self.num_edges = len(self.edge_i)
        # one set of edges only, updated as direction changes as in dense
        self.edges = np.zeros(self.num_edges)

        # lists of neighbours
        self.row2edges = [np.where(self.edge_i == i)[0] for i in range(self.m)]
        self.col2edges = [np.where(self.edge_j == j)[0] for j in range(self.n)]

    def set_initial_llrs(self):
        self.p_l = np.log((1 - self.p) / self.p)
        self.edges.fill(0.0)
        return self.edges

    def data_to_parity(self, edges):
        incoming_sum = np.zeros(self.n)
        for j in range(self.n):
            incoming_sum[j] = np.sum(edges[self.col2edges[j]])

        new_edges = np.zeros_like(edges)
        for j in range(self.n):
            for idx in self.col2edges[j]:
                new_edges[idx] = incoming_sum[j] - edges[idx] + self.p_l
        return new_edges

    def parity_to_data(self, edges, syndrome, alpha):
        new_edges = np.zeros_like(edges)

        for i in range(self.m):
            edge_idxs = self.row2edges[i]
            msgs = edges[edge_idxs]

            parity_hole = (-1) ** syndrome[i]
            signs = np.sign(msgs)
            abs_vals = np.abs(msgs)

            total_sign = np.prod(signs)
            for k, idx in enumerate(edge_idxs):
                sign_prod = total_sign / signs[k]
                min_abs = np.min(np.delete(abs_vals, k))
                new_edges[idx] = parity_hole * alpha * sign_prod * min_abs

        return new_edges

    def run_bp(self, syndrome, max_iter=50):
        edges = self.set_initial_llrs()

        for it in range(1, max_iter + 1):
            alpha = 1 - 2 ** (-it)

            edges = self.data_to_parity(edges)
            edges = self.parity_to_data(edges, syndrome, alpha)

            soft = np.zeros(self.n)
            for j in range(self.n):
                soft[j] = np.sum(edges[self.col2edges[j]]) + self.p_l

            e_BP = (soft < 0).astype(int)

            if np.all(self.H_csc.dot(e_BP) % 2 == syndrome):
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
    print(f"Time taken: {end-start:.4f} s")
