"""
Initial implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

Design:
    - data structures: one dense message array continually updated
    - operations on that array np-vectorised as much as was straightforward
    - couldn't vectorise the min-abs factor of the parity_to_data....
    - memory blowup
"""

import numpy as np
import time

from bp_decoder import AbstractBPDecoder
from code_constructions import make_random_regular_ldpc, make_repetition, get_syndrome


class BPDecoder(AbstractBPDecoder):

    def __init__(self, H: np.array, p: float):
        """
        Args:
            H (np.ndarray): Parity-check matrix.
            p (float): Bit-flip probability.
        """
        super().__init__(H, p)

    def set_initial_llrs(self):
        self.p_l = np.log((1 - self.p) / self.p)
        self.edges = np.zeros_like(self.H, dtype=float)
        return self.edges

    def data_to_parity(self, edges):
        incoming_sum = np.sum(edges * self.H, axis=0)  # sum up each column
        edges = incoming_sum[None, :] - edges  # subtract yourself
        edges += self.p_l
        return edges

    def parity_to_data(self, edges, syndrome, alpha):
        parity_hole = (-1) ** syndrome

        llr_signs = np.sign(edges)
        llr_signs_masked = np.where(self.H, llr_signs, 1)
        incoming_prod = np.prod(
            llr_signs_masked, axis=1
        )  # product of all signs except zero elements in each row
        sign_prod = incoming_prod[:, None] / llr_signs_masked  # quotient out yourself

        abs_edges = np.where(self.H, np.abs(edges), np.inf)
        min_abs_llr = np.full_like(edges, np.inf)
        for j in range(self.H.shape[1]):
            mask = self.H.astype(bool)
            mask[:, j] = False
            min_abs_llr[:, j] = np.min(np.where(mask, abs_edges, np.inf), axis=1)

        edges = parity_hole[:, None] * alpha * sign_prod * min_abs_llr

        return edges

    def run_bp(self, syndrome, max_iter=50):

        edges = self.set_initial_llrs()

        for it in range(1, max_iter + 1):
            alpha = 1 - 2 ** (-it)

            edges = self.data_to_parity(edges)
            edges = self.parity_to_data(edges, syndrome, alpha)

            soft = np.sum(edges * self.H, axis=0) + self.p_l
            e_BP = (soft < 0).astype(int)

            if np.all(self.H @ e_BP % 2 == syndrome):
                return True, e_BP, it, soft

        return False, e_BP, it, soft


if __name__ == "__main__":
    H = make_repetition(3)
    p=0.2
    print(H)
    syndrome, received = get_syndrome(H, p)
    decoder = BPDecoder(H, p)
    start = time.time()
    result, e_BP, it, soft = decoder.run_bp(syndrome,3)
    end = time.time()
    print(result)
    print(soft)
    print(f"Time taken: {end-start} s")