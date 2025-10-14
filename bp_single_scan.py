"""
Implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

Basically bp_dicts.py adapted to the single scan approach: https://arxiv.org/pdf/cs/0609090

Design:
    - message dictionaries giving graph structure, explicit with Python classes
    - loops over neighbour lists
    - no explicit vectorisation
    - heavy on python object overhead, dict access
    - dense matrix representation for initial storage and H.dot(e_BP)
    - not GPU-ready: all pure Python
    - single scan
"""

import time

import numpy as np

from code_constructions import get_syndrome, make_random_regular_ldpc, make_repetition


class Node:
    def __init__(self):
        self.neighbours = []

    def __repr__(self):
        return f"{self.__class__.__name__}(id={getattr(self, 'idx', None)})"


class DataNode(Node):
    def __init__(self, idx, llr=0.0):
        super().__init__()
        self.idx = idx
        self.llr = llr  # Zn
        self.msg_to_parity = {}  # Zmn (messages sent to parity nodes)


class ParityNode(Node):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx


class BPDecoder:
    """Single-scan min-sum BP decoder for binary linear codes."""

    def __init__(self, H, p):
        self.H = H.astype(int)
        self.m, self.n = H.shape
        self.p = p
        self.data_nodes = [DataNode(j) for j in range(self.n)]
        self.parity_nodes = [ParityNode(i) for i in range(self.m)]

        # connect nodes
        for i in range(self.m):
            for j in range(self.n):
                if H[i, j]:
                    self.data_nodes[j].neighbours.append(self.parity_nodes[i])
                    self.parity_nodes[i].neighbours.append(self.data_nodes[j])

    def set_initial_llrs(self):
        """Initialize log-likelihood ratios and messages."""
        for v in self.data_nodes:
            v.llr = np.log((1 - self.p) / self.p)
            v.msg_to_parity = {u: 0.0 for u in v.neighbours}

    def parity_to_data_single_scan(self, syndrome, alpha):
        """Single-scan min-sum update with message subtraction trick."""
        sign = lambda x: 1 if x >= 0 else -1

        for i, u in enumerate(self.parity_nodes):
            parity_hole = (-1) ** syndrome[i]

            for v in u.neighbours:
                # subtract old message contribution from neighbor LLRs
                other_msgs = [
                    vj.llr - vj.msg_to_parity.get(u, 0)
                    for vj in u.neighbours
                    if vj != v
                ]
                if not other_msgs:
                    continue

                product_sign = np.prod([sign(m) for m in other_msgs])
                min_abs = min(abs(m) for m in other_msgs)

                # new message from parity to variable
                msg = parity_hole * alpha * product_sign * min_abs

                # update variable node: remove old msg, add new
                v.llr = (v.llr - v.msg_to_parity.get(u, 0)) + msg
                v.msg_to_parity[u] = msg  # store for next iteration

    def run_bp(self, syndrome, max_iter=50):
        """Run single-scan min-sum BP decoder."""
        e_BP = np.zeros(self.n, dtype=int)
        soft = np.zeros(self.n)

        self.set_initial_llrs()

        for it in range(1, max_iter + 1):
            alpha = 1 - 2 ** (-it)

            self.parity_to_data_single_scan(syndrome, alpha)

            soft = np.array([v.llr for v in self.data_nodes])
            e_BP = (soft < 0).astype(int)

            if np.all(self.H @ e_BP % 2 == syndrome):
                return True, e_BP, it, soft

        return False, e_BP, max_iter, soft


if __name__ == "__main__":
    H = make_repetition(33333)
    p = 0.2
    syndrome, received = get_syndrome(H, p)
    decoder = BPDecoder(H, p)

    start = time.time()
    result, e_BP, it, soft = decoder.run_bp(syndrome, 33)
    end = time.time()

    print(result)
    print(f"Time taken: {end - start:.4f} s")
