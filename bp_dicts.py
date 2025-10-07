"""
Initial implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

Design:
    - data structures: message dictionaries, explicit graph structure with Node classes
    - nested loops over lists of Node instances
    - no explicit vectorisation
    - heavy on python object overhead, dict access
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from bp_decoder import AbstractBPDecoder
from code_constructions import make_random_regular_ldpc, make_repetition, get_syndrome


# -----------------------------
# Node classes
# -----------------------------


class Node:
    def __init__(self):
        self.neighbours = []

    def __repr__(self):
        return f"{self.__class__.__name__}(id={getattr(self, 'idx', None)})"


class DataNode(Node):
    def __init__(self, idx, llr=0.0):
        super().__init__()
        self.idx = idx
        self.llr = llr
        self.msg_to_parity = {}
        self.received_msgs = {}


class ParityNode(Node):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.received_msgs = {}
        self.msgs_to_data = {}


# -----------------------------
# BP Decoder
# -----------------------------


class BPDecoder(AbstractBPDecoder):
    """Belief Propagation decoder for binary linear codes."""

    def __init__(self, H, p):
        """
        Args:
            H (np.ndarray): Parity-check matrix.
            p (float): Bit-flip probability.
        """
        super().__init__(H, p)
        self.data_nodes = [DataNode(j) for j in range(self.n)]
        self.parity_nodes = [ParityNode(i) for i in range(self.m)]

        # connect nodes
        for i in range(self.m):
            for j in range(self.n):
                if H[i, j]:
                    self.data_nodes[j].neighbours.append(self.parity_nodes[i])
                    self.parity_nodes[i].neighbours.append(self.data_nodes[j])

    def set_initial_llrs(self):
        """Initialize log-likelihood ratios for data nodes and messages to parity nodes."""
        for v in self.data_nodes:
            v.llr = np.log((1 - self.p) / self.p)
            for u in v.neighbours:
                v.received_msgs[u] = 0

    def parity_to_data(self, syndrome, alpha):
        """Compute messages from parity nodes to data nodes using min-sum BP."""
        sign = lambda x: 1 if x >= 0 else -1
        for i, u in enumerate(self.parity_nodes):
            parity_hole = (-1) ** syndrome[i]
            for v in u.neighbours:
                other_msgs = [u.received_msgs[vj] for vj in u.neighbours if vj != v]
                if not other_msgs:
                    continue
                product_sign = np.prod([sign(m) for m in other_msgs])
                min_abs = min(abs(m) for m in other_msgs)
                msg = parity_hole * alpha * product_sign * min_abs
                u.msgs_to_data[v] = msg
                v.received_msgs[u] = msg

    def data_to_parity(self):
        """Compute messages from data nodes to parity nodes."""
        for v in self.data_nodes:
            for u in v.neighbours:
                incoming = sum(v.received_msgs[up] for up in v.neighbours if up != u)
                msg = v.llr + incoming
                v.msg_to_parity[u] = msg
                u.received_msgs[v] = msg

    def run_bp(self, syndrome, max_iter=50):
        """
        Run BP decoder.

        Args:
            syndrome (np.ndarray): Syndrome vector of length m.
            max_iter (int): Maximum iterations.

        Returns:
            tuple: (success: bool, estimated_error: np.ndarray, iterations: int)
        """
        self.set_initial_llrs()

        # --- initial priming step (iteration 0) ---
        # self.data_to_parity()

        for it in range(1, max_iter + 1):
            alpha = 1 - 2 ** (-it)

            # swapped order (hence initial priming step)
            self.data_to_parity()
            self.parity_to_data(syndrome, alpha)

            # compute soft decision vector
            soft = np.array(
                [
                    sum(v.received_msgs.values()) + v.llr 
                    for v in self.data_nodes
                ]
            )
            # compute hard decision vector
            e_BP = (soft < 0).astype(int)

            if np.all(self.H @ e_BP % 2 == syndrome):
                return True, e_BP, it, soft
            
        return False, e_BP, it, soft

    def print_tanner_graph(self):
        """Visualize the Tanner graph of the code."""
        B = nx.Graph()
        data_labels = [f"d{i}" for i in range(self.n)]
        parity_labels = [f"p{i}" for i in range(self.m)]
        B.add_nodes_from(data_labels, bipartite=0)
        B.add_nodes_from(parity_labels, bipartite=1)

        for i, v in enumerate(self.data_nodes):
            for u in v.neighbours:
                j = self.parity_nodes.index(u)
                B.add_edge(data_labels[i], parity_labels[j])

        pos = nx.bipartite_layout(B, data_labels)
        nx.draw_networkx_nodes(
            B, pos, nodelist=data_labels, node_color="lightblue", node_shape="o"
        )
        nx.draw_networkx_nodes(
            B, pos, nodelist=parity_labels, node_color="lightgreen", node_shape="s"
        )
        nx.draw_networkx_labels(B, pos)
        nx.draw_networkx_edges(B, pos)
        plt.show()


# -----------------------------
# Demo
# -----------------------------

if __name__ == "__main__":

    # m, n = 5, 10
    # row_weight, col_weight = 4, 2
    # H = make_random_regular_ldpc(m, n, row_weight, col_weight)
    # print("H:\n", H)

    # p = 0.1
    # syndrome, received = get_syndrome(H, p)
    # print("Syndrome:\n", syndrome)

    # decoder = BPDecoder(H, p)
    # success, e_BP, iterations, soft = decoder.run_bp(syndrome, max_iter=50)
    # print("Success:", success)
    # if success:
    #     print("Estimated error:", e_BP)
    #     print("Iterations:", iterations)

    H = make_repetition(33333)
    p=0.2
    print(H)
    syndrome, received = get_syndrome(H, p)
    decoder = BPDecoder(H, p)
    start = time.time()
    result, e_BP, it, soft = decoder.run_bp(syndrome,33)
    end = time.time()
    print(result)
    print(soft)
    print(f"Time taken: {end-start} s")

    # Computes with n=33333 and max_it=33 in 137s