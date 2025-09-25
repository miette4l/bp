"""
Initial implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

with design inspiration from https://pennylane.ai/qml/demos/tutorial_bp_catalyst

Design:
    - data structures: np arrays only
    - nested loops over array indices
    - no explicit vectorisation
    - WARNING: huge slowdown presumably from jax overhead!!
    - functions aren't actually jitted so not sure what they were doing
    - do not use, horrible on CPU
"""

import jax.numpy as jnp
import jax
import time

from code_constructions import make_random_regular_ldpc, make_repetition, get_syndrome
from bp_decoder import AbstractBPDecoder

class BPDecoder(AbstractBPDecoder):

    def __init__(self, H, p):
        """
        Args:
            H (np.ndarray): Parity-check matrix.
            p (float): Bit-flip probability.
        """
        super().__init__(H, p)
        self.H = jnp.asarray(H, dtype=jnp.int32)
        self.p_l = jnp.log((1 - p) / p)

        self.d_nei, self.p_nei = self._build_graph(H)
        
    def _build_graph(self, H):

        # data structure: arrays (list of lists, so neighbours for each node) 
        # later converted into nested tuples
        data, parity = [[] for _ in range(self.n)], [[] for _ in range(self.m)]

        # build graph i.e. populate nested lists with neighbour indices
        for i in range(self.m):
            for j in range(self.n):
                if H[i, j]:
                    data[j].append(i)
                    parity[i].append(j)

        # return as immutable, hashable objects (tuples)
        # each tuple at top level is a data or parity node
        # each tuple at lower level is a collection of its neighbours of diff type
        return tuple(map(tuple, data)), tuple(map(tuple, parity))
    
    def _p2d_update(self, m_p2d_prev, syndrome, d_nei, p_nei, p_l):
        m_p2d_next = jnp.zeros_like(m_p2d_prev)

        alpha = 1.0

        # Loop over parity check nodes (outer) then their data nodes (inner):
        
        for u in range(self.m):
            data_neighbours = p_nei[u]
            if len(data_neighbours) < 2:
                continue  # degree-1 checks have no new info

            for v in data_neighbours:
                incoming_msgs = []
                for v_prime in data_neighbours:
                    if v_prime == v:
                        continue
                    incoming = p_l
                    for u_prime in d_nei[v_prime]:
                        if u_prime != u:
                            incoming += m_p2d_prev[u_prime, v_prime]
                    incoming_msgs.append(incoming)

                # Min-sum magnitude + sign product
                min_abs = jnp.min(jnp.abs(jnp.array(incoming_msgs)))
                sign_prod = jnp.prod(jnp.sign(jnp.array(incoming_msgs)))

                msg = ((-1) ** syndrome[u]) * alpha * sign_prod * min_abs
                m_p2d_next = m_p2d_next.at[u, v].set(msg)
        
        return m_p2d_next
    
    def _posterior_llrs(self, m_p2d_final):
        llr = jnp.full(self.n, self.p_l)
        for v in range(self.n):
            for u in self.d_nei[v]:
                llr = llr.at[v].add(m_p2d_final[u, v])
        return llr

    def run_bp(self, syndrome, max_iter: int = 50):
        syndrome = jnp.asarray(syndrome, dtype=jnp.int32)

        m_p2d = jnp.zeros((self.m, self.n), dtype=jnp.float32)

        it = 0

        def cond_fun(state):
            it, m_p2d = state
            soft = self._posterior_llrs(m_p2d)
            hard = (soft < 0).astype(jnp.int32)
            syndrome_met = jnp.all((self.H @ hard) % 2 == syndrome)
            return jnp.logical_and(it < max_iter, jnp.logical_not(syndrome_met))

        def body_fun(state):
            it, m_p2d = state
            m_p2d = self._p2d_update(m_p2d, syndrome, self.d_nei, self.p_nei, self.p_l)
            return it + 1, m_p2d

        it, m_p2d = jax.lax.while_loop(cond_fun, body_fun, (it, m_p2d))

        print(f"BP decoder stopped at iteration {it} / {max_iter}")

        soft = self._posterior_llrs(m_p2d)
        hard = (soft < 0).astype(jnp.int32)

        success = jnp.all((self.H @ hard) % 2 == syndrome)
        return success, hard, it, soft

    def print_tanner_graph(self):
        """Visualize the Tanner graph of the code."""
        pass


# -----------------------------
# Demo
# -----------------------------

if __name__ == "__main__":
    H = make_repetition(3)
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

    # n=333, maxit=33, time=20s
    # n=3333, maxit=33, time=412s