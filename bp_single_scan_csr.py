"""
Implementation of BP algorithm from:
"Decoding Across the Quantum LDPC Code Landscape"
Ref.: Appendix C in https://arxiv.org/abs/2005.07016

Design:
    - single-scan min-sum schedule using CSR (no CSC mapping)
    - sparse representation built manually without SciPy
    - message updates mostly vectorised except "min excluding self" loop
    - supports CuPy arrays for GPU execution (NumPy fallback on CPU)
    - uses CSR structure for both parity→data updates and soft recomputation
    - no Python objects or dicts; array-based message storage
    - partially GPU-ready (pre-optimised for future full GPU kernel)
"""

import time

from numba import njit  # for cpu jit compilation?? could try before gpu

try:
    import cupy as cp

    asnumpy = cp.asnumpy
    print("Using GPU with CuPy")
except ImportError:
    import numpy as cp

    asnumpy = lambda x: x
    print("CuPy not found, falling back to NumPy (CPU)")

from code_constructions import get_syndrome, make_repetition


def dense_to_csr(H):
    """Convert dense matrix to CSR arrays."""
    m, n = H.shape
    nnz_row, nnz_col = cp.nonzero(H)
    col_idx_csr = nnz_col.astype(int)
    row_ptr = cp.zeros(m + 1, dtype=int)
    counts = cp.bincount(nnz_row, minlength=m)
    row_ptr[1:] = cp.cumsum(counts)
    return col_idx_csr, row_ptr


class BPDecoder:
    """Single-scan min-sum BP decoder using CSR only."""

    def __init__(self, H, p):
        self.H_dense = cp.array(H.astype(int))
        self.m, self.n = H.shape
        self.p = float(p)
        self.llr_init = cp.log((1.0 - self.p) / self.p)

        # convert to CSR
        self.col_idx_csr, self.row_ptr = dense_to_csr(H)
        self.num_edges = len(self.col_idx_csr)
        self.edges = cp.zeros(self.num_edges, dtype=cp.float32)

        self.degree_per_check = cp.diff(self.row_ptr)
        # need to store Z_n for single scan
        self.Z_n = cp.full(self.n, self.llr_init, dtype=cp.float32)

    def set_initial_llrs(self):
        self.edges.fill(0.0)
        self.Z_n.fill(self.llr_init)

    def parity_to_data_single_scan(self, syndrome, alpha):
        """Single-scan min-sum BP in CSR."""
        parity_hole = (-1) ** syndrome

        # loop over parity nodes
        for i in range(self.m):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            vars = self.col_idx_csr[start:end]

            # loop over each variable node connected to this parity node
            for k, var_idx in enumerate(vars):
                # Exclude self
                other_idx = cp.delete(cp.arange(len(vars)), k)
                if len(other_idx) == 0:
                    # single-edge check
                    msg = 0.0
                else:
                    # Contributions from neighbors excluding self
                    contribs = (
                        self.Z_n[vars[other_idx]] - self.edges[start:end][other_idx]
                    )
                    signs = cp.sign(contribs)
                    min_abs = cp.min(cp.abs(contribs))
                    prod_sign = cp.prod(signs)

                    msg = parity_hole[i] * alpha * prod_sign * min_abs

                self.edges[start + k] = msg

                self.Z_n[var_idx] = self.llr_init + cp.sum(
                    self.edges[self.col_idx_csr == var_idx]
                )

    def run_bp(self, syndrome, max_iter=50):
        self.set_initial_llrs()
        e_BP = cp.zeros(self.n, dtype=int)
        syndrome = cp.array(syndrome)
        softs = cp.zeros(self.n, dtype=int)

        for it in range(1, max_iter + 1):
            alpha = 1 - 2 ** (-it)
            self.parity_to_data_single_scan(syndrome, alpha)
            softs = self.Z_n
            e_BP = (softs < 0).astype(int)

            if cp.all((self.H_dense @ e_BP) % 2 == syndrome):
                return True, asnumpy(e_BP), it, asnumpy(softs)

        return False, asnumpy(e_BP), max_iter, asnumpy(softs)


if __name__ == "__main__":
    H = make_repetition(3333)
    p = 0.2
    syndrome, received = get_syndrome(H, p)

    bp = BPDecoder(H, p)
    start = time.time()
    success, e_BP, iterations, soft = bp.run_bp(syndrome, 33)
    end = time.time()

    print(f"Success: {success}, Iterations: {iterations}")
    print(f"Time taken: {end - start:.4f} s")
