"""
Usage:

Run in CLI with choice of BP module (no file extension needed):

    pytest test_bp.py -v -s --bp_module=<bp_module_name>

where -vs -> verbosity + print statements in stdout

All bp_module files must provide the same interface:
- BPDecoder(H, p)
- success, inferred_error, iterations_to_convergence, soft = decoder.run_bp(syndrome, max_iter=n)
"""

import numpy as np
import pytest
from scipy.stats import binom
import scipy.sparse as sp

from code_constructions import make_repetition, make_random_regular_ldpc, get_syndrome



# -----------------------------
# Helper functions
# -----------------------------


def repetition_bp_stats(bp_module, num_trials=50, distances=[3, 5, 7], p=0.2):
    """Compute convergence %, logical failure %, and avg iterations for repetition codes."""
    stats = {}
    for n in distances:
        convergence_successes = 0
        logical_failures = 0
        iterations_to_converge = []

        for _ in range(num_trials):
            H = make_repetition(n)
            if bp_module.__name__ == "bp_sparse":
                H = sp.csr_matrix(H)
            syndrome, received = get_syndrome(H, p)
            decoder = bp_module.BPDecoder(H, p)
            success, inferred_error, it, soft = decoder.run_bp(syndrome, max_iter=n)

            if success:
                convergence_successes += 1

            iterations_to_converge.append(it)

            combined_error = (received + inferred_error) % 2
            logical_failures += int(np.sum(combined_error) > n // 2)

        convergence_rate = 100 * convergence_successes / num_trials
        logical_failure_rate = 100 * logical_failures / num_trials
        avg_iterations = np.mean(iterations_to_converge)
        stats[n] = (convergence_rate, logical_failure_rate, avg_iterations)

    return stats


def expected_logical_failure(n, p):
    """Exact expected logical failure probability (%) for n-bit repetition code."""
    t = n // 2
    return sum(binom.pmf(k, n, p) for k in range(t + 1, n + 1)) * 100


def ldpc_convergence_likelihood(
    bp_module, num_trials=100, m=5, n=10, row_weight=4, col_weight=2, p=0.1
):
    """Compute BP convergence likelihood (%) on random LDPC codes with given parameters."""
    successes = 0
    for _ in range(num_trials):
        H = make_random_regular_ldpc(m, n, row_weight, col_weight)
        if bp_module.__name__ == "bp_sparse":
            H = sp.csr_matrix(H)
        syndrome, _ = get_syndrome(H, p)
        decoder = bp_module.BPDecoder(H, p)
        success, _, _, _ = decoder.run_bp(syndrome, max_iter=n)
        if success:
            successes += 1
    return 100 * successes / num_trials


# -----------------------------
# Tests
# -----------------------------


@pytest.fixture
def small_repetition(bp_module):
    """3-bit repetition code fixture."""
    H = make_repetition(3)
    if bp_module.__name__ == "bp_sparse":
        H = sp.csr_matrix(H)
    syndrome = np.array([1, 0])
    return H, syndrome

def test_repetition_bp_convergence_and_logical_failure(bp_module):
    """Check that repetition code BP decoders converge and logical failure matches expectation."""
    distances = [3, 5, 11, 15, 19]
    num_trials = 50
    p = 0.2

    stats = repetition_bp_stats(
        bp_module, num_trials=num_trials, distances=distances, p=p
    )

    for n, (convergence, logical_failure, avg_iter) in stats.items():
        # Convergence check
        assert convergence >= 100, f"Does not converge for distance {n}: {convergence}%"

        # Logical failure check
        expected_failure = expected_logical_failure(n, p)
        assert abs(logical_failure - expected_failure) <= 10.0, (
            f"Logical failure {logical_failure:.1f}% deviates too much from expected {expected_failure:.1f}% for distance {n}"
        )

        # Average iterations check
        assert avg_iter <= n, (
            f"Average iterations too high for distance {n}: {avg_iter:.1f} (max allowed {n})"
        )

    print("")
    print("Repetition Code BP decoder stats:")
    print("distance: (convergence %, logical_failure %, avg_iterations")
    for dist, values in stats.items():
        print(f"{dist}-bit: {values}")


def test_ldpc_bp_convergence(bp_module):
    """Check BP convergence likelihood on small random LDPC codes is 65-75%."""
    likelihood = ldpc_convergence_likelihood(bp_module, num_trials=100)
    assert 65 <= likelihood <= 75, (
        f"Convergence likelihood {likelihood:.1f}% outside expected range (65-75%)"
    )
    print(f"Random LDPC BP convergence likelihood: {likelihood:.1f}%")