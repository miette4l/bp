from abc import ABC, abstractmethod
import numpy as np

class AbstractBPDecoder(ABC):
    """Abstract base class for Belief Propagation decoders."""

    def __init__(self, H: np.ndarray, p: float):
        """
        Args:
            H (np.ndarray): Parity-check matrix.
            p (float): Bit-flip probability.
        """
        if p == 0:
            raise ValueError("p cannot be zero")
        self.H = H.astype(int)
        self.m, self.n = H.shape
        self.p = p

    @abstractmethod
    def run_bp(self, syndrome: np.ndarray, max_iter: int = 50):
        """
        Run BP decoder.

        Args:
            syndrome (np.ndarray): Syndrome vector of length m.
            max_iter (int): Maximum iterations.

        Returns:
            tuple: (success: bool, estimated_error: np.ndarray, iterations: int, soft: np.ndarray)
        """
        pass