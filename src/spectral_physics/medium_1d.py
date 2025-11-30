import numpy as np
import scipy.linalg
from dataclasses import dataclass


@dataclass
class OscillatorChain1D:
    """
    Linear chain of coupled oscillators (toy model of a medium).
    
    Attributes:
        n: Number of oscillators.
        k: Coupling stiffness between neighbors.
        m: Mass of each oscillator.
        gamma: Damping coefficient (optional, default 0.0).
    """
    n: int
    k: float
    m: float
    gamma: float = 0.0
    
    def __post_init__(self):
        """Validate parameters."""
        if self.n < 1:
            raise ValueError("Number of oscillators must be >= 1")
        if self.m <= 0:
            raise ValueError("Mass must be positive")
        if self.k < 0:
            raise ValueError("Stiffness must be non-negative")
        if self.gamma < 0:
            raise ValueError("Damping must be non-negative")
    
    def stiffness_matrix(self) -> np.ndarray:
        """
        Construct the stiffness matrix for the chain with nearest-neighbor coupling.
        
        Boundary conditions: Fixed ends (both ends are anchored).
        
        The stiffness matrix K is tridiagonal:
        - Diagonal: 2*k (force from both neighbors)
        - Off-diagonal: -k (coupling to neighbor)
        
        Returns:
            Symmetric stiffness matrix of shape (n, n).
        """
        # Diagonal elements: 2*k for each oscillator
        diagonal = np.full(self.n, 2.0 * self.k)
        
        # Off-diagonal elements: -k for coupling
        off_diagonal = np.full(self.n - 1, -self.k)
        
        # Build tridiagonal matrix
        K = (
            np.diag(diagonal) +
            np.diag(off_diagonal, k=1) +
            np.diag(off_diagonal, k=-1)
        )
        
        return K
    
    def eigenmodes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenfrequencies and eigenmodes of the oscillator chain.
        
        Solves the eigenvalue problem: K * v = (m * omega^2) * v
        where K is the stiffness matrix.
        
        Returns:
            omega: Array of eigenfrequencies (sorted, >= 0).
            modes: Matrix of eigenvectors (n, n), each column is a mode.
        
        Note:
            Eigenfrequencies are angular frequencies (rad/s).
            Modes are normalized eigenvectors.
        """
        K = self.stiffness_matrix()
        
        # Solve eigenvalue problem
        # K * v = lambda * v, where lambda = m * omega^2
        eigenvalues, eigenvectors = scipy.linalg.eigh(K)
        
        # Convert eigenvalues to frequencies
        # lambda = m * omega^2 => omega = sqrt(lambda / m)
        # Clip negative values (numerical noise) to zero
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(eigenvalues / self.m)
        
        return omega, eigenvectors
