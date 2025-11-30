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
    k: float | np.ndarray
    m: float | np.ndarray
    gamma: float = 0.0
    
    def __post_init__(self):
        """Validate parameters."""
        if self.n < 1:
            raise ValueError("Number of oscillators must be >= 1")
            
        # Validate mass
        if np.any(np.asarray(self.m) <= 0):
            raise ValueError("Mass must be positive")
            
        # Validate stiffness
        if np.any(np.asarray(self.k) < 0):
            raise ValueError("Stiffness must be non-negative")
            
        if self.gamma < 0:
            raise ValueError("Damping must be non-negative")
    
    def stiffness_matrix(self) -> np.ndarray:
        """
        Construct the stiffness matrix for the chain with nearest-neighbor coupling.
        
        Boundary conditions: Fixed ends (both ends are anchored).
        
        If k is scalar: uniform stiffness.
        If k is array: k[i] is stiffness between node i and i+1.
        Wait, we have N nodes. We have N+1 springs if fixed walls?
        Or N-1 springs between nodes?
        
        Let's assume:
        - If k is scalar: all springs are k (including walls).
        - If k is array: it must have length N-1 (internal springs) or N+1 (all springs).
        
        For LayeredMedium1D, I passed k of length N-1.
        So let's assume walls have "default" stiffness or same as nearest?
        
        Let's standardize:
        If k is array of length N-1:
           k[i] connects node i and i+1.
           Wall-0 connection: use k[0] (or some default?)
           (N-1)-Wall connection: use k[N-2]
           
        Actually, in LayeredMedium1D, I calculate k_springs for N-1 intervals.
        I didn't specify wall stiffness.
        
        Let's assume for now that if k is array (N-1), we use k[0] for left wall and k[-1] for right wall.
        """
        N = self.n
        K = np.zeros((N, N))
        
        # Helper to get k for interval i (between node i and i+1)
        # i goes from -1 (left wall) to N-1 (right wall)
        
        k_arr = np.asarray(self.k)
        if k_arr.ndim == 0:
            # Scalar
            k_vals = np.full(N + 1, float(k_arr))
        else:
            # Array
            if len(k_arr) == N - 1:
                # Pad with edge values for walls
                k_vals = np.zeros(N + 1)
                k_vals[1:-1] = k_arr
                k_vals[0] = k_arr[0]
                k_vals[-1] = k_arr[-1]
            elif len(k_arr) == N + 1:
                k_vals = k_arr
            else:
                raise ValueError(f"Stiffness array length must be {N-1} or {N+1}")
        
        # k_vals[i] is spring between node i-1 and i?
        # Let's say indices 0..N are springs.
        # Spring 0: Wall -> Node 0
        # Spring 1: Node 0 -> Node 1
        # ...
        # Spring N: Node N-1 -> Wall
        
        # Re-map k_vals to this logic
        # If we had N-1 internal springs (0..N-2), they correspond to indices 1..N-1 in my new list.
        
        # Let's build K
        for i in range(N):
            # Node i
            
            # Spring to left (index i)
            k_left = k_vals[i]
            K[i, i] += k_left
            if i > 0:
                K[i, i-1] = -k_left
                
            # Spring to right (index i+1)
            k_right = k_vals[i+1]
            K[i, i] += k_right
            if i < N - 1:
                K[i, i+1] = -k_right
                
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
    def eigenmodes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenfrequencies and eigenmodes of the oscillator chain.
        
        Solves the generalized eigenvalue problem: K * v = (omega^2 * M) * v
        where K is stiffness matrix, M is mass matrix (diagonal).
        
        Returns:
            omega: Array of eigenfrequencies (sorted, >= 0).
            modes: Matrix of eigenvectors (n, n), each column is a mode.
        """
        K = self.stiffness_matrix()
        
        # Mass matrix
        m_arr = np.asarray(self.m)
        if m_arr.ndim == 0:
            # Uniform mass
            M = np.diag(np.full(self.n, float(m_arr)))
        else:
            M = np.diag(m_arr)
            
        # Solve generalized eigenvalue problem
        # K * v = lambda * M * v
        # lambda = omega^2
        eigenvalues, eigenvectors = scipy.linalg.eigh(K, b=M)
        
        # Convert eigenvalues to frequencies
        # lambda = omega^2 => omega = sqrt(lambda)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(eigenvalues)
        
        return omega, eigenvectors
