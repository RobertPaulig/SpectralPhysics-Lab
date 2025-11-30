import numpy as np
import scipy.linalg

class OscillatorChain1D:
    """
    Линейная цепочка связанных осцилляторов с ближайшими соседями.

    Используется как игрушечная модель "Среды":
    - каждая точка = масса на пружине,
    - пружины соединяют соседей.
    - Граничные условия: фиксированные концы (fixed ends).
    """

    def __init__(self, n_points, k_coupling=1.0, m=1.0, damping=0.0):
        """
        Параметры:
            n_points: число осцилляторов
            k_coupling: коэффициент жесткости связи (k)
            m: масса каждой точки
            damping: коэффициент затухания
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
            
        self.n_points = n_points
        self.k = float(k_coupling)
        self.m = float(m)
        self.damping = float(damping)
        
        # State arrays: positions and velocities
        self.x = np.zeros(n_points, dtype=float)
        self.v = np.zeros(n_points, dtype=float)

    def compute_modes(self):
        """
        Находит собственные частоты и собственные векторы (моды).
        
        Возвращает:
            freqs: массив собственных частот (sqrt(eigenvalue))
            modes: матрица (n_points, n_modes), где столбцы - собственные векторы
        """
        # Construct stiffness matrix K for fixed ends
        # Diag = 2k, Off-diag = -k
        # M = m * I
        
        # We solve K v = lambda M v  => K v = (m * w^2) v
        # So eigenvalues of K are (m * w^2).
        # w = sqrt(eigenvalue_of_K / m)
        
        # Build K matrix
        # Main diagonal
        diag_main = np.full(self.n_points, 2.0 * self.k)
        # Off diagonal
        diag_off = np.full(self.n_points - 1, -1.0 * self.k)
        
        K = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)
        
        # Find eigenvalues and eigenvectors
        # eigh is for symmetric/hermitian matrices
        evals, evecs = scipy.linalg.eigh(K)
        
        # evals are m * w^2
        # w = sqrt(evals / m)
        # Note: numerical noise might make small evals slightly negative, clip to 0
        evals = np.maximum(evals, 0.0)
        freqs = np.sqrt(evals / self.m)
        
        return freqs, evecs

    def step(self, dt: float) -> None:
        """
        Advance the system by one time step dt using a simple
        explicit integrator (Velocity-Verlet).
        """
        # Compute forces at current positions
        # F[i] = k * (x[i+1] - x[i]) - k * (x[i] - x[i-1]) - damping * v[i]
        # With fixed boundary: x[-1]=0, x[n]=0
        
        forces = np.zeros(self.n_points)
        
        for i in range(self.n_points):
            f = 0.0
            # Force from left neighbor (or boundary)
            if i == 0:
                f -= self.k * self.x[i]  # Left boundary is fixed at 0
            else:
                f += self.k * (self.x[i-1] - self.x[i])
            
            # Force from right neighbor (or boundary)
            if i == self.n_points - 1:
                f -= self.k * self.x[i]  # Right boundary is fixed at 0
            else:
                f += self.k * (self.x[i+1] - self.x[i])
            
            # Damping force
            f -= self.damping * self.v[i]
            
            forces[i] = f
        
        # Velocity-Verlet integration
        # v(t + dt/2) = v(t) + (dt/2) * a(t)
        # x(t + dt) = x(t) + dt * v(t + dt/2)
        # a(t + dt) = F(x(t+dt)) / m
        # v(t + dt) = v(t + dt/2) + (dt/2) * a(t+dt)
        
        # Simpler Euler method for stability with damping
        acc = forces / self.m
        self.v += dt * acc
        self.x += dt * self.v

    def energy(self) -> float:
        """
        Return total energy (kinetic + potential) of the chain.
        """
        # Kinetic energy: sum(0.5 * m * v^2)
        ke = 0.5 * self.m * np.sum(self.v ** 2)
        
        # Potential energy: sum over springs
        # Each spring between i and i+1 has energy 0.5 * k * (x[i+1] - x[i])^2
        # Also springs to fixed boundaries
        pe = 0.0
        
        # Left boundary spring (to x=0)
        if self.n_points > 0:
            pe += 0.5 * self.k * self.x[0] ** 2
        
        # Internal springs
        for i in range(self.n_points - 1):
            pe += 0.5 * self.k * (self.x[i+1] - self.x[i]) ** 2
        
        # Right boundary spring (to x=0)
        if self.n_points > 0:
            pe += 0.5 * self.k * self.x[-1] ** 2
        
        return float(ke + pe)

