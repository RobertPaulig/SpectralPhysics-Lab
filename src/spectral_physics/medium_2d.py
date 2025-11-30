import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from dataclasses import dataclass

@dataclass
class OscillatorGrid2D:
    """
    2D сетка связанных осцилляторов (игрушечная модель упругой пластины).

    Схема:
    - каждая точка (i, j) связана с соседями по x и y
    - возможны разные жёсткости kx, ky
    - масса в узле одна и та же
    - Граничные условия: закреплённые края (нулевое смещение за пределами сетки).
    """
    nx: int
    ny: int
    kx: float
    ky: float
    m: float
    
    # Optional local parameters (defects)
    mass_map: np.ndarray | None = None  # (ny, nx)
    kx_map: np.ndarray | None = None    # (ny, nx) - stiffness to right neighbor?
    # Let's define: kx_map[i, j] is stiffness between (i,j) and (i,j+1)
    # ky_map[i, j] is stiffness between (i,j) and (i+1,j)
    # Or simpler: kx_map[i, j] is "local stiffness coefficient" for node (i,j) in x-direction
    # Let's stick to the simple model: 
    # kx_map[i, j] overrides self.kx for connections involving node (i,j).
    # But connections are shared. Let's say connection (i,j)-(i,j+1) uses average or min?
    # Simpler: kx_map defines the link to the RIGHT (j+1)
    # ky_map defines the link to the DOWN (i+1)
    # This is unambiguous.
    
    ky_map: np.ndarray | None = None

    def stiffness_matrix(self) -> np.ndarray:
        """
        Собрать матрицу жёсткости размера (nx*ny, nx*ny).
        Используется 5-точечный шаблон (центр, лево, право, верх, низ).
        
        Индексация: k = i * nx + j, где i - строка (0..ny-1), j - столбец (0..nx-1).
        """
        N = self.nx * self.ny
        K = np.zeros((N, N))
        
        # Helper to get stiffness
        def get_kx(i, j):
            if self.kx_map is not None:
                return self.kx_map[i, j]
            return self.kx
            
        def get_ky(i, j):
            if self.ky_map is not None:
                return self.ky_map[i, j]
            return self.ky
        
        for i in range(self.ny):
            for j in range(self.nx):
                k = i * self.nx + j
                
                diag_val = 0.0
                
                # Link to Right (j+1)
                if j < self.nx - 1:
                    kval = get_kx(i, j)
                    k_right = i * self.nx + (j + 1)
                    
                    K[k, k] += kval
                    K[k_right, k_right] += kval
                    K[k, k_right] = -kval
                    K[k_right, k] = -kval
                else:
                    # Boundary condition: fixed wall on the right?
                    # Original code: 
                    # diag_val += self.kx (always added for left/right)
                    # If we follow original logic: "spring exists always"
                    # For right boundary: spring connects to wall.
                    kval = get_kx(i, j)
                    K[k, k] += kval
                    
                # Link to Left (j-1)
                # Handled by (j-1)'s link to Right?
                # No, we iterate all nodes.
                # Let's be careful not to double count if we iterate all.
                # Standard assembly: iterate elements (springs) or nodes?
                # If iterating nodes, we add contribution of all connected springs.
                pass

        # Re-implementing with clear "spring-based" assembly to avoid confusion
        K = np.zeros((N, N))
        
        # 1. Horizontal springs
        for i in range(self.ny):
            for j in range(self.nx):
                # Spring to the right of (i,j)
                # Connects (i,j) and (i,j+1) OR (i,j) and Wall
                
                k_curr = i * self.nx + j
                kval = get_kx(i, j)
                
                # Add to current node diagonal
                K[k_curr, k_curr] += kval
                
                if j < self.nx - 1:
                    k_right = i * self.nx + (j + 1)
                    K[k_right, k_right] += kval
                    K[k_curr, k_right] = -kval
                    K[k_right, k_curr] = -kval
                # If j == nx-1, it connects to wall (displacement 0), so only diagonal term remains.
                
                # Wait, what about the spring to the LEFT of (i,0)?
                # We need a spring there too if boundaries are fixed.
                # My previous logic: "diag_val += self.kx" implied spring to left and right.
                # So we need to iterate j from -1 to nx-1?
                # Let's assume:
                # Node j has spring to j+1 (defined by kx_map[i,j])
                # Node j has spring to j-1 (defined by kx_map[i,j-1]?)
                # Boundary: Wall -> Node 0. Let's say this is defined by kx_map[i, -1] ?? No.
                
                # Let's assume kx_map has shape (ny, nx+1) to cover all intervals?
                # Or just use kx_map for internal links and kx for boundaries?
                # Or simpler: kx_map[i,j] is the spring to the RIGHT of node j.
                # And we assume default kx for the spring to the LEFT of node 0.
                pass
        
        # Let's stick to the node-based logic but be consistent.
        # Node (i,j) is connected to:
        # Left: (i, j-1). Stiffness? Use get_kx(i, j-1) if j>0. If j=0, use get_kx(i, -1)?
        # Right: (i, j+1). Stiffness? Use get_kx(i, j).
        
        # To make it symmetric and simple:
        # kx_map[i, j] is the stiffness of the link between (i,j) and (i,j+1).
        # For boundary springs (Wall-0 and (N-1)-Wall), let's use self.kx (global).
        
        K = np.zeros((N, N))
        
        for i in range(self.ny):
            for j in range(self.nx):
                k_curr = i * self.nx + j
                
                # --- Horizontal ---
                
                # Link Left: (i, j-1) <-> (i, j)
                if j == 0:
                    k_left_val = self.kx # Wall
                else:
                    k_left_val = get_kx(i, j-1)
                    k_left = i * self.nx + (j - 1)
                    K[k_curr, k_left] = -k_left_val
                
                K[k_curr, k_curr] += k_left_val
                
                # Link Right: (i, j) <-> (i, j+1)
                if j == self.nx - 1:
                    k_right_val = self.kx # Wall
                else:
                    k_right_val = get_kx(i, j)
                    k_right = i * self.nx + (j + 1)
                    K[k_curr, k_right] = -k_right_val
                    
                K[k_curr, k_curr] += k_right_val
                
                # --- Vertical ---
                
                # Link Up: (i-1, j) <-> (i, j)
                if i == 0:
                    k_up_val = self.ky # Wall
                else:
                    k_up_val = get_ky(i-1, j)
                    k_up = (i - 1) * self.nx + j
                    K[k_curr, k_up] = -k_up_val
                    
                K[k_curr, k_curr] += k_up_val
                
                # Link Down: (i, j) <-> (i+1, j)
                if i == self.ny - 1:
                    k_down_val = self.ky # Wall
                else:
                    k_down_val = get_ky(i, j)
                    k_down = (i + 1) * self.nx + j
                    K[k_curr, k_down] = -k_down_val
                    
                K[k_curr, k_curr] += k_down_val
                
        return K

    def eigenmodes(self, n_modes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Найти собственные частоты и моды.
        """
        K = self.stiffness_matrix()
        
        # Mass matrix M
        if self.mass_map is not None:
            m_vec = self.mass_map.flatten()
        else:
            m_vec = np.full(self.nx * self.ny, self.m)
            
        # Generalized eigenvalue problem: K v = lambda M v
        # Since M is diagonal, we can transform to standard EVP:
        # M^(-1/2) K M^(-1/2) u = lambda u, where v = M^(-1/2) u
        # Or just use eigh with b argument (if positive definite)
        
        # Let's use eigh(K, b=M) if possible, but M is diagonal array.
        # scipy.linalg.eigh supports b as matrix.
        M = np.diag(m_vec)
        
        if n_modes is not None and n_modes < len(m_vec):
             eigvals, eigvecs = scipy.linalg.eigh(
                K, b=M,
                subset_by_index=(0, n_modes - 1)
            )
        else:
            eigvals, eigvecs = scipy.linalg.eigh(K, b=M)
            
        eigvals = np.maximum(eigvals, 0.0)
        omega = np.sqrt(eigvals) # lambda = omega^2 (mass is already in M)
        
        return omega, eigvecs
        omega = np.sqrt(eigvals / self.m)
        
        return omega, eigvecs

    def ldos_map(
        self,
        n_modes: int,
        freq_window: tuple[float, float],
    ) -> np.ndarray:
        """
        Вернуть LDOS-карту формы (ny, nx) в заданном частотном окне.
        Использует eigenmodes(...) + ldos_from_modes(...).
        """
        from .ldos import ldos_from_modes
        
        omega, modes = self.eigenmodes(n_modes=n_modes)
        ldos_flat = ldos_from_modes(modes, omega, freq_window)
        
        return ldos_flat.reshape((self.ny, self.nx))
