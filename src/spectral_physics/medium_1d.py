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

    def __init__(self, n_points, k_coupling=1.0, m=1.0):
        """
        Параметры:
            n_points: число осцилляторов
            k_coupling: коэффициент жесткости связи (k)
            m: масса каждой точки
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
            
        self.n_points = n_points
        self.k = float(k_coupling)
        self.m = float(m)

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
