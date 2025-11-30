import numpy as np
import pytest
from spectral_physics.medium_2d import OscillatorGrid2D

def test_grid_2x2_stiffness():
    # 2x2 grid
    # Indices:
    # 0 1
    # 2 3
    #
    # Neighbors:
    # 0: 1(x), 2(y) + walls
    # 1: 0(x), 3(y) + walls
    # ...
    
    grid = OscillatorGrid2D(nx=2, ny=2, kx=1.0, ky=2.0, m=1.0)
    K = grid.stiffness_matrix()
    
    assert K.shape == (4, 4)
    
    # Check symmetry
    assert np.allclose(K, K.T)
    
    # Check diagonal elements
    # Each node has 2 neighbors in x (one might be wall) and 2 in y (one might be wall)
    # With fixed boundaries, every node is connected to 2 neighbors in X (k=1) and 2 in Y (k=2)
    # Total stiffness = 2*kx + 2*ky = 2*1 + 2*2 = 6
    expected_diag = 6.0
    assert np.allclose(np.diag(K), expected_diag)
    
    # Check off-diagonals
    # 0-1 (horizontal): -kx = -1
    assert K[0, 1] == -1.0
    # 0-2 (vertical): -ky = -2
    assert K[0, 2] == -2.0
    # 0-3 (diagonal): 0
    assert K[0, 3] == 0.0

def test_eigenmodes_positive():
    grid = OscillatorGrid2D(nx=3, ny=3, kx=1.0, ky=1.0, m=1.0)
    omega, modes = grid.eigenmodes()
    
    assert len(omega) == 9
    assert modes.shape == (9, 9)
    
    # Frequencies should be positive (stable system)
    assert np.all(omega >= 0)
    
    # First mode should be > 0 (no zero modes for fixed boundaries)
    assert omega[0] > 0

def test_eigenmodes_subset():
    grid = OscillatorGrid2D(nx=5, ny=5, kx=1.0, ky=1.0, m=1.0)
    n_modes = 5
    omega, modes = grid.eigenmodes(n_modes=n_modes)
    
    assert len(omega) == n_modes
    assert modes.shape == (25, n_modes)
    
    # Check sorted
    assert np.all(np.diff(omega) >= 0)
