import numpy as np
import pytest
from spectral_physics.ldos import ldos_from_modes
from spectral_physics.medium_2d import OscillatorGrid2D

def test_ldos_empty_window():
    # 3 points, 2 modes
    modes = np.array([[1, 0], [0, 1], [1, 1]])
    omegas = np.array([1.0, 2.0])
    
    # Window that captures nothing
    ldos = ldos_from_modes(modes, omegas, (3.0, 4.0))
    
    assert np.all(ldos == 0)

def test_ldos_full_window():
    # 2 points, 2 modes
    # Mode 0: [1, 0] at w=1
    # Mode 1: [0, 1] at w=2
    modes = np.array([[1.0, 0.0], [0.0, 1.0]])
    omegas = np.array([1.0, 2.0])
    
    # Capture both
    ldos = ldos_from_modes(modes, omegas, (0.0, 3.0))
    
    # Point 0: 1^2 + 0^2 = 1
    # Point 1: 0^2 + 1^2 = 1
    assert np.allclose(ldos, [1.0, 1.0])

def test_grid_ldos_map_shape():
    grid = OscillatorGrid2D(nx=5, ny=5, kx=1.0, ky=1.0, m=1.0)
    
    # Calculate LDOS
    ldos_map = grid.ldos_map(n_modes=10, freq_window=(0.0, 10.0))
    
    assert ldos_map.shape == (5, 5)
    assert np.all(ldos_map >= 0)
