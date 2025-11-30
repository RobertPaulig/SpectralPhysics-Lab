import pytest
import numpy as np
from spectral_physics.medium_2d import OscillatorGrid2D
from spectral_physics.ndt import build_ndt_profile, score_ndt_state, ndt_defect_mask, NDTProfile

def test_build_ndt_profile():
    grid = OscillatorGrid2D(nx=10, ny=10, kx=1.0, ky=1.0, m=1.0)
    profile = build_ndt_profile(grid, n_modes=10, freq_window=(0.0, 2.0), n_samples=2, noise_level=0.1)
    
    assert isinstance(profile, NDTProfile)
    assert profile.ldos_mean.shape == (10, 10)
    assert profile.ldos_std.shape == (10, 10)
    assert profile.freq_window == (0.0, 2.0)

def test_score_ndt_state_no_defect():
    grid = OscillatorGrid2D(nx=10, ny=10, kx=1.0, ky=1.0, m=1.0)
    profile = build_ndt_profile(grid, n_modes=10, freq_window=(0.0, 2.0), n_samples=1, noise_level=0.0)
    
    # Same grid, should have zero score
    ldos = grid.ldos_map(n_modes=10, freq_window=(0.0, 2.0))
    scores = score_ndt_state(profile, ldos)
    
    assert np.allclose(scores, 0.0)

def test_score_ndt_state_with_defect():
    # Healthy
    grid = OscillatorGrid2D(nx=10, ny=10, kx=1.0, ky=1.0, m=1.0)
    profile = build_ndt_profile(grid, n_modes=10, freq_window=(0.0, 2.0), n_samples=1, noise_level=0.0)
    
    # Defect
    mass_map = np.ones((10, 10))
    mass_map[5, 5] = 10.0
    grid_defect = OscillatorGrid2D(nx=10, ny=10, kx=1.0, ky=1.0, m=1.0, mass_map=mass_map)
    
    ldos = grid_defect.ldos_map(n_modes=10, freq_window=(0.0, 2.0))
    scores = score_ndt_state(profile, ldos)
    
    # Center should have high score
    assert scores[5, 5] > 0.0
    
    # Mask
    mask = ndt_defect_mask(scores, threshold=0.1)
    assert mask[5, 5]
