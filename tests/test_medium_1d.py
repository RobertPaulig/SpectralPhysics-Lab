import pytest
import numpy as np
from spectral_physics.medium_1d import OscillatorChain1D

def test_chain_creation():
    chain = OscillatorChain1D(n_points=5, k_coupling=2.0, m=0.5)
    assert chain.n_points == 5
    assert chain.k == 2.0
    assert chain.m == 0.5

def test_single_oscillator():
    # 1 point, fixed ends.
    # Equation: m x'' = -k(2x) (connected to two walls)
    # w^2 = 2k/m
    k = 1.0
    m = 1.0
    chain = OscillatorChain1D(n_points=1, k_coupling=k, m=m)
    freqs, modes = chain.compute_modes()
    
    expected_w = np.sqrt(2 * k / m)
    assert np.isclose(freqs[0], expected_w)
    assert modes.shape == (1, 1)

def test_two_oscillators():
    # 2 points, fixed ends.
    # Matrix K = [[2k, -k], [-k, 2k]]
    # Eigenvalues of [[2, -1], [-1, 2]] are 1 and 3.
    # So w^2 = k/m * 1 and k/m * 3
    k = 1.0
    m = 1.0
    chain = OscillatorChain1D(n_points=2, k_coupling=k, m=m)
    freqs, modes = chain.compute_modes()
    
    expected_freqs = np.sqrt(np.array([1.0, 3.0]) * k / m)
    
    assert np.allclose(freqs, expected_freqs)
    assert modes.shape == (2, 2)

def test_modes_ordering():
    # Frequencies should be increasing
    chain = OscillatorChain1D(n_points=10)
    freqs, _ = chain.compute_modes()
    
    assert np.all(np.diff(freqs) >= 0)
