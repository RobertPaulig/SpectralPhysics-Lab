import numpy as np
import pytest
from spectral_physics.medium_1d import OscillatorChain1D


def test_oscillator_chain_creation():
    """Test basic OscillatorChain1D creation."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0, gamma=0.0)
    
    assert chain.n == 10
    assert chain.k == 1.0
    assert chain.m == 1.0
    assert chain.gamma == 0.0


def test_oscillator_chain_validation():
    """Test parameter validation."""
    # n must be >= 1
    with pytest.raises(ValueError, match="Number of oscillators"):
        OscillatorChain1D(n=0, k=1.0, m=1.0)
    
    # m must be positive
    with pytest.raises(ValueError, match="Mass must be positive"):
        OscillatorChain1D(n=10, k=1.0, m=0.0)
    
    # k must be non-negative
    with pytest.raises(ValueError, match="Stiffness must be non-negative"):
        OscillatorChain1D(n=10, k=-1.0, m=1.0)
    
    # gamma must be non-negative
    with pytest.raises(ValueError, match="Damping must be non-negative"):
        OscillatorChain1D(n=10, k=1.0, m=1.0, gamma=-0.1)


def test_stiffness_matrix_shape():
    """Test stiffness matrix has correct shape."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0)
    K = chain.stiffness_matrix()
    
    assert K.shape == (10, 10)


def test_stiffness_matrix_symmetric():
    """Test that stiffness matrix is symmetric."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0)
    K = chain.stiffness_matrix()
    
    np.testing.assert_array_almost_equal(K, K.T)


def test_stiffness_matrix_structure():
    """Test stiffness matrix has correct tridiagonal structure."""
    chain = OscillatorChain1D(n=5, k=2.0, m=1.0)
    K = chain.stiffness_matrix()
    
    # Diagonal should be 2*k
    diagonal = np.diag(K)
    np.testing.assert_array_almost_equal(diagonal, np.full(5, 4.0))
    
    # Off-diagonal should be -k
    off_diag_upper = np.diag(K, k=1)
    np.testing.assert_array_almost_equal(off_diag_upper, np.full(4, -2.0))
    
    off_diag_lower = np.diag(K, k=-1)
    np.testing.assert_array_almost_equal(off_diag_lower, np.full(4, -2.0))


def test_eigenmodes_shape():
    """Test eigenmodes return correct shapes."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0)
    omega, modes = chain.eigenmodes()
    
    assert omega.shape == (10,)
    assert modes.shape == (10, 10)


def test_eigenmodes_positive_frequencies():
    """Test that all eigenfrequencies are non-negative."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0)
    omega, modes = chain.eigenmodes()
    
    assert np.all(omega >= 0)


def test_eigenmodes_count():
    """Test that number of modes equals n."""
    for n in [1, 5, 10, 20]:
        chain = OscillatorChain1D(n=n, k=1.0, m=1.0)
        omega, modes = chain.eigenmodes()
        
        assert len(omega) == n
        assert modes.shape[1] == n


def test_eigenmodes_sorted():
    """Test that eigenfrequencies are sorted."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0)
    omega, modes = chain.eigenmodes()
    
    # eigh returns sorted eigenvalues
    assert np.all(omega[:-1] <= omega[1:])


def test_eigenmodes_orthogonal():
    """Test that eigenmodes are orthogonal."""
    chain = OscillatorChain1D(n=10, k=1.0, m=1.0)
    omega, modes = chain.eigenmodes()
    
    # Eigenvectors should be orthogonal: V^T V = I
    identity = modes.T @ modes
    np.testing.assert_array_almost_equal(identity, np.eye(10), decimal=10)


def test_different_parameters():
    """Test oscillator chain with different parameters."""
    chain1 = OscillatorChain1D(n=10, k=1.0, m=1.0)
    chain2 = OscillatorChain1D(n=10, k=2.0, m=1.0)  # Stiffer
    chain3 = OscillatorChain1D(n=10, k=1.0, m=2.0)  # Heavier
    
    omega1, _ = chain1.eigenmodes()
    omega2, _ = chain2.eigenmodes()
    omega3, _ = chain3.eigenmodes()
    
    # Stiffer chain should have higher frequencies
    assert np.all(omega2 > omega1)
    
    # Heavier chain should have lower frequencies
    assert np.all(omega3 < omega1)
