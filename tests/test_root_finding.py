import numpy as np
import pytest
from spectral_physics.root_finding import symmetric_newton


def test_simple_quadratic():
    """Test f(x) = x^2 - 2, root should be sqrt(2)."""
    def f(x):
        return x**2 - 2
    
    x_root, n_iter = symmetric_newton(f, x0=1.0)
    
    expected = np.sqrt(2)
    assert abs(x_root - expected) < 1e-8
    assert n_iter < 50


def test_cubic_at_zero():
    """Test f(x) = x^3, root at 0 with flat minimum."""
    def f(x):
        return x**3
    
    x_root, n_iter = symmetric_newton(f, x0=0.1, tol=1e-6)
    
    assert abs(x_root) < 1e-4
    assert n_iter < 50


def test_with_kink():
    """Test function with kink: f(x) = abs(x) - 1e-3."""
    def f(x):
        return abs(x) - 1e-3
    
    # Start from positive side
    x_root, n_iter = symmetric_newton(f, x0=1.0, tol=1e-6)
    
    # Should converge to one of the roots (±1e-3)
    assert abs(abs(x_root) - 1e-3) < 1e-5


def test_cubic_polynomial():
    """Test f(x) = x^3 - x - 1."""
    def f(x):
        return x**3 - x - 1
    
    x_root, n_iter = symmetric_newton(f, x0=1.5)
    
    # Check that it's actually a root
    assert abs(f(x_root)) < 1e-8
    assert n_iter < 50


def test_return_type():
    """Test that function returns tuple of (float, int)."""
    def f(x):
        return x - 5
    
    result = symmetric_newton(f, x0=1.0)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], int)


def test_convergence_count():
    """Test that iteration count is reasonable."""
    def f(x):
        return x - 10  # Linear, should converge in 1 iteration
    
    x_root, n_iter = symmetric_newton(f, x0=0.0)
    
    assert abs(x_root - 10) < 1e-8
    assert n_iter <= 5  # Should be very fast for linear


def test_exponential_function():
    """Test f(x) = exp(x) - 2."""
    def f(x):
        return np.exp(x) - 2
    
    x_root, n_iter = symmetric_newton(f, x0=0.0)
    
    expected = np.log(2)
    assert abs(x_root - expected) < 1e-8


def test_trigonometric_function():
    """Test f(x) = sin(x)."""
    def f(x):
        return np.sin(x)
    
    x_root, n_iter = symmetric_newton(f, x0=3.0)
    
    # Should converge to pi
    assert abs(x_root - np.pi) < 1e-6


def test_max_iterations():
    """Test that max_iter limit is respected."""
    def f(x):
        return x**2 + 1  # No real root
    
    x_root, n_iter = symmetric_newton(f, x0=1.0, max_iter=10)
    
    assert n_iter == 10


def test_initial_guess_is_root():
    """Test when initial guess is already the root."""
    def f(x):
        return x - 5
    
    x_root, n_iter = symmetric_newton(f, x0=5.0)
    
    assert abs(x_root - 5.0) < 1e-8
    assert n_iter == 0  # Should converge immediately


def test_tol_step_stops_on_flat_region():
    """Test that tol_step criterion stops iteration on flat regions."""
    def f(x):
        return x**3
    
    # With very tight tol, we'd wait forever
    # But tol_step should kick in and stop earlier
    x_root, n_iter = symmetric_newton(f, x0=1.0, tol=1e-20, tol_step=1e-6)
    
    # Should stop due to small step, not small function value
    assert n_iter < 50
    # Result should still be close to zero
    assert abs(x_root) < 0.1

