import pytest
import math
from spectral_physics.root_finding import symmetric_newton

def test_linear_function():
    # f(x) = x - 3 -> root is 3
    f = lambda x: x - 3
    root = symmetric_newton(f, x0=0)
    assert abs(root - 3.0) < 1e-6

def test_quadratic_function():
    # f(x) = x^2 - 4 -> roots are 2 and -2
    f = lambda x: x**2 - 4
    
    # Start near 2
    root1 = symmetric_newton(f, x0=1.0)
    assert abs(root1 - 2.0) < 1e-6
    
    # Start near -2
    root2 = symmetric_newton(f, x0=-1.0)
    assert abs(root2 - (-2.0)) < 1e-6

def test_bad_derivative_function():
    # f(x) = x^(1/3). Derivative at 0 is infinite.
    # But symmetric difference might handle it or jump over it.
    # Actually x^(1/3) is defined for negative x if we handle it carefully, 
    # but pow(x, 1/3) in python for negative x returns complex.
    # Let's use a function that behaves like x^(1/3) but is safe: np.cbrt(x)
    import numpy as np
    f = lambda x: np.cbrt(x)
    
    # Start somewhere away from 0
    root = symmetric_newton(f, x0=0.5)
    
    # Should converge to 0
    assert abs(root) < 1e-4

def test_no_convergence_small_derivative():
    # f(x) = x^3 around 0 has small derivative, but let's try a flat function
    # f(x) = arctan(x) has derivative 1/(1+x^2).
    # Let's try f(x) = 1 (no root)
    f = lambda x: 1.0
    # Derivative is 0. Should handle gracefully (return start or last point)
    root = symmetric_newton(f, x0=0.0)
    # It won't find a root, but shouldn't crash
    assert root == 0.0 # Should stay or break
