import numpy as np
import pytest
from spectral_physics.spectrum import Spectrum1D
from spectral_physics.grav_toy import spectral_pressure_difference


def test_spectral_pressure_equal_transparency():
    """Test that equal transparency gives zero pressure difference."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 1.0])
    spectrum = Spectrum1D(omega=omega, power=power)
    
    alpha = np.array([0.5, 0.5, 0.5])  # Same on both sides
    
    delta_p = spectral_pressure_difference(spectrum, alpha, alpha)
    
    assert abs(delta_p) < 1e-14


def test_spectral_pressure_left_blocks_more():
    """Test that if left blocks more, pressure pushes right (positive ΔP)."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 1.0, 1.0])
    spectrum = Spectrum1D(omega=omega, power=power)
    
    alpha_left = np.array([0.2, 0.2, 0.2])   # Blocks more (low transparency)
    alpha_right = np.array([0.8, 0.8, 0.8])  # Blocks less (high transparency)
    
    delta_p = spectral_pressure_difference(spectrum, alpha_left, alpha_right)
    
    # ΔP = Σ power * (alpha_right - alpha_left)
    # = Σ 1.0 * (0.8 - 0.2) = 3 * 0.6 = 1.8
    expected = 3 * 0.6
    assert abs(delta_p - expected) < 1e-14


def test_spectral_pressure_right_blocks_more():
    """Test that if right blocks more, pressure pushes left (negative ΔP)."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([2.0, 2.0, 2.0])
    spectrum = Spectrum1D(omega=omega, power=power)
    
    alpha_left = np.array([0.9, 0.9, 0.9])   # Blocks less
    alpha_right = np.array([0.1, 0.1, 0.1])  # Blocks more
    
    delta_p = spectral_pressure_difference(spectrum, alpha_left, alpha_right)
    
    # ΔP = Σ 2.0 * (0.1 - 0.9) = 3 * 2.0 * (-0.8) = -4.8
    expected = 3 * 2.0 * (-0.8)
    assert abs(delta_p - expected) < 1e-14


def test_spectral_pressure_frequency_dependent():
    """Test with frequency-dependent transparency."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    spectrum = Spectrum1D(omega=omega, power=power)
    
    # Left transparent to low freq, blocks high freq
    alpha_left = np.array([1.0, 0.5, 0.0])
    
    # Right blocks low freq, transparent to high freq
    alpha_right = np.array([0.0, 0.5, 1.0])
    
    delta_p = spectral_pressure_difference(spectrum, alpha_left, alpha_right)
    
    # ΔP = 1.0*(0.0-1.0) + 2.0*(0.5-0.5) + 3.0*(1.0-0.0)
    #    = -1.0 + 0.0 + 3.0 = 2.0
    expected = -1.0 + 0.0 + 3.0
    assert abs(delta_p - expected) < 1e-14


def test_spectral_pressure_shape_mismatch_left():
    """Test that mismatched left alpha shape raises ValueError."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    spectrum = Spectrum1D(omega=omega, power=power)
    
    alpha_left = np.array([0.5, 0.5])  # Wrong size
    alpha_right = np.array([0.5, 0.5, 0.5])
    
    with pytest.raises(ValueError, match="alpha_left"):
        spectral_pressure_difference(spectrum, alpha_left, alpha_right)


def test_spectral_pressure_shape_mismatch_right():
    """Test that mismatched right alpha shape raises ValueError."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    spectrum = Spectrum1D(omega=omega, power=power)
    
    alpha_left = np.array([0.5, 0.5, 0.5])
    alpha_right = np.array([0.5, 0.5])  # Wrong size
    
    with pytest.raises(ValueError, match="alpha_right"):
        spectral_pressure_difference(spectrum, alpha_left, alpha_right)


def test_spectral_pressure_realistic_scenario():
    """Test a more realistic scenario with Gaussian spectrum."""
    # Gaussian spectrum
    omega = np.linspace(0, 10, 100)
    power = np.exp(-(omega - 5)**2 / 2)
    spectrum = Spectrum1D(omega=omega, power=power)
    
    # Material that blocks high frequencies more
    alpha_left = np.exp(-omega / 10)  # Decays with frequency
    alpha_right = np.ones_like(omega) * 0.8  # Uniform transparency
    
    delta_p = spectral_pressure_difference(spectrum, alpha_left, alpha_right)
    
    # Should be positive (because left blocks more at high freq where power is)
    # The exact value depends on the convolution, but it should be positive
    assert delta_p != 0  # Non-trivial result
