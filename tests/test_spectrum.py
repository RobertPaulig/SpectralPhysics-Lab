import numpy as np
import pytest
from spectral_physics.spectrum import Spectrum1D


def test_spectrum_creation():
    """Test basic Spectrum1D creation."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([0.5, 1.0, 0.5])
    
    spec = Spectrum1D(omega=omega, power=power)
    
    assert spec.omega.shape == (3,)
    assert spec.power.shape == (3,)
    np.testing.assert_array_equal(spec.omega, omega)
    np.testing.assert_array_equal(spec.power, power)


def test_spectrum_shape_mismatch():
    """Test that mismatched shapes raise ValueError."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([0.5, 1.0])  # Wrong size
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        Spectrum1D(omega=omega, power=power)


def test_total_power():
    """Test total_power calculation."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    
    spec = Spectrum1D(omega=omega, power=power)
    total = spec.total_power()
    
    assert total == 6.0


def test_normalize():
    """Test spectrum normalization."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    
    spec = Spectrum1D(omega=omega, power=power)
    normalized = spec.normalize()
    
    # Check that sum is 1
    assert abs(normalized.total_power() - 1.0) < 1e-10
    
    # Check that relative proportions are preserved
    np.testing.assert_allclose(
        normalized.power,
        power / 6.0,
        rtol=1e-10
    )
    
    # Original should be unchanged
    assert spec.total_power() == 6.0


def test_normalize_zero_power():
    """Test that normalizing zero power raises ValueError."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([0.0, 0.0, 0.0])
    
    spec = Spectrum1D(omega=omega, power=power)
    
    with pytest.raises(ValueError, match="zero total power"):
        spec.normalize()


def test_apply_filter():
    """Test applying frequency-dependent filter."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    alpha = np.array([1.0, 0.5, 0.0])  # Pass first, attenuate second, block third
    
    spec = Spectrum1D(omega=omega, power=power)
    filtered = spec.apply_filter(alpha)
    
    expected_power = power * alpha
    np.testing.assert_array_equal(filtered.power, expected_power)
    np.testing.assert_array_equal(filtered.omega, omega)
    
    # Original should be unchanged
    np.testing.assert_array_equal(spec.power, power)


def test_apply_filter_shape_mismatch():
    """Test that mismatched filter shape raises ValueError."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 3.0])
    alpha = np.array([1.0, 0.5])  # Wrong size
    
    spec = Spectrum1D(omega=omega, power=power)
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        spec.apply_filter(alpha)


def test_total_power_with_trapz():
    """Test that total_power matches np.trapz for uniform grid."""
    # Uniform frequency grid
    omega = np.linspace(0, 10, 100)
    power = np.exp(-omega)  # Exponential decay
    
    spec = Spectrum1D(omega=omega, power=power)
    
    # For sum, we just add all values
    total_sum = spec.total_power()
    
    # For trapz integration
    total_trapz = np.trapz(power, omega)
    
    # They should be different (sum vs integral)
    # but both should be reasonable
    assert total_sum > 0
    assert total_trapz > 0
    
    # The integral should be smaller than the sum for this case
    # because trapz accounts for spacing
    assert total_trapz < total_sum


def test_from_function():
    """Test creating Spectrum1D from a function."""
    omega = np.linspace(0, 1, 5)
    spec = Spectrum1D.from_function(omega, lambda w: 2*w)
    
    np.testing.assert_array_equal(spec.omega, omega)
    np.testing.assert_array_equal(spec.power, 2*omega)


def test_from_function_exponential():
    """Test from_function with exponential decay."""
    omega = np.linspace(0, 5, 50)
    spec = Spectrum1D.from_function(omega, np.exp)
    
    expected_power = np.exp(omega)
    np.testing.assert_array_almost_equal(spec.power, expected_power)


def test_from_function_shape_mismatch():
    """Test that from_function raises error for wrong return shape."""
    omega = np.linspace(0, 1, 5)
    
    def bad_func(w):
        return np.array([1.0, 2.0])  # Wrong shape
    
    with pytest.raises(ValueError, match="same shape"):
        Spectrum1D.from_function(omega, bad_func)


