import numpy as np
import pytest
from spectral_physics.material import MaterialSignature
from spectral_physics.spectrum import Spectrum1D


def test_distance_identical_spectra():
    """Test that distance between identical spectra is zero."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 1.0])
    
    spec1 = Spectrum1D(omega=omega, power=power)
    spec2 = Spectrum1D(omega=omega, power=power.copy())
    
    sig = MaterialSignature(reference=spec1)
    distance = sig.distance_l2(spec2)
    
    assert abs(distance) < 1e-10


def test_distance_slightly_different():
    """Test distance for slightly different spectra."""
    omega = np.array([1.0, 2.0, 3.0])
    power1 = np.array([1.0, 2.0, 1.0])
    power2 = np.array([1.0, 2.1, 1.0])  # Slight difference
    
    spec1 = Spectrum1D(omega=omega, power=power1)
    spec2 = Spectrum1D(omega=omega, power=power2)
    
    sig = MaterialSignature(reference=spec1)
    distance = sig.distance_l2(spec2)
    
    # Should be small but non-zero
    assert distance > 0
    assert distance < 1.0


def test_distance_very_different():
    """Test distance for very different spectra."""
    omega = np.array([1.0, 2.0, 3.0])
    power1 = np.array([1.0, 2.0, 1.0])
    power2 = np.array([5.0, 0.5, 3.0])  # Very different shape
    
    spec1 = Spectrum1D(omega=omega, power=power1)
    spec2 = Spectrum1D(omega=omega, power=power2)
    
    sig = MaterialSignature(reference=spec1)
    distance = sig.distance_l2(spec2)
    
    # Should be large
    assert distance > 0.5


def test_distance_mismatched_frequencies():
    """Test that mismatched frequency grids raise ValueError."""
    omega1 = np.array([1.0, 2.0, 3.0])
    omega2 = np.array([1.0, 2.0, 3.0, 4.0])  # Different length
    
    spec1 = Spectrum1D(omega=omega1, power=np.array([1.0, 2.0, 1.0]))
    spec2 = Spectrum1D(omega=omega2, power=np.array([1.0, 2.0, 1.0, 1.0]))
    
    sig = MaterialSignature(reference=spec1)
    
    with pytest.raises(ValueError, match="Frequency grids do not match"):
        sig.distance_l2(spec2)


def test_is_anomalous_normal():
    """Test that identical spectrum is not anomalous."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 1.0])
    
    spec = Spectrum1D(omega=omega, power=power)
    sig = MaterialSignature(reference=spec)
    
    is_anom = sig.is_anomalous(spec, threshold=0.01)
    
    assert is_anom is False


def test_is_anomalous_below_threshold():
    """Test that small difference is not anomalous."""
    omega = np.array([1.0, 2.0, 3.0])
    power1 = np.array([1.0, 2.0, 1.0])
    power2 = np.array([1.0, 2.01, 1.0])  # Tiny difference
    
    spec1 = Spectrum1D(omega=omega, power=power1)
    spec2 = Spectrum1D(omega=omega, power=power2)
    
    sig = MaterialSignature(reference=spec1)
    is_anom = sig.is_anomalous(spec2, threshold=1.0)  # High threshold
    
    assert is_anom is False


def test_is_anomalous_above_threshold():
    """Test that large difference is anomalous."""
    omega = np.array([1.0, 2.0, 3.0])
    power1 = np.array([1.0, 2.0, 1.0])
    power2 = np.array([5.0, 0.5, 3.0])  # Very different
    
    spec1 = Spectrum1D(omega=omega, power=power1)
    spec2 = Spectrum1D(omega=omega, power=power2)
    
    sig = MaterialSignature(reference=spec1)
    is_anom = sig.is_anomalous(spec2, threshold=0.1)  # Low threshold
    
    assert is_anom is True


def test_distance_normalized():
    """Test that distance is based on normalized spectra."""
    omega = np.array([1.0, 2.0, 3.0])
    power1 = np.array([1.0, 2.0, 1.0])
    power2 = np.array([2.0, 4.0, 2.0])  # Same shape, double amplitude
    
    spec1 = Spectrum1D(omega=omega, power=power1)
    spec2 = Spectrum1D(omega=omega, power=power2)
    
    sig = MaterialSignature(reference=spec1)
    distance = sig.distance_l2(spec2)
    
    # Should be zero since normalized shapes are identical
    assert abs(distance) < 1e-10
