import numpy as np
import pytest
from spectral_physics.spectrum import Spectrum1D
from spectral_physics.diagnostics import extract_features, spectral_band_power, spectral_entropy

def test_extract_features_structure():
    # Create a dummy spectrum
    omega = np.linspace(0, 100, 101)
    power = np.ones_like(omega)
    spec = Spectrum1D(omega, power)
    
    bands = [(0, 10), (10, 50)]
    features = extract_features(spec, bands)
    
    # Expect len(bands) + 1 (entropy)
    assert len(features) == 3
    assert isinstance(features, np.ndarray)

def test_spectral_band_power():
    # Spectrum with power 1.0 everywhere
    # omega is 0..100 rad/s -> freq is 0..100/(2pi) ~ 15.9 Hz
    # Let's use Hz directly for clarity in construction if possible, 
    # but Spectrum1D takes omega.
    
    # Let's make a simple discrete spectrum
    # 1 Hz = 2pi rad/s
    freq_hz = np.array([1.0, 2.0, 3.0])
    omega = freq_hz * 2 * np.pi
    power = np.array([10.0, 20.0, 30.0])
    
    spec = Spectrum1D(omega, power)
    
    # Band 0.5-1.5 Hz should capture 1.0 Hz (power 10)
    p1 = spectral_band_power(spec, 0.5, 1.5)
    assert p1 == 10.0
    
    # Band 1.5-2.5 Hz should capture 2.0 Hz (power 20)
    p2 = spectral_band_power(spec, 1.5, 2.5)
    assert p2 == 20.0
    
    # Band 0-10 Hz should capture all (60)
    p_all = spectral_band_power(spec, 0.0, 10.0)
    assert p_all == 60.0

def test_spectral_entropy():
    # 1. Flat spectrum (max entropy)
    omega = np.array([1, 2, 3, 4])
    power = np.array([1, 1, 1, 1]) # Normalized: 0.25 each
    spec = Spectrum1D(omega, power)
    
    # H = - sum(0.25 * ln(0.25)) * 4 = - ln(0.25) = ln(4)
    expected = np.log(4)
    assert np.isclose(spectral_entropy(spec), expected)
    
    # 2. Delta function (min entropy)
    power2 = np.array([1, 0, 0, 0])
    spec2 = Spectrum1D(omega, power2)
    # H = - (1*ln(1) + 0) = 0
    assert spectral_entropy(spec2) == 0.0
