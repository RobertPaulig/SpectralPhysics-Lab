import pytest
import numpy as np
from spectral_physics.spectrum import Spectrum1D

def test_create_simple_spectrum():
    freqs = [1.0, 2.0, 3.0]
    amps = [0.5, 1.0, 0.5]
    spec = Spectrum1D(freqs, amps)
    
    assert len(spec.freqs) == 3
    assert len(spec.amps) == 3
    assert np.allclose(spec.phases, 0.0)

def test_shape_mismatch():
    freqs = [1.0, 2.0]
    amps = [1.0] # Mismatch
    with pytest.raises(ValueError):
        Spectrum1D(freqs, amps)

def test_power_calculation():
    freqs = [1.0, 2.0]
    amps = [3.0, 4.0]
    spec = Spectrum1D(freqs, amps)
    
    p = spec.power()
    assert np.allclose(p, [9.0, 16.0])
    
    tp = spec.total_power()
    assert np.isclose(tp, 25.0)

def test_normalize_power():
    freqs = [10.0, 20.0]
    amps = [1.0, 1.0] # total power = 2
    spec = Spectrum1D(freqs, amps)
    
    spec.normalize_power(target=8.0)
    # New total power should be 8.0
    # amps should be scaled by sqrt(8/2) = 2.0 -> [2.0, 2.0]
    
    assert np.isclose(spec.total_power(), 8.0)
    assert np.allclose(spec.amps, [2.0, 2.0])

def test_copy():
    freqs = [1.0]
    amps = [1.0]
    spec = Spectrum1D(freqs, amps)
    spec_copy = spec.copy()
    
    spec.amps[0] = 999.0
    assert spec_copy.amps[0] == 1.0 # Should not change
