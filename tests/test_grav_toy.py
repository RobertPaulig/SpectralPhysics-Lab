import pytest
import numpy as np
from spectral_physics.spectrum import Spectrum1D
from spectral_physics.grav_toy import spectral_pressure_difference

def test_pressure_diff_positive():
    # Left has more power
    s1 = Spectrum1D([1, 2], [2, 2]) # power = 4+4=8
    s2 = Spectrum1D([1, 2], [1, 1]) # power = 1+1=2
    
    diff = spectral_pressure_difference(s1, s2)
    assert np.isclose(diff, 6.0)

def test_pressure_diff_negative():
    # Right has more power
    s1 = Spectrum1D([1], [1]) # power 1
    s2 = Spectrum1D([1], [3]) # power 9
    
    diff = spectral_pressure_difference(s1, s2)
    assert np.isclose(diff, -8.0)
