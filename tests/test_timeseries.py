import numpy as np
import pytest
from spectral_physics.timeseries import timeseries_to_spectrum


def test_single_sine_wave():
    """Test that single sine wave produces peak at correct frequency."""
    # Generate 50 Hz sine wave
    freq = 50.0  # Hz
    duration = 1.0  # seconds
    dt = 0.001  # 1 ms sampling
    
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t)
    
    spectrum = timeseries_to_spectrum(signal, dt=dt)
    
    # Find peak frequency
    peak_idx = np.argmax(spectrum.power)
    peak_omega = spectrum.omega[peak_idx]
    peak_freq_hz = peak_omega / (2 * np.pi)
    
    # Should be close to 50 Hz
    assert abs(peak_freq_hz - freq) < 2.0  # Within 2 Hz


def test_two_sine_waves():
    """Test that sum of two sines produces two peaks."""
    freq1 = 50.0  # Hz
    freq2 = 120.0  # Hz
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    
    spectrum = timeseries_to_spectrum(signal, dt=dt)
    
    # Convert omega to Hz
    freq_hz = spectrum.omega / (2 * np.pi)
    
    # Find two largest peaks
    peak_indices = np.argsort(spectrum.power)[-2:]
    peak_freqs = sorted(freq_hz[peak_indices])
    
    # Should have peaks near 50 and 120 Hz
    assert abs(peak_freqs[0] - freq1) < 5.0
    assert abs(peak_freqs[1] - freq2) < 5.0


def test_constant_signal():
    """Test that constant signal has maximum only near zero frequency."""
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = np.ones_like(t) * 5.0  # Constant
    
    spectrum = timeseries_to_spectrum(signal, dt=dt)
    
    # Peak should be at or very close to zero frequency
    peak_idx = np.argmax(spectrum.power)
    peak_omega = spectrum.omega[peak_idx]
    
    assert peak_omega < 10.0  # Very low frequency


def test_window_none():
    """Test that window=None works."""
    freq = 50.0
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t)
    
    spectrum = timeseries_to_spectrum(signal, dt=dt, window=None)
    
    # Should still produce spectrum
    assert len(spectrum.omega) > 0
    assert len(spectrum.power) > 0


def test_window_hann():
    """Test that window='hann' works."""
    freq = 50.0
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t)
    
    spectrum = timeseries_to_spectrum(signal, dt=dt, window="hann")
    
    # Should still produce spectrum
    assert len(spectrum.omega) > 0
    assert len(spectrum.power) > 0


def test_invalid_window():
    """Test that invalid window raises ValueError."""
    signal = np.array([1.0, 2.0, 3.0])
    dt = 0.1
    
    with pytest.raises(ValueError, match="Unknown window type"):
        timeseries_to_spectrum(signal, dt=dt, window="invalid")


def test_empty_signal():
    """Test that empty signal raises ValueError."""
    signal = np.array([])
    dt = 0.1
    
    with pytest.raises(ValueError, match="must not be empty"):
        timeseries_to_spectrum(signal, dt=dt)


def test_multidimensional_signal():
    """Test that 2D signal raises ValueError."""
    signal = np.array([[1.0, 2.0], [3.0, 4.0]])
    dt = 0.1
    
    with pytest.raises(ValueError, match="must be 1D array"):
        timeseries_to_spectrum(signal, dt=dt)


def test_dc_removal():
    """Test that DC component is removed."""
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = 10.0 + np.sin(2 * np.pi * 50 * t)  # DC offset + sine
    
    spectrum = timeseries_to_spectrum(signal, dt=dt)
    
    # DC component (first element) should be small after DC removal
    # Note: won't be exactly zero due to window, but should be reduced
    assert spectrum.power[0] < np.max(spectrum.power)


def test_positive_frequencies_only():
    """Test that only positive frequencies are returned."""
    signal = np.random.randn(100)
    dt = 0.01
    
    spectrum = timeseries_to_spectrum(signal, dt=dt)
    
    # All frequencies should be >= 0
    assert np.all(spectrum.omega >= 0)
