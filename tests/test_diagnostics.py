import numpy as np
import pytest
from spectral_physics.diagnostics import ChannelConfig, SpectralAnalyzer, HealthMonitor
from spectral_physics.spectrum import Spectrum1D


def test_channel_config_creation():
    """Test basic ChannelConfig creation."""
    config = ChannelConfig(
        name="test",
        dt=0.001,
        window="hann",
        freq_min=10.0,
        freq_max=100.0
    )
    
    assert config.name == "test"
    assert config.dt == 0.001
    assert config.window == "hann"
    assert config.freq_min == 10.0
    assert config.freq_max == 100.0


def test_spectral_analyzer_basic():
    """Test SpectralAnalyzer with known sine wave."""
    freq = 50.0  # Hz
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t)
    
    config = ChannelConfig(name="test", dt=dt)
    analyzer = SpectralAnalyzer(config)
    spectrum = analyzer.analyze(signal)
    
    # Should have peak near 50 Hz
    freq_hz = spectrum.omega / (2 * np.pi)
    peak_idx = np.argmax(spectrum.power)
    peak_freq = freq_hz[peak_idx]
    
    assert abs(peak_freq - freq) < 5.0  # Within 5 Hz


def test_spectral_analyzer_freq_filter():
    """Test that freq_min/freq_max actually filter frequencies."""
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    # Mix of low and high frequencies
    signal = (
        np.sin(2 * np.pi * 10 * t) +
        np.sin(2 * np.pi * 100 * t)
    )
    
    # Analyze with no filter
    config_all = ChannelConfig(name="all", dt=dt)
    analyzer_all = SpectralAnalyzer(config_all)
    spectrum_all = analyzer_all.analyze(signal)
    
    # Analyze with filter (only 50-150 Hz)
    config_filtered = ChannelConfig(
        name="filtered",
        dt=dt,
        freq_min=50.0,
        freq_max=150.0
    )
    analyzer_filtered = SpectralAnalyzer(config_filtered)
    spectrum_filtered = analyzer_filtered.analyze(signal)
    
    # Filtered spectrum should be smaller
    assert len(spectrum_filtered.omega) < len(spectrum_all.omega)
    
    # All frequencies in filtered spectrum should be in range
    freq_hz = spectrum_filtered.omega / (2 * np.pi)
    assert np.all(freq_hz >= 50.0)
    assert np.all(freq_hz <= 150.0)


def test_health_monitor_score():
    """Test HealthMonitor.score calculation."""
    omega = np.array([1.0, 2.0, 3.0])
    power_ref = np.array([1.0, 2.0, 1.0])
    power_current = np.array([1.0, 2.0, 1.0])  # Identical
    
    ref_spec = Spectrum1D(omega=omega, power=power_ref)
    current_spec = Spectrum1D(omega=omega, power=power_current)
    
    monitor = HealthMonitor(reference=ref_spec, threshold=0.1)
    score = monitor.score(current_spec)
    
    # Score should be zero for identical spectra
    assert abs(score) < 1e-10


def test_health_monitor_is_anomalous_false():
    """Test that  HealthMonitor correctly identifies normal spectrum."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 1.0])
    
    spec = Spectrum1D(omega=omega, power=power)
    monitor = HealthMonitor(reference=spec, threshold=0.1)
    
    is_anom = monitor.is_anomalous(spec)
    
    assert is_anom is False


def test_health_monitor_is_anomalous_true():
    """Test that HealthMonitor correctly identifies anomaly."""
    omega = np.array([1.0, 2.0, 3.0])
    power_ref = np.array([1.0, 2.0, 1.0])
    power_anom = np.array([5.0, 0.5, 3.0])  # Very different
    
    ref_spec = Spectrum1D(omega=omega, power=power_ref)
    anom_spec = Spectrum1D(omega=omega, power=power_anom)
    
    monitor = HealthMonitor(reference=ref_spec, threshold=0.1)
    is_anom = monitor.is_anomalous(anom_spec)
    
    assert is_anom is True


def test_spectral_analyzer_window_none():
    """Test SpectralAnalyzer with no window."""
    freq = 50.0
    duration = 1.0
    dt = 0.001
    
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t)
    
    config = ChannelConfig(name="test", dt=dt, window=None)
    analyzer = SpectralAnalyzer(config)
    spectrum = analyzer.analyze(signal)
    
    # Should still work
    assert len(spectrum.omega) > 0
    assert len(spectrum.power) > 0
