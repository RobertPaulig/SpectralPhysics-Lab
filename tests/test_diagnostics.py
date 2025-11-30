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


def test_average_spectrum_basic():
    """Test averaging two identical spectra."""
    from spectral_physics.diagnostics import average_spectrum
    
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([1.0, 2.0, 1.0])
    
    spec1 = Spectrum1D(omega=omega, power=power)
    spec2 = Spectrum1D(omega=omega, power=power)
    
    avg = average_spectrum([spec1, spec2])
    
    np.testing.assert_array_equal(avg.omega, omega)
    np.testing.assert_array_equal(avg.power, power)


def test_average_spectrum_different_power():
    """Test averaging spectra with different powers."""
    from spectral_physics.diagnostics import average_spectrum
    
    omega = np.array([1.0, 2.0])
    spec1 = Spectrum1D(omega=omega, power=np.array([1.0, 1.0]))
    spec2 = Spectrum1D(omega=omega, power=np.array([3.0, 3.0]))
    
    avg = average_spectrum([spec1, spec2])
    
    # Average of 1 and 3 is 2
    expected = np.array([2.0, 2.0])
    np.testing.assert_array_equal(avg.power, expected)


def test_average_spectrum_mismatch():
    """Test that averaging spectra with different grids raises ValueError."""
    from spectral_physics.diagnostics import average_spectrum
    
    spec1 = Spectrum1D(omega=np.array([1.0]), power=np.array([1.0]))
    spec2 = Spectrum1D(omega=np.array([2.0]), power=np.array([1.0]))
    
    with pytest.raises(ValueError, match="different frequency grid"):
        average_spectrum([spec1, spec2])


def test_build_health_profile_simple():
    """Test building health profile from training data."""
    from spectral_physics.diagnostics import build_health_profile
    
    omega = np.array([1.0, 2.0])
    power = np.array([1.0, 1.0])
    spec = Spectrum1D(omega=omega, power=power)
    
    training_data = {
        "ch1": [spec, spec],
        "ch2": [spec, spec]
    }
    
    profile = build_health_profile(training_data)
    
    assert "ch1" in profile.signatures
    assert "ch2" in profile.signatures
    
    # Check that signatures are correct (distance to original should be 0)
    scores = profile.score({"ch1": spec, "ch2": spec})
    assert scores["ch1"] < 1e-10
    assert scores["ch2"] < 1e-10


def test_spectral_band_power():
    """Test spectral band power calculation."""
    from spectral_physics.diagnostics import spectral_band_power
    
    # 10 Hz and 100 Hz
    omega = np.array([2*np.pi*10, 2*np.pi*100])
    power = np.array([1.0, 2.0])
    spec = Spectrum1D(omega=omega, power=power)
    
    # Band covering only 10 Hz
    p1 = spectral_band_power(spec, freq_min=5, freq_max=15)
    assert abs(p1 - 1.0) < 1e-10
    
    # Band covering only 100 Hz
    p2 = spectral_band_power(spec, freq_min=90, freq_max=110)
    assert abs(p2 - 2.0) < 1e-10
    
    # Band covering both
    p3 = spectral_band_power(spec, freq_min=0, freq_max=200)
    assert abs(p3 - 3.0) < 1e-10


def test_spectral_entropy():
    """Test spectral entropy calculation."""
    from spectral_physics.diagnostics import spectral_entropy
    
    omega = np.array([1.0, 2.0, 3.0])
    
    # Uniform spectrum (max entropy)
    spec_uniform = Spectrum1D(omega=omega, power=np.array([1.0, 1.0, 1.0]))
    h_uniform = spectral_entropy(spec_uniform)
    
    # Peaked spectrum (lower entropy)
    spec_peaked = Spectrum1D(omega=omega, power=np.array([0.0, 10.0, 0.0]))
    h_peaked = spectral_entropy(spec_peaked)
    
    assert h_uniform > h_peaked
    # For single peak, entropy should be 0 (-1*log(1))
    assert abs(h_peaked) < 1e-10


def test_extract_features():
    """Test feature extraction."""
    from spectral_physics.diagnostics import extract_features
    
    # 10 Hz (power 1.0) and 100 Hz (power 2.0)
    omega = np.array([2*np.pi*10, 2*np.pi*100])
    power = np.array([1.0, 2.0])
    spec = Spectrum1D(omega=omega, power=power)
    
    bands = [(5, 15), (90, 110)]
    
    features = extract_features(spec, bands)
    
    assert len(features) == 3  # 2 bands + 1 entropy
    assert abs(features[0] - 1.0) < 1e-10
    assert abs(features[1] - 2.0) < 1e-10
    # Entropy should be > 0 for 2 peaks
    assert features[2] > 0



