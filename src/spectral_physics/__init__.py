"""
SpectralPhysics-Lab: базовые численные инструменты
для спектральной физики.
"""

from .root_finding import symmetric_newton
from .spectrum import Spectrum1D
from .medium_1d import OscillatorChain1D
from .grav_toy import spectral_pressure_difference
from .material import MaterialSignature
from .timeseries import timeseries_to_spectrum
from .diagnostics import (
    ChannelConfig, 
    MultiChannelConfig,
    SpectralAnalyzer, 
    HealthMonitor, 
    build_health_profile,
    average_spectrum,
    spectral_band_power,
    spectral_entropy
)
from .io import (
    load_timeseries_csv, 
    save_spectrum_npz, 
    load_spectrum_npz,
    save_health_profile,
    load_health_profile
)
from .report import generate_markdown_report

__all__ = [
    "symmetric_newton",
    "Spectrum1D",
    "OscillatorChain1D",
    "spectral_pressure_difference",
    "MaterialSignature",
    "timeseries_to_spectrum",
    "ChannelConfig",
    "MultiChannelConfig",
    "SpectralAnalyzer",
    "HealthMonitor",
    "build_health_profile",
    "average_spectrum",
    "spectral_band_power",
    "spectral_entropy",
    "load_timeseries_csv",
    "save_spectrum_npz",
    "load_spectrum_npz",
    "save_health_profile",
    "load_health_profile",
    "generate_markdown_report",
]


