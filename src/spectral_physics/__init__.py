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
from .diagnostics import ChannelConfig, SpectralAnalyzer, HealthMonitor
from .io import load_timeseries_csv, save_spectrum_npz, load_spectrum_npz

__all__ = [
    "symmetric_newton",
    "Spectrum1D",
    "OscillatorChain1D",
    "spectral_pressure_difference",
    "MaterialSignature",
    "timeseries_to_spectrum",
    "ChannelConfig",
    "SpectralAnalyzer",
    "HealthMonitor",
    "load_timeseries_csv",
    "save_spectrum_npz",
    "load_spectrum_npz",
]

