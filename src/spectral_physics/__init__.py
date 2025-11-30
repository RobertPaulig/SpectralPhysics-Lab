"""
SpectralPhysics-Lab: базовые численные инструменты
для спектральной физики.
"""

from .root_finding import symmetric_newton
from .spectrum import Spectrum1D
from .medium_1d import OscillatorChain1D
from .grav_toy import spectral_pressure_difference

__all__ = [
    "symmetric_newton",
    "Spectrum1D",
    "OscillatorChain1D",
    "spectral_pressure_difference",
]

