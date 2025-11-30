"""
SpectralPhysics-Lab: базовые численные инструменты
для спектральной физики.
"""

__all__ = [
    "symmetric_newton",
    "Spectrum1D",
    "OscillatorChain1D",
    "spectral_pressure_difference",
]

# Lazy imports - пользователи должны импортировать нужные модули явно:
#   from spectral_physics.root_finding import symmetric_newton
#   from spectral_physics.spectrum import Spectrum1D
#   from spectral_physics.medium_1d import OscillatorChain1D
#   from spectral_physics.grav_toy import spectral_pressure_difference
