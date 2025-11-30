"""
SpectralPhysics-Lab: A library for spectral physics simulations.

Modules:
    - root_finding: Symmetric Newton method for root finding
    - spectrum: 1D spectrum analysis tools
    - medium_1d: 1D oscillator chain model
    - grav_toy: Toy model for gravity via spectral shadows
"""

__all__ = ["root_finding", "spectrum", "medium_1d", "grav_toy"]

# Note: We don't import modules at top level to avoid heavy dependencies
# Users should import specific modules as needed:
#   from spectral_physics import root_finding
#   from spectral_physics.spectrum import Spectrum1D
