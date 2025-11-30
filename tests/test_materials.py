import numpy as np
import pytest
from spectral_physics.materials import MaterialPatch, effective_coupling
from spectral_physics.atoms import AtomicResonator

def test_material_patch_spectrum():
    # Atom A: line at 1.0, power 1.0
    A = AtomicResonator.from_lines("A", [(1.0, 1.0)], max_bonds=1)
    # Atom B: line at 2.0, power 1.0
    B = AtomicResonator.from_lines("B", [(2.0, 1.0)], max_bonds=1)
    
    # Mix 50/50
    patch = MaterialPatch(atoms=[A, B], weights=np.array([0.5, 0.5]))
    
    spec = patch.surface_spectrum()
    
    assert len(spec.omega) == 2
    assert np.allclose(spec.omega, [1.0, 2.0])
    assert np.allclose(spec.power, [0.5, 0.5])

def test_effective_coupling():
    # Atom A: line at 1.0, power 1.0
    A = AtomicResonator.from_lines("A", [(1.0, 1.0)], max_bonds=1)
    patch = MaterialPatch(atoms=[A], weights=np.array([1.0]))
    
    # LDOS is constant 2.0
    ldos = np.array([2.0, 2.0, 2.0])
    
    # Window covers the line
    coupling = effective_coupling(ldos, patch, freq_window=(0.5, 1.5))
    
    # Power = 1.0. Avg LDOS = 2.0. Result = 2.0
    assert coupling == 2.0
    
    # Window does NOT cover the line
    coupling_zero = effective_coupling(ldos, patch, freq_window=(2.0, 3.0))
    assert coupling_zero == 0.0
