import pytest
import numpy as np
from spectral_physics.atoms import (
    AtomicResonator, 
    spectral_overlap, 
    can_form_bond,
    H, O, C
)

def test_atomic_resonator_creation():
    atom = AtomicResonator.from_lines("Test", [(10.0, 1.0), (20.0, 0.5)], max_bonds=3)
    assert len(atom.omega) == 2
    assert atom.max_bonds == 3
    assert atom.name == "Test"
    
    spec = atom.spectrum()
    assert np.allclose(spec.omega, [10.0, 20.0])

def test_spectral_overlap_identical():
    # H vs H should have overlap
    # H has line at 1.0
    score = spectral_overlap(H, H, freq_tol=0.1)
    # Both normalized power is 1.0. Min(1,1) = 1.
    assert score > 0.9

def test_spectral_overlap_none():
    # Atom with far frequency
    X = AtomicResonator.from_lines("X", [(100.0, 1.0)], max_bonds=1)
    
    score = spectral_overlap(H, X, freq_tol=0.1)
    assert score == 0.0

def test_can_form_bond_logic():
    # H and O should bond (H=1.0, O has 1.0)
    assert can_form_bond(H, O, freq_tol=0.1, threshold=0.1)
    
    # Inert gas (He)
    He = AtomicResonator.from_lines("He", [(1.0, 1.0)], max_bonds=0)
    
    # Even if frequencies match, max_bonds=0 prevents bonding
    assert not can_form_bond(He, H, freq_tol=0.1, threshold=0.1)

def test_carbon_bonds():
    # C should bond with H
    assert can_form_bond(C, H, freq_tol=0.1, threshold=0.1)
    # C should bond with O
    assert can_form_bond(C, O, freq_tol=0.1, threshold=0.1)
