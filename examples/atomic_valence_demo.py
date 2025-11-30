"""
Atomic Valence as Surface Resonances

This demo explores the idea that chemical valence can be modeled as "spectral compatibility" 
between resonators. We use toy models of atoms (H, O, C) defined by their resonance frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.atoms import H, O, C, spectral_overlap, can_form_bond

def plot_atom_spectra(atom1, atom2):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot lines for Atom 1
    ax.vlines(atom1.omega, 0, atom1.power, colors='blue', label=f'{atom1.name} lines', lw=2)
    
    # Plot lines for Atom 2
    ax.vlines(atom2.omega, 0, atom2.power, colors='red', label=f'{atom2.name} lines', lw=2, linestyle='--')
    
    ax.set_title(f"Spectral Comparison: {atom1.name} vs {atom2.name}")
    ax.set_xlabel("Frequency (arbitrary units)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def main():
    print("=" * 60)
    print("Atomic Valence as Surface Resonances Demo")
    print("=" * 60)
    
    # 1. Hydrogen vs Oxygen (Water)
    print("\n1. Hydrogen vs Oxygen (Water)")
    print("-" * 40)
    print("Oxygen has multiple lines around 1.0, matching Hydrogen's single line at 1.0.")
    
    fig1 = plot_atom_spectra(H, O)
    
    overlap = spectral_overlap(H, O)
    print(f"Spectral Overlap (H-O): {overlap:.3f}")
    print(f"Can form bond? {can_form_bond(H, O, freq_tol=0.1, threshold=0.1)}")
    print(f"O max bonds: {O.max_bonds} -> Can hold two H atoms (H-O-H)")
    
    plt.show()
    
    # 2. Carbon vs Hydrogen (Methane)
    print("\n2. Carbon vs Hydrogen (Methane)")
    print("-" * 40)
    print("Carbon has a broader spectrum, overlapping well with Hydrogen.")
    
    fig2 = plot_atom_spectra(C, H)
    
    overlap = spectral_overlap(C, H)
    print(f"Spectral Overlap (C-H): {overlap:.3f}")
    print(f"C max bonds: {C.max_bonds} -> Can hold four H atoms (CH4)")
    
    plt.show()
    
    # 3. Carbon vs Oxygen (CO2)
    print("\n3. Carbon vs Oxygen (CO2)")
    print("-" * 40)
    print("Carbon and Oxygen also overlap significantly.")
    
    fig3 = plot_atom_spectra(C, O)
    
    overlap = spectral_overlap(C, O)
    print(f"Spectral Overlap (C-O): {overlap:.3f}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Conclusion: Valence emerges from spectral resonance compatibility!")
    print("=" * 60)

if __name__ == "__main__":
    main()
