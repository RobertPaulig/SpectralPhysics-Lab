"""
Atomic Valence Table Demo

Visualizes the "Periodic Table of Resonances" for selected atoms.
Shows core modes (stable) and valence modes (active for bonding).
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.atoms import load_default_atom_db

def main():
    print("=" * 60)
    print("Atomic Valence Table: Spectral Visualization")
    print("=" * 60)
    
    db = load_default_atom_db()
    atoms = ['H', 'C', 'O', 'Si', 'Fe']
    
    fig, axes = plt.subplots(len(atoms), 1, figsize=(10, 2 * len(atoms)), sharex=True)
    
    if len(atoms) == 1:
        axes = [axes]
        
    for i, symbol in enumerate(atoms):
        atom = db[symbol]
        ax = axes[i]
        
        # Plot spectrum lines
        spec = atom.spectrum()
        ax.vlines(spec.omega, 0, spec.power, color='black', linewidth=2, label='Resonances')
        
        # Highlight Core Modes
        for (f_min, f_max) in atom.core_modes:
            ax.axvspan(f_min, f_max, color='blue', alpha=0.2, label='Core (Stable)')
            
        # Highlight Valence Modes
        for (f_min, f_max) in atom.valence_modes:
            ax.axvspan(f_min, f_max, color='red', alpha=0.2, label='Valence (Bonding)')
            
        ax.set_ylabel(f"Intensity")
        ax.set_title(f"Atom: {atom.name} (Max Bonds: {atom.max_bonds})")
        ax.grid(True, alpha=0.3)
        
        # Only add legend to first plot to avoid clutter
        if i == 0:
            ax.legend(loc='upper right')
            
    axes[-1].set_xlabel("Frequency (arbitrary units)")
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete.")

if __name__ == "__main__":
    main()
