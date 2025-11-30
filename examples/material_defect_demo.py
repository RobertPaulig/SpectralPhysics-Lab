"""
Material Defect Demo: LDOS Anomalies

This script demonstrates how a local defect (change in mass or stiffness) affects 
the Local Spectral Density (LDOS) of a 2D medium.
We compare a "pristine" plate with a "defective" one.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.medium_2d import OscillatorGrid2D

def main():
    print("=" * 60)
    print("Material Defect Demo: LDOS Anomalies")
    print("=" * 60)
    
    # 1. Setup Pristine Grid
    nx, ny = 20, 20
    grid_clean = OscillatorGrid2D(nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0)
    
    # Calculate LDOS map in a low-frequency window
    freq_window = (0.0, 1.5)
    n_modes = 50  # Calculate enough modes to cover the window
    
    print("\nCalculating LDOS for pristine grid...")
    ldos_clean = grid_clean.ldos_map(n_modes=n_modes, freq_window=freq_window)
    
    # 2. Setup Defective Grid
    # Defect: Heavy mass in the center (3x3 block)
    mass_map = np.ones((ny, nx))
    cx, cy = nx // 2, ny // 2
    mass_map[cy-1:cy+2, cx-1:cx+2] = 5.0  # 5x heavier
    
    grid_defect = OscillatorGrid2D(nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0, mass_map=mass_map)
    
    print("Calculating LDOS for defective grid...")
    ldos_defect = grid_defect.ldos_map(n_modes=n_modes, freq_window=freq_window)
    
    # 3. Visualize Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(ldos_clean.min(), ldos_defect.min())
    vmax = max(ldos_clean.max(), ldos_defect.max())
    
    im0 = axes[0].imshow(ldos_clean, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title("Pristine LDOS")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(ldos_defect, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title("Defective LDOS (Heavy Center)")
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    diff = ldos_defect - ldos_clean
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r')
    axes[2].set_title("Difference (Defect - Clean)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("The heavy defect changes the local spectral density.")
    print("In the low-frequency window, the heavy mass might accumulate")
    print("more energy (or less, depending on the modes shifted).")
    print("The difference map clearly highlights the location of the anomaly.")
    print("=" * 60)

if __name__ == "__main__":
    main()
