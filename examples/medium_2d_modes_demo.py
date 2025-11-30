"""
2D Medium Modes Demo

This script visualizes the eigenmodes (standing waves) of a 2D oscillator grid (elastic plate).
We use OscillatorGrid2D to calculate the modes and matplotlib to display them.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.medium_2d import OscillatorGrid2D

def main():
    # 1. Create a 2D Grid
    nx, ny = 20, 20
    grid = OscillatorGrid2D(nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0)
    
    print(f"Grid size: {nx}x{ny} = {nx*ny} oscillators")
    
    # 2. Calculate Eigenmodes
    n_modes = 9
    print(f"Calculating first {n_modes} modes...")
    omega, modes = grid.eigenmodes(n_modes=n_modes)
    
    print("Frequencies:", omega)
    
    # 3. Visualize Modes
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_modes):
        ax = axes[i]
        
        # Extract mode vector and reshape to 2D grid
        mode_vec = modes[:, i]
        mode_grid = mode_vec.reshape((ny, nx))
        
        # Plot
        im = ax.imshow(mode_grid, cmap='RdBu', origin='lower', interpolation='bicubic')
        ax.set_title(f"Mode {i+1}: $\\omega$ = {omega[i]:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.suptitle("2D Medium Eigenmodes", y=1.001)
    plt.show()
    
    print("\nThese patterns represent the fundamental resonances of the medium.")
    print("In a 'spectral physics' context, these are the available 'slots' for energy to occupy.")

if __name__ == "__main__":
    main()
