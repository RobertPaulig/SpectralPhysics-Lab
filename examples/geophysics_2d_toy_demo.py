"""
GeoSpectra 2D: Toy Demo (Cavity Detection)

This script demonstrates 2D geophysical modeling:
1. Create a ground model with a hidden cavity.
2. Simulate surface response (LDOS).
3. Visualize the "hidden" structure and the surface signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.geophysics_2d import GeoGrid2D

def main():
    print("=" * 60)
    print("GeoSpectra 2D: Cavity Detection Demo")
    print("=" * 60)
    
    # 1. Define Ground Model
    nx, ny = 40, 30
    depth_scale = 1.0 # meters per pixel
    
    print(f"1. Creating Ground Model ({nx}x{ny}, {depth_scale}m/px)...")
    
    # Background: Stiff rock
    stiffness_map = np.full((ny, nx), 10.0)
    density_map = np.full((ny, nx), 3.0)
    
    # Top layer: Softer soil (top 5 rows)
    # Note: y=0 is bottom, y=ny-1 is top.
    top_layer_height = 5
    stiffness_map[-top_layer_height:, :] = 2.0
    density_map[-top_layer_height:, :] = 1.5
    
    # Cavity: Hidden void deep underground
    # Low stiffness, low density (air/water)
    cx, cy = 20, 10
    r = 4
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    
    stiffness_map[mask] = 0.1
    density_map[mask] = 0.1
    
    geo_grid = GeoGrid2D(
        nx=nx, ny=ny, depth_scale=depth_scale,
        stiffness_map=stiffness_map,
        density_map=density_map
    )
    
    # 2. Simulate Surface Response
    print("2. Simulating Surface Response...")
    freq_window = (0.0, 2.0)
    n_modes = 60
    
    # Get full LDOS for visualization
    grid = geo_grid.to_oscillator_grid()
    ldos_map = grid.ldos_map(n_modes=n_modes, freq_window=freq_window)
    
    # Extract surface signal (top row)
    surface_signal = ldos_map[-1, :]
    
    # 3. Visualize
    print("3. Visualizing...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stiffness Map (Ground Truth)
    im0 = axes[0, 0].imshow(stiffness_map, origin='lower', cmap='cividis')
    axes[0, 0].set_title("Ground Stiffness (True Model)")
    plt.colorbar(im0, ax=axes[0, 0], label='Stiffness')
    
    # Density Map
    im1 = axes[0, 1].imshow(density_map, origin='lower', cmap='bone')
    axes[0, 1].set_title("Ground Density (True Model)")
    plt.colorbar(im1, ax=axes[0, 1], label='Density')
    
    # Full LDOS Map (What's happening inside)
    im2 = axes[1, 0].imshow(ldos_map, origin='lower', cmap='inferno')
    axes[1, 0].set_title("Internal Vibration Intensity (LDOS)")
    plt.colorbar(im2, ax=axes[1, 0], label='LDOS')
    
    # Surface Signal (What we measure)
    ax3 = axes[1, 1]
    ax3.plot(surface_signal, 'r-o', linewidth=2)
    ax3.set_title("Surface Measurement (LDOS at Top)")
    ax3.set_xlabel("Position (x)")
    ax3.set_ylabel("Response Intensity")
    ax3.grid(True)
    
    # Highlight cavity position on surface plot
    ax3.axvspan(cx - r, cx + r, color='yellow', alpha=0.3, label='Cavity X-range')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Demo Complete.")
    print("Notice the dip/peak in surface response above the cavity?")
    print("=" * 60)

if __name__ == "__main__":
    main()
