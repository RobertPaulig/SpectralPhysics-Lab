"""
GeoSpectra 2D: Foundation Health Monitoring

This script demonstrates monitoring a building foundation:
1. Baseline: Healthy soil under foundation.
2. Current: "Washout" (erosion) under one side.
3. Compare surface response to detect the issue.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.geophysics_2d import GeoGrid2D

def create_foundation_model(washout=False):
    nx, ny = 40, 20
    depth_scale = 0.5
    
    # Base soil
    stiffness = np.full((ny, nx), 5.0)
    density = np.full((ny, nx), 2.0)
    
    # Foundation Block (Concrete) on surface (top 2 rows, center)
    # High stiffness, high density
    fx_start, fx_end = 15, 25
    fy_start = ny - 2
    
    stiffness[fy_start:, fx_start:fx_end] = 20.0
    density[fy_start:, fx_start:fx_end] = 5.0
    
    if washout:
        # Erosion under the right side of foundation
        # Low stiffness (water/air mix)
        wx_start, wx_end = 20, 25
        wy_start, wy_end = fy_start - 3, fy_start
        
        stiffness[wy_start:wy_end, wx_start:wx_end] = 1.0
        density[wy_start:wy_end, wx_start:wx_end] = 1.0
        
    return GeoGrid2D(nx, ny, depth_scale, stiffness, density)

def main():
    print("=" * 60)
    print("GeoSpectra 2D: Foundation Health Monitor")
    print("=" * 60)
    
    # 1. Simulate Baseline
    print("1. Simulating Baseline (Healthy Foundation)...")
    geo_base = create_foundation_model(washout=False)
    resp_base = geo_base.forward_response(freq_window=(0.0, 2.0), n_modes=50)
    
    # 2. Simulate Washout
    print("2. Simulating Current State (Washout/Erosion)...")
    geo_curr = create_foundation_model(washout=True)
    resp_curr = geo_curr.forward_response(freq_window=(0.0, 2.0), n_modes=50)
    
    # 3. Compare
    diff = np.abs(resp_curr - resp_base)
    max_diff_idx = np.argmax(diff)
    print(f"   Max difference at x={max_diff_idx}")
    
    # 4. Visualize
    print("3. Visualizing...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Maps
    im0 = axes[0, 0].imshow(geo_base.stiffness_map, origin='lower', cmap='cividis')
    axes[0, 0].set_title("Baseline Stiffness")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(geo_curr.stiffness_map, origin='lower', cmap='cividis')
    axes[0, 1].set_title("Washout Stiffness (Note low val under right)")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Responses
    ax2 = axes[1, 0]
    ax2.plot(resp_base, 'b-', label='Baseline')
    ax2.plot(resp_curr, 'r--', label='Current (Washout)')
    ax2.set_title("Surface Response (LDOS)")
    ax2.legend()
    ax2.grid(True)
    
    # Difference
    ax3 = axes[1, 1]
    ax3.plot(diff, 'k-', linewidth=2)
    ax3.fill_between(range(len(diff)), diff, color='orange', alpha=0.3)
    ax3.set_title("Difference Signal (Anomaly)")
    ax3.set_xlabel("Position (x)")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    if np.max(diff) > 0.01:
        print("ALERT: Significant foundation anomaly detected!")
    else:
        print("Status: Stable.")
    print("=" * 60)

if __name__ == "__main__":
    main()
