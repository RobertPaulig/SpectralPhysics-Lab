"""
NDT Synthetic Defect Demo

This script demonstrates the Spectral NDT workflow:
1. Train a profile on a healthy grid.
2. Introduce a defect (mass anomaly).
3. Detect the defect using the NDT profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.medium_2d import OscillatorGrid2D
from spectral_physics.ndt import build_ndt_profile, score_ndt_state, ndt_defect_mask

def main():
    print("=" * 60)
    print("Spectral NDT: Synthetic Defect Detection")
    print("=" * 60)
    
    # 1. Setup Healthy Grid
    nx, ny = 30, 30
    print(f"\n1. Creating Healthy Grid ({nx}x{ny})...")
    grid_clean = OscillatorGrid2D(nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0)
    
    # 2. Train NDT Profile
    print("2. Training NDT Profile...")
    # We use a low-frequency window where mass effects are visible
    freq_window = (0.0, 1.5)
    n_modes = 60
    
    # Add some noise to training to make it robust (simulate real measurements)
    profile = build_ndt_profile(
        grid=grid_clean,
        n_modes=n_modes,
        freq_window=freq_window,
        n_samples=5,
        noise_level=0.05
    )
    
    print(f"   Profile built. Mean LDOS range: [{profile.ldos_mean.min():.4f}, {profile.ldos_mean.max():.4f}]")
    
    # 3. Create Defective Grid
    print("\n3. Creating Defective Grid...")
    # Defect: Heavy mass block in the center-right
    mass_map = np.ones((ny, nx))
    
    # Add defect
    cx, cy = 20, 15
    r = 3
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    mass_map[mask] = 5.0  # 5x heavier
    
    grid_defect = OscillatorGrid2D(
        nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0,
        mass_map=mass_map
    )
    
    # Calculate current LDOS
    print("   Calculating LDOS for defective grid...")
    ldos_current = grid_defect.ldos_map(n_modes=n_modes, freq_window=freq_window)
    
    # 4. Score and Detect
    print("\n4. Scoring and Detecting...")
    scores = score_ndt_state(profile, ldos_current)
    
    threshold = 5.0 # Z-score threshold
    defect_mask = ndt_defect_mask(scores, threshold)
    
    n_defects = np.sum(defect_mask)
    print(f"   Defects detected: {n_defects} pixels (Threshold={threshold})")
    
    # 5. Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Healthy Mean
    im0 = axes[0, 0].imshow(profile.ldos_mean, origin='lower', cmap='inferno')
    axes[0, 0].set_title("Healthy Profile (Mean LDOS)")
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Current LDOS
    im1 = axes[0, 1].imshow(ldos_current, origin='lower', cmap='inferno')
    axes[0, 1].set_title("Current LDOS (with Defect)")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Score Map
    im2 = axes[1, 0].imshow(scores, origin='lower', cmap='Reds')
    axes[1, 0].set_title("Defect Scores (Z-score)")
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Defect Mask
    im3 = axes[1, 1].imshow(defect_mask, origin='lower', cmap='gray')
    axes[1, 1].set_title(f"Defect Mask (Threshold={threshold})")
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    if n_defects > 0:
        print("SUCCESS: Defect detected!")
    else:
        print("FAILURE: Defect NOT detected.")
    print("=" * 60)

if __name__ == "__main__":
    main()
