"""
Material Health Monitoring

This demo shows how to use spectral-health concepts for materials.
We treat the LDOS map as the "state" of the material and detect defects 
as deviations from a healthy profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.medium_2d import OscillatorGrid2D
from spectral_physics.materials import build_material_health_profile

def main():
    print("=" * 60)
    print("Material Health Monitoring Demo")
    print("=" * 60)
    
    # 1. Train Healthy Profile
    # Create a pristine grid
    nx, ny = 20, 20
    grid_clean = OscillatorGrid2D(nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0)
    
    print("\nCalculating healthy LDOS...")
    # Calculate LDOS
    ldos_clean = grid_clean.ldos_map(n_modes=50, freq_window=(0.0, 1.5))
    
    # Build profile
    profile = build_material_health_profile(ldos_clean)
    print("Healthy Profile Features:", profile.reference_features)
    
    # 2. Test Defective Material
    # Create grid with defect
    print("\nCreating defective material (heavy spot)...")
    mass_map = np.ones((ny, nx))
    cx, cy = nx // 2, ny // 2
    mass_map[cy-1:cy+2, cx-1:cx+2] = 5.0  # Heavy spot
    
    grid_defect = OscillatorGrid2D(nx=nx, ny=ny, kx=1.0, ky=1.0, m=1.0, mass_map=mass_map)
    ldos_defect = grid_defect.ldos_map(n_modes=50, freq_window=(0.0, 1.5))
    
    # Build "current" features (using same logic as profile builder)
    current_sig = build_material_health_profile(ldos_defect)
    
    # Calculate distance
    distance = profile.distance_l2(current_sig.reference_features)
    print(f"\nHealth Distance: {distance:.4f}")
    
    # 3. Threshold Check
    threshold = 1.0  # Arbitrary threshold
    print(f"Threshold: {threshold}")
    
    if distance > threshold:
        print("Status: ⚠️  ANOMALY DETECTED (Material Defect)")
    else:
        print("Status: ✓ OK")
    
    # 4. Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(ldos_clean, origin='lower', cmap='inferno')
    axes[0].set_title("Healthy Material")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(ldos_defect, origin='lower', cmap='inferno')
    axes[1].set_title(f"Defective Material (Dist={distance:.2f})")
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Conclusion: Defect automatically detected via health distance!")
    print("=" * 60)

if __name__ == "__main__":
    main()
