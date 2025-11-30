"""
Geophysics Health Monitoring

This demo applies health monitoring to geophysical data.
We assume a "healthy" baseline response (e.g., from a stable formation) 
and detect changes (e.g., fluid injection, subsidence).
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.geophysics_1d import (
    Layer, LayeredMedium1D, simulate_pulse_response, build_geo1d_health_profile
)

def main():
    print("=" * 60)
    print("Geophysics Health Monitoring Demo")
    print("=" * 60)
    
    # 1. Baseline (Healthy) Model
    print("\nSetting up baseline (healthy) model...")
    l1 = Layer(thickness=15.0, density=1.5, stiffness=2.0)
    l2 = Layer(thickness=50.0, density=3.0, stiffness=10.0)
    medium_base = LayeredMedium1D(layers=[l1, l2], dx=0.5)
    
    t, sig_base = simulate_pulse_response(medium_base, t_max=50.0, dt=0.1)
    
    profile = build_geo1d_health_profile(sig_base)
    print("Baseline Features:", profile.reference_features)
    
    # 2. Changed Model (Anomaly)
    # Suppose the top layer becomes softer (e.g., water saturation)
    print("\nSimulating anomaly (softened top layer)...")
    l1_soft = Layer(thickness=15.0, density=1.4, stiffness=1.5)  # Stiffness dropped
    medium_anom = LayeredMedium1D(layers=[l1_soft, l2], dx=0.5)
    
    _, sig_anom = simulate_pulse_response(medium_anom, t_max=50.0, dt=0.1)
    
    current_sig = build_geo1d_health_profile(sig_anom)
    
    distance = profile.distance_l2(current_sig.reference_features)
    print(f"\nHealth Distance: {distance:.4f}")
    
    # 3. Threshold check
    threshold = 50.0  # Arbitrary threshold
    print(f"Threshold: {threshold}")
    
    if distance > threshold:
        print("Status: ⚠️  ANOMALY DETECTED (Formation Change)")
    else:
        print("Status: ✓ OK")
    
    # 4. Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(t, sig_base, label='Baseline')
    plt.plot(t, sig_anom, label='Anomaly (Soft Layer)')
    plt.title(f"Seismic Response Change (Dist={distance:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 60)
    print("Conclusion: Formation change detected via health monitoring!")
    print("=" * 60)

if __name__ == "__main__":
    main()
