"""
1D Geophysics: Layered Earth & Inversion

This demo simulates a seismic pulse propagating through a layered medium.
We then solve a simple inverse problem: determining the thickness of the top layer 
from the surface response.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.geophysics_1d import (
    Layer, LayeredMedium1D, simulate_pulse_response, invert_single_layer_thickness
)

def main():
    print("=" * 60)
    print("1D Geophysics: Layered Earth & Inversion")
    print("=" * 60)
    
    # 1. Setup "True" Earth Model
    # Layer 1: Soft soil (low density/stiffness), unknown thickness
    true_h = 15.0
    layer1 = Layer(thickness=true_h, density=1.5, stiffness=2.0)
    
    # Layer 2: Hard rock (high density/stiffness), known properties
    layer2 = Layer(thickness=50.0, density=3.0, stiffness=10.0)
    
    medium_true = LayeredMedium1D(layers=[layer1, layer2], dx=0.5)
    
    print(f"\nTrue Model: Top layer thickness = {true_h}")
    
    # 2. Simulate Surface Response (The "Measurement")
    t_max = 50.0
    dt = 0.1
    
    print("\nSimulating seismic response...")
    t, signal_true = simulate_pulse_response(medium_true, t_max=t_max, dt=dt)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal_true, label='Observed Signal')
    plt.title("Seismic Response (Surface Displacement)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # 3. Solve Inverse Problem
    # We know density/stiffness of top layer, but not thickness.
    # We know the substrate (layer2).
    
    guess_h = 10.0
    print(f"\nStarting inversion with guess h = {guess_h}...")
    
    found_h = invert_single_layer_thickness(
        target_signal=signal_true,
        t=t,
        density=layer1.density,
        stiffness=layer1.stiffness,
        thickness_guess=guess_h,
        fixed_layers_below=[layer2],
        dx=0.5
    )
    
    print(f"Inversion Result: h = {found_h:.2f} (True: {true_h})")
    error = abs(found_h - true_h)
    print(f"Error: {error:.2f}")
    
    # 4. Verify Result
    # Simulate with found thickness
    print("\nVerifying result...")
    layer_found = Layer(thickness=found_h, density=layer1.density, stiffness=layer1.stiffness)
    medium_found = LayeredMedium1D(layers=[layer_found, layer2], dx=0.5)
    _, signal_found = simulate_pulse_response(medium_found, t_max=t_max, dt=dt)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal_true, 'k-', alpha=0.5, lw=3, label='Observed')
    plt.plot(t, signal_found, 'r--', label='Model (Inverted)')
    plt.title(f"Model Fit (h_found={found_h:.2f})")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 60)
    print("Conclusion:")
    print("We successfully recovered the thickness of the top layer")
    print("by matching the seismic response.")
    print("This is a toy example of full-waveform inversion (FWI).")
    print("=" * 60)

if __name__ == "__main__":
    main()
