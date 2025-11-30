import numpy as np
import pytest
from spectral_physics.geophysics_1d import (
    Layer, LayeredMedium1D, simulate_pulse_response, invert_single_layer_thickness
)

def test_layered_medium_creation():
    l1 = Layer(thickness=10.0, density=1.0, stiffness=1.0)
    l2 = Layer(thickness=10.0, density=2.0, stiffness=4.0)
    medium = LayeredMedium1D(layers=[l1, l2], dx=1.0)
    
    chain = medium.to_oscillator_chain()
    
    # Total depth 20, dx=1 -> 20 nodes
    assert chain.n == 20
    
    # First 10 nodes (0..9) should have mass ~ 1.0 * 1.0 = 1.0
    assert np.allclose(chain.m[:10], 1.0)
    
    # Next 10 nodes (10..19) should have mass ~ 2.0 * 1.0 = 2.0
    assert np.allclose(chain.m[10:], 2.0)

def test_simulation_runs():
    l1 = Layer(thickness=5.0, density=1.0, stiffness=1.0)
    medium = LayeredMedium1D(layers=[l1], dx=1.0)
    
    t, sig = simulate_pulse_response(medium, t_max=10.0, dt=0.1)
    
    assert len(t) == 100
    assert len(sig) == 100
    # Signal should not be all zeros (pulse happened)
    assert np.max(np.abs(sig)) > 0

def test_inversion_sanity():
    # Create target
    true_h = 5.0
    l1 = Layer(thickness=true_h, density=1.0, stiffness=1.0)
    l_sub = Layer(thickness=10.0, density=10.0, stiffness=10.0) # Hard substrate
    
    medium = LayeredMedium1D(layers=[l1, l_sub], dx=1.0)
    t_max = 20.0
    dt = 0.1
    t, target_sig = simulate_pulse_response(medium, t_max, dt)
    
    # Invert
    # Start guess 4.0
    h_found = invert_single_layer_thickness(
        target_sig, t, 
        density=1.0, stiffness=1.0, 
        thickness_guess=4.0,
        fixed_layers_below=[l_sub],
        dx=1.0
    )
    
    # Should be close to 5.0
    # Note: dx=1.0 makes resolution limited. 5.0 is exactly 5 nodes.
    # 4.0 is 4 nodes.
    # The optimizer should find 5.0 (or close to it if continuous interpretation)
    assert abs(h_found - true_h) < 1.5 # Tolerance due to dx discretization
