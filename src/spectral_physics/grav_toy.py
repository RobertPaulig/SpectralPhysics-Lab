from .medium_1d import OscillatorChain1D
from .spectrum import Spectrum1D
import numpy as np


def compute_local_spectrum(chain: OscillatorChain1D, idx_range: slice) -> Spectrum1D:
    """
    Record time series of oscillations for a subset of nodes (idx_range),
    then build Spectrum1D for the averaged signal in this region.
    
    Note: This function expects the chain to have a recorded history.
    For actual use, you need to run simulation first and store time series.
    This is a helper that would typically be called after running the simulation.
    """
    # This is a placeholder implementation
    # In real usage, you would:
    # 1. Run chain.step(dt) many times, recording x values
    # 2. Average x values over idx_range at each time step
    # 3. Build spectrum from that time series
    
    # For now, return a dummy spectrum
    # A real implementation would need the time series data
    raise NotImplementedError(
        "compute_local_spectrum requires time series data. "
        "Run simulation first and pass recorded data."
    )


def spectral_pressure_difference(
    chain: OscillatorChain1D,
    left_range: slice,
    right_range: slice,
    *,
    t_total: float,
    dt: float,
    band: tuple[float, float] | None = None,
) -> float:
    """
    Compute toy-model 'force' as difference of spectral power between left and right regions.

    Steps:
        1. Run simulation of chain for time t_total with step dt.
        2. Build spectra for left_range and right_range via compute_local_spectrum.
        3. Compute power (or band power) for both sides.
        4. Return P_left - P_right as effective spectral pressure difference.
        
    Args:
        chain: OscillatorChain1D instance (initial state should be set before calling)
        left_range: slice defining indices for left region
        right_range: slice defining indices for right region
        t_total: total simulation time
        dt: time step for integration
        band: optional tuple (omega_min, omega_max) to compute band power instead of total power
        
    Returns:
        Spectral pressure difference (P_left - P_right)
    """
    n_steps = int(t_total / dt)
    
    # Record time series for left and right regions
    time_points = []
    left_signals = []
    right_signals = []
    
    t = 0.0
    for _ in range(n_steps):
        time_points.append(t)
        
        # Average position in left region
        left_x = np.mean(chain.x[left_range])
        left_signals.append(left_x)
        
        # Average position in right region
        right_x = np.mean(chain.x[right_range])
        right_signals.append(right_x)
        
        # Step forward
        chain.step(dt)
        t += dt
    
    # Convert to numpy arrays
    t_array = np.array(time_points)
    left_array = np.array(left_signals)
    right_array = np.array(right_signals)
    
    # Build spectra
    spec_left = Spectrum1D.from_time_signal(t_array, left_array)
    spec_right = Spectrum1D.from_time_signal(t_array, right_array)
    
    # Compute power
    if band is not None:
        omega_min, omega_max = band
        p_left = spec_left.band_power(omega_min, omega_max)
        p_right = spec_right.band_power(omega_min, omega_max)
    else:
        p_left = spec_left.total_power()
        p_right = spec_right.total_power()
    
    return float(p_left - p_right)
