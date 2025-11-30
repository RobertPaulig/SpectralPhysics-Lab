import numpy as np
from dataclasses import dataclass
from .medium_2d import OscillatorGrid2D

@dataclass
class NDTProfile:
    """
    Profile of a 'healthy' material state for NDT.
    Stores statistical properties of the LDOS map.
    """
    freq_window: tuple[float, float]
    ldos_mean: np.ndarray   # Mean LDOS map (ny, nx)
    ldos_std: np.ndarray    # Std dev of LDOS map (ny, nx)

def build_ndt_profile(
    grid: OscillatorGrid2D,
    n_modes: int,
    freq_window: tuple[float, float],
    n_samples: int = 1,
    noise_level: float = 0.0
) -> NDTProfile:
    """
    Build an NDT profile by sampling the grid's LDOS.
    
    Args:
        grid: The OscillatorGrid2D instance (healthy state).
        n_modes: Number of modes to calculate for LDOS.
        freq_window: Frequency window (min, max) for LDOS.
        n_samples: Number of samples to average (useful if adding noise).
        noise_level: Amplitude of random mass noise to add for robustness.
        
    Returns:
        NDTProfile containing mean and std of LDOS.
    """
    ldos_maps = []
    
    base_mass = grid.m
    # If grid has mass_map, use it as base
    if grid.mass_map is not None:
        base_mass_map = grid.mass_map.copy()
    else:
        base_mass_map = np.full((grid.ny, grid.nx), base_mass)
        
    for _ in range(n_samples):
        # Perturb mass slightly if noise requested
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=base_mass_map.shape)
            current_mass = base_mass_map + noise
            # Ensure mass stays positive
            current_mass = np.maximum(current_mass, 1e-3)
            
            # Create temp grid with perturbed mass
            temp_grid = OscillatorGrid2D(
                nx=grid.nx, ny=grid.ny,
                kx=grid.kx, ky=grid.ky,
                m=grid.m, # Base scalar m, but we override with map
                mass_map=current_mass,
                kx_map=grid.kx_map,
                ky_map=grid.ky_map
            )
            ldos = temp_grid.ldos_map(n_modes=n_modes, freq_window=freq_window)
        else:
            # No noise, just calc once (or n_samples times same result)
            ldos = grid.ldos_map(n_modes=n_modes, freq_window=freq_window)
            
        ldos_maps.append(ldos)
        
    ldos_stack = np.array(ldos_maps)
    
    mean_ldos = np.mean(ldos_stack, axis=0)
    if n_samples > 1 and noise_level > 0:
        std_ldos = np.std(ldos_stack, axis=0)
    else:
        # If single sample, std is undefined/zero. 
        std_ldos = np.zeros_like(mean_ldos)
        
    return NDTProfile(
        freq_window=freq_window,
        ldos_mean=mean_ldos,
        ldos_std=std_ldos
    )

def score_ndt_state(
    profile: NDTProfile,
    ldos_current: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compare current LDOS with profile.
    Returns a 'defect score' map.
    
    If profile has valid std, use Z-score: |x - mean| / (std + eps).
    Otherwise, use absolute difference: |x - mean|.
    """
    diff = np.abs(ldos_current - profile.ldos_mean)
    
    # Check if we have valid std (non-zero max)
    if np.max(profile.ldos_std) > epsilon:
        # Z-score like metric
        score = diff / (profile.ldos_std + epsilon)
    else:
        # Just difference
        score = diff
        
    return score

def ndt_defect_mask(
    scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Return boolean mask where score > threshold.
    """
    return scores > threshold
