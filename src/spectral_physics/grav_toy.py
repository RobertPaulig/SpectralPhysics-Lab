import numpy as np
from .spectrum import Spectrum1D


def spectral_pressure_difference(
    spectrum_bg: Spectrum1D,
    alpha_left: np.ndarray,
    alpha_right: np.ndarray,
) -> float:
    """
    Estimate pressure difference of background field on left and right sides.
    
    This is a toy model of "spectral shadow" and attractive force.
    The idea is that matter with different spectral transparency creates
    an imbalance in the background spectrum, leading to a net force.
    
    Args:
        spectrum_bg: Background spectrum (omega, power).
        alpha_left: Spectral transparency on the left side (same shape as power).
        alpha_right: Spectral transparency on the right side (same shape as power).
    
    Returns:
        Pressure difference: ΔP = Σ power(ω) * (alpha_right(ω) - alpha_left(ω))
        
        Positive ΔP means net force pushes to the right.
        Negative ΔP means net force pushes to the left.
    
    Raises:
        ValueError: If alpha arrays have incompatible shapes.
    
    Example:
        If left side has lower transparency (blocks more), then more
        background radiation reaches from the right, creating a net
        force pushing the object leftward (toward the shadow).
    """
    alpha_left = np.asarray(alpha_left, dtype=float)
    alpha_right = np.asarray(alpha_right, dtype=float)
    
    # Validate shapes
    if alpha_left.shape != spectrum_bg.power.shape:
        raise ValueError(
            f"alpha_left shape {alpha_left.shape} doesn't match "
            f"spectrum power shape {spectrum_bg.power.shape}"
        )
    
    if alpha_right.shape != spectrum_bg.power.shape:
        raise ValueError(
            f"alpha_right shape {alpha_right.shape} doesn't match "
            f"spectrum power shape {spectrum_bg.power.shape}"
        )
    
    # Compute pressure difference
    # ΔP = Σ P(ω) * [α_right(ω) - α_left(ω)]
    delta_alpha = alpha_right - alpha_left
    delta_p = np.sum(spectrum_bg.power * delta_alpha)
    
    return float(delta_p)
