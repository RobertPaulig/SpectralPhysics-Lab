import numpy as np
from dataclasses import dataclass
from .medium_2d import OscillatorGrid2D

@dataclass
class GeoGrid2D:
    """
    2D Geophysical Model (Cross-section of Earth).
    
    Wraps OscillatorGrid2D to provide geophysical context:
    - stiffness_map represents shear modulus or bulk modulus.
    - density_map represents rock density.
    - depth_scale (meters per pixel).
    """
    nx: int
    ny: int
    depth_scale: float  # meters per grid unit
    stiffness_map: np.ndarray  # (ny, nx)
    density_map: np.ndarray    # (ny, nx)
    
    def __post_init__(self):
        if self.stiffness_map.shape != (self.ny, self.nx):
            raise ValueError("stiffness_map shape mismatch")
        if self.density_map.shape != (self.ny, self.nx):
            raise ValueError("density_map shape mismatch")
            
    def to_oscillator_grid(self) -> OscillatorGrid2D:
        """
        Convert to OscillatorGrid2D for simulation.
        We assume isotropic stiffness (kx = ky = stiffness).
        Mass = density * volume (assuming unit thickness and dx=dy=depth_scale).
        Actually, in the toy model:
        m ~ density
        k ~ stiffness
        """
        # Create grid with maps
        # We use mean values for base kx, ky, m to satisfy init, 
        # but maps will override.
        
        return OscillatorGrid2D(
            nx=self.nx,
            ny=self.ny,
            kx=1.0, # Placeholder
            ky=1.0, # Placeholder
            m=1.0,  # Placeholder
            mass_map=self.density_map,
            kx_map=self.stiffness_map,
            ky_map=self.stiffness_map
        )

    def forward_response(
        self,
        freq_window: tuple[float, float],
        n_modes: int = 50
    ) -> np.ndarray:
        """
        Calculate surface response (LDOS at top row).
        
        Returns:
            1D array of LDOS values along the surface (x-axis).
        """
        grid = self.to_oscillator_grid()
        
        # Calculate full LDOS map
        ldos_map = grid.ldos_map(n_modes=n_modes, freq_window=freq_window)
        
        # Return only the top row (surface)
        # Assuming y=0 is bottom and y=ny-1 is top? 
        # In imshow origin='lower', index 0 is bottom.
        # Let's assume index -1 (last row) is surface.
        return ldos_map[-1, :]


def invert_stiffness(
    surface_ldos: np.ndarray,
    prior_model: GeoGrid2D,
    iterations: int = 10
) -> np.ndarray:
    """
    Toy inversion: try to adjust stiffness map to match surface LDOS.
    
    This is a placeholder for a real inversion. 
    Real inversion would require gradients.
    
    Here we just return a 'reconstructed' map that is a smoothed version 
    of the prior, maybe slightly adjusted by surface residuals?
    
    For the demo, we might just return the prior's stiffness map 
    or a simple modification to show "we tried".
    """
    # Placeholder logic
    reconstructed = prior_model.stiffness_map.copy()
    
    # In a real scenario, we would iterate:
    # 1. Forward(model) -> predicted_surface
    # 2. Diff = predicted - observed
    # 3. Update model to minimize Diff
    
    return reconstructed
