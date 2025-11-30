import numpy as np
from dataclasses import dataclass


@dataclass
class Spectrum1D:
    """
    One-dimensional discrete spectrum.
    
    Attributes:
        omega: Angular frequencies (1D array).
        power: Spectral power/energy density at each frequency (1D array).
    """
    omega: np.ndarray
    power: np.ndarray
    
    def __post_init__(self):
        """Validate that omega and power have the same shape."""
        self.omega = np.asarray(self.omega, dtype=float)
        self.power = np.asarray(self.power, dtype=float)
        
        if self.omega.shape != self.power.shape:
            raise ValueError(
                f"Shape mismatch: omega has shape {self.omega.shape}, "
                f"power has shape {self.power.shape}"
            )
    
    def normalize(self) -> "Spectrum1D":
        """
        Return a new Spectrum1D with normalized power (sum(power) = 1).
        
        Returns:
            New Spectrum1D instance with normalized power.
        
        Raises:
            ValueError: If total power is zero.
        """
        total = self.total_power()
        if total == 0:
            raise ValueError("Cannot normalize spectrum with zero total power")
        
        return Spectrum1D(
            omega=self.omega.copy(),
            power=self.power / total
        )
    
    def total_power(self) -> float:
        """
        Compute total power (integral/sum over all frequencies).
        
        Returns:
            Total power as a scalar.
        """
        return float(np.sum(self.power))
    
    def apply_filter(self, alpha: np.ndarray) -> "Spectrum1D":
        """
        Apply frequency-dependent filter/transparency alpha(omega).
        
        Args:
            alpha: Transparency/filter coefficients (same shape as power).
        
        Returns:
            New Spectrum1D with filtered power: new_power = power * alpha.
        
        Raises:
            ValueError: If alpha has incompatible shape.
        """
        alpha = np.asarray(alpha, dtype=float)
        
        if alpha.shape != self.power.shape:
            raise ValueError(
                f"Shape mismatch: alpha has shape {alpha.shape}, "
                f"expected {self.power.shape}"
            )
        
        return Spectrum1D(
            omega=self.omega.copy(),
            power=self.power * alpha
        )
    
    @classmethod
    def from_function(cls, omega: np.ndarray, func):
        """
        Создать Spectrum1D из функции power(omega).
        
        Args:
            omega: Массив частот.
            func: Функция, принимающая omega и возвращающая power.
        
        Returns:
            Новый экземпляр Spectrum1D.
        
        Raises:
            ValueError: Если func(omega) вернула массив несовместимой формы.
        
        Example:
            >>> omega = np.linspace(0, 10, 100)
            >>> spec = Spectrum1D.from_function(omega, lambda w: np.exp(-w))
        """
        omega = np.asarray(omega, dtype=float)
        power = np.asarray(func(omega), dtype=float)
        
        if power.shape != omega.shape:
            raise ValueError(
                f"func(omega) must return array with same shape as omega. "
                f"Got power shape {power.shape}, expected {omega.shape}"
            )
        
        return cls(omega=omega, power=power)

