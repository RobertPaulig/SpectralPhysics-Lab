import numpy as np

class Spectrum1D:
    """
    Дискретный 1D-спектр: частоты, амплитуды, фазы.

    Используется как базовый кирпич для работы со спектром колебаний Среды.
    """

    def __init__(self, freqs, amps, phases=None):
        """
        Создает объект спектра.
        
        Параметры:
            freqs: массив частот (1D)
            amps: массив амплитуд (1D)
            phases: массив фаз (1D), опционально (по умолчанию нули)
        """
        self.freqs = np.asarray(freqs, dtype=float)
        self.amps = np.asarray(amps, dtype=float)
        
        if self.freqs.shape != self.amps.shape:
            raise ValueError(f"Shape mismatch: freqs {self.freqs.shape} vs amps {self.amps.shape}")
            
        if phases is not None:
            self.phases = np.asarray(phases, dtype=float)
            if self.phases.shape != self.freqs.shape:
                raise ValueError(f"Shape mismatch: freqs {self.freqs.shape} vs phases {self.phases.shape}")
        else:
            self.phases = np.zeros_like(self.freqs)

    def power(self):
        """Возвращает массив мощности (amps^2)."""
        return self.amps ** 2

    def total_power(self):
        """Возвращает суммарную мощность (скаляр)."""
        return np.sum(self.power())

    def normalize_power(self, target=1.0):
        """
        Нормирует амплитуды так, чтобы total_power() стала равна target.
        Изменяет текущий объект in-place.
        """
        current_p = self.total_power()
        if current_p == 0:
            if target == 0:
                return # Already 0
            else:
                raise ValueError("Cannot normalize zero power spectrum to non-zero target")
        
        # We need sum((k * amps)^2) = target
        # k^2 * current_p = target
        # k = sqrt(target / current_p)
        scale = np.sqrt(target / current_p)
        self.amps *= scale

    @classmethod
    def from_time_signal(cls, t: np.ndarray, x: np.ndarray) -> "Spectrum1D":
        """
        Build spectrum from real-valued time signal x(t) using FFT.

        Args:
            t: time grid (1D array, uniform step).
            x: signal values on grid t.

        Returns:
            Spectrum1D instance with positive frequencies only.
        """
        if len(t) != len(x):
            raise ValueError(f"Length mismatch: t has {len(t)} points, x has {len(x)}")
        
        n = len(x)
        # Compute time step (assuming uniform grid)
        dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 1.0
        
        # FFT
        fft_vals = np.fft.fft(x)
        fft_freqs = np.fft.fftfreq(n, d=dt)
        
        # Take only positive frequencies (including zero)
        # For real signals, negative frequencies are redundant
        positive_mask = fft_freqs >= 0
        omega = 2 * np.pi * fft_freqs[positive_mask]  # Convert to angular frequency
        complex_amps = fft_vals[positive_mask]
        
        # Extract amplitude and phase
        amps = np.abs(complex_amps) / n  # Normalize by n
        phases = np.angle(complex_amps)
        
        return cls(freqs=omega, amps=amps, phases=phases)

    def band_power(self, omega_min: float, omega_max: float) -> float:
        """
        Integrate power over the band [omega_min, omega_max].

        Returns:
            Scalar total power in the given band.
        """
        mask = (self.freqs >= omega_min) & (self.freqs <= omega_max)
        return float(np.sum(self.power()[mask]))

    def copy(self):
        """Возвращает глубокую копию спектра."""
        return Spectrum1D(
            freqs=self.freqs.copy(),
            amps=self.amps.copy(),
            phases=self.phases.copy()
        )
