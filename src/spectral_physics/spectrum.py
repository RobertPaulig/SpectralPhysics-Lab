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

    def copy(self):
        """Возвращает глубокую копию спектра."""
        return Spectrum1D(
            freqs=self.freqs.copy(),
            amps=self.amps.copy(),
            phases=self.phases.copy()
        )
