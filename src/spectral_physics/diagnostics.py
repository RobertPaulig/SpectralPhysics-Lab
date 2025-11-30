import numpy as np
from dataclasses import dataclass
from .timeseries import timeseries_to_spectrum
from .spectrum import Spectrum1D
from .material import MaterialSignature


@dataclass
class ChannelConfig:
    """
    Конфигурация одного канала измерений.

    Attributes:
        name: Имя канала (строка для отчётов).
        dt: Шаг по времени (секунды).
        window: Тип окна для FFT ("hann" или None).
        freq_min: Минимальная частота анализа (Гц).
        freq_max: Максимальная частота анализа (Гц).
    """
    name: str
    dt: float
    window: str = "hann"
    freq_min: float | None = None
    freq_max: float | None = None


class SpectralAnalyzer:
    """
    Инструмент: временной ряд -> спектр -> обрезка по диапазону частот.
    """

    def __init__(self, config: ChannelConfig):
        """
        Args:
            config: Конфигурация канала.
        """
        self.config = config

    def analyze(self, signal: np.ndarray) -> Spectrum1D:
        """
        Преобразовать сигнал во времени в Spectrum1D с учётом:
        - удаления DC,
        - окна,
        - вырезания диапазона частот [freq_min, freq_max] (если заданы).

        Args:
            signal: Временной ряд (1D array).

        Returns:
            Spectrum1D только по выбранному диапазону частот.
        """
        # Преобразование в спектр
        spectrum = timeseries_to_spectrum(
            signal,
            dt=self.config.dt,
            window=self.config.window
        )
        
        # Если не заданы ограничения частот, вернуть весь спектр
        if self.config.freq_min is None and self.config.freq_max is None:
            return spectrum
        
        # Преобразовать omega в Герцы
        freq_hz = spectrum.omega / (2 * np.pi)
        
        # Определить маску для диапазона частот
        mask = np.ones(len(freq_hz), dtype=bool)
        
        if self.config.freq_min is not None:
            mask &= (freq_hz >= self.config.freq_min)
        
        if self.config.freq_max is not None:
            mask &= (freq_hz <= self.config.freq_max)
        
        # Вырезать диапазон
        omega_filtered = spectrum.omega[mask]
        power_filtered = spectrum.power[mask]
        
        return Spectrum1D(omega=omega_filtered, power=power_filtered)


class HealthMonitor:
    """
    Монитор "здоровья" канала по спектру.
    """

    def __init__(self, reference: Spectrum1D, threshold: float):
        """
        Args:
            reference: Эталонный спектр "здорового" состояния.
            threshold: Порог аномальности по L2-дистанции.
        """
        self.signature = MaterialSignature(reference=reference)
        self.threshold = float(threshold)

    def score(self, current: Spectrum1D) -> float:
        """
        Вернуть L2-дистанцию между текущим спектром и эталонным.
        
        Args:
            current: Текущий спектр.
        
        Returns:
            L2-расстояние (скаляр).
        """
        return self.signature.distance_l2(current)

    def is_anomalous(self, current: Spectrum1D) -> bool:
        """
        True, если текущий спектр аномален (distance > threshold).
        
        Args:
            current: Текущий спектр.
        
        Returns:
            True если аномален, False иначе.
        """
        return self.signature.is_anomalous(current, self.threshold)
