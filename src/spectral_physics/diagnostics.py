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


@dataclass
class MultiChannelConfig:
    """
    Конфигурация системы с несколькими каналами.
    """
    channels: dict[str, ChannelConfig]



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


def average_spectrum(spectra: list[Spectrum1D]) -> Spectrum1D:
    """
    Усреднить несколько спектров с одинаковой частотной сеткой.

    Args:
        spectra: Список спектров.

    Returns:
        Новый Spectrum1D с усредненной мощностью.
    
    Raises:
        ValueError: Если список пуст или сетки частот не совпадают.
    """
    if not spectra:
        raise ValueError("Cannot average empty list of spectra")
    
    # Берем первый спектр как эталон сетки
    omega_ref = spectra[0].omega
    n_points = len(omega_ref)
    
    sum_power = np.zeros_like(spectra[0].power)
    
    for i, spec in enumerate(spectra):
        if len(spec.omega) != n_points or not np.allclose(spec.omega, omega_ref):
            raise ValueError(
                f"Spectrum at index {i} has different frequency grid"
            )
        sum_power += spec.power
        
    avg_power = sum_power / len(spectra)
    
    return Spectrum1D(omega=omega_ref.copy(), power=avg_power)


def build_health_profile(
    training_data: dict[str, list[Spectrum1D]]
) -> "HealthProfile":
    """
    Построить HealthProfile по обучающим данным.

    Args:
        training_data: Словарь {имя_канала: список_спектров}.

    Returns:
        HealthProfile с усредненными сигнатурами.
    """
    from .material import HealthProfile, MaterialSignature
    
    signatures = {}
    
    for channel_name, spectra_list in training_data.items():
        if not spectra_list:
            continue
            
        # 1. Усредняем спектры
        avg_spec = average_spectrum(spectra_list)
        
        # 2. Создаем сигнатуру
        signature = MaterialSignature(reference=avg_spec)
        
        # 3. Сохраняем
        signatures[channel_name] = signature
        
    return HealthProfile(signatures=signatures)


def spectral_band_power(
    spectrum: Spectrum1D,
    freq_min: float,
    freq_max: float,
) -> float:
    """
    Энергия спектра в диапазоне [freq_min, freq_max] (Гц).
    
    Args:
        spectrum: Спектр для анализа.
        freq_min: Минимальная частота (Гц).
        freq_max: Максимальная частота (Гц).
    
    Returns:
        Суммарная мощность в полосе.
    """
    freq_hz = spectrum.omega / (2 * np.pi)
    mask = (freq_hz >= freq_min) & (freq_hz <= freq_max)
    return float(np.sum(spectrum.power[mask]))


def spectral_entropy(spectrum: Spectrum1D) -> float:
    """
    Спектральная энтропия: H = - sum(p_i * log(p_i)),
    где p_i = нормированная мощность.
    
    Args:
        spectrum: Спектр для анализа.
    
    Returns:
        Значение энтропии.
    """
    total = spectrum.total_power()
    if total == 0:
        return 0.0
    
    # Нормируем мощность (как вероятность)
    p = spectrum.power / total
    
    # Исключаем нули для логарифма
    p = p[p > 0]
    
    return float(-np.sum(p * np.log(p)))


def extract_features(
    spectrum: Spectrum1D,
    bands_hz: list[tuple[float, float]],
) -> np.ndarray:
    """
    Построить вектор фич:
    [ band_power_1, ..., band_power_N, spectral_entropy ]
    
    Args:
        spectrum: Спектр.
        bands_hz: Список кортежей (min_hz, max_hz).
        
    Returns:
        NumPy массив фич.
    """
    features = []
    for fmin, fmax in bands_hz:
        features.append(spectral_band_power(spectrum, fmin, fmax))
    
    features.append(spectral_entropy(spectrum))
    return np.asarray(features, dtype=float)



