import numpy as np
from .spectrum import Spectrum1D


def timeseries_to_spectrum(
    signal: np.ndarray,
    dt: float,
    window: str = "hann",
) -> Spectrum1D:
    """
    Преобразовать одномерный временной сигнал в амплитудный спектр.
    
    Параметры:
        signal: 1D-массив отсчётов во времени.
        dt: шаг по времени (секунды).
        window: тип окна ("hann" или None).
    
    Возвращает:
        Spectrum1D с:
            omega: угловые частоты (rad/s) для положительных частот.
            power: |FFT|^2 / N или аналогичная нормировка.
    
    Example:
        >>> t = np.arange(0, 1.0, 0.001)
        >>> signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz
        >>> spectrum = timeseries_to_spectrum(signal, dt=0.001)
    """
    signal = np.asarray(signal, dtype=float)
    
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D array, got shape {signal.shape}")
    
    n = len(signal)
    if n == 0:
        raise ValueError("Signal must not be empty")
    
    # Удаление DC-компонента (вычитание среднего)
    signal_ac = signal - np.mean(signal)
    
    # Применение оконной функции
    if window == "hann":
        window_func = np.hanning(n)
        signal_windowed = signal_ac * window_func
    elif window is None:
        signal_windowed = signal_ac
    else:
        raise ValueError(f"Unknown window type: {window}. Use 'hann' or None.")
    
    # FFT (только положительные частоты)
    fft_vals = np.fft.rfft(signal_windowed)
    
    # Частоты в Герцах
    freq_hz = np.fft.rfftfreq(n, d=dt)
    
    # Преобразование в угловые частоты (rad/s)
    omega = 2 * np.pi * freq_hz
    
    # Мощность: |FFT|^2, нормированная на количество точек
    # Коэффициент 2 для учёта энергии в отрицательных частотах (кроме DC и Nyquist)
    power = np.abs(fft_vals) ** 2 / n
    
    # Удвоить мощность для всех частот кроме DC (индекс 0) и Nyquist (последний для чётного n)
    if n % 2 == 0:
        # Чётное n: DC и Nyquist не удваиваются
        power[1:-1] *= 2
    else:
        # Нечётное n: только DC не удваивается
        power[1:] *= 2
    
    return Spectrum1D(omega=omega, power=power)
