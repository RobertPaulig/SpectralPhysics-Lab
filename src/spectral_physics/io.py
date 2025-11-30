import numpy as np
from pathlib import Path
from .spectrum import Spectrum1D


def load_timeseries_csv(
    path: str,
    column: int = 0,
    skip_header: bool = True
) -> np.ndarray:
    """
    Загрузить одномерный временной ряд из CSV.

    Параметры:
        path: Путь к файлу CSV.
        column: Индекс колонки (0-based), где лежит сигнал.
        skip_header: Если True, пропустить первую строку (заголовок).

    Возвращает:
        1D np.ndarray с данными сигнала.
    
    Raises:
        ValueError: Если файл пустой, колонка отсутствует или файл не найден.
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise ValueError(f"File not found: {path}")
    
    # Определяем число строк для пропуска
    skiprows = 1 if skip_header else 0
    
    try:
        # Загружаем данные
        data = np.loadtxt(path, delimiter=',', skiprows=skiprows, ndmin=2)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file {path}: {e}")
    
    # Проверка на пустой файл
    if data.size == 0:
        raise ValueError(f"File {path} is empty")
    
    # Если данные одномерные, сделать двумерными
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Проверка корректности индекса колонки
    if column >= data.shape[1] or column < 0:
        raise ValueError(
            f"Column index {column} is out of range. "
            f"File has {data.shape[1]} columns (indices 0..{data.shape[1]-1})"
        )
    
    # Извлечь нужную колонку
    signal = data[:, column]
    
    return signal


def save_spectrum_npz(spectrum: Spectrum1D, path: str) -> None:
    """
    Сохранить Spectrum1D в .npz (omega, power).
    
    Args:
        spectrum: Спектр для сохранения.
        path: Путь к файлу .npz.
    """
    np.savez(path, omega=spectrum.omega, power=spectrum.power)


def load_spectrum_npz(path: str) -> Spectrum1D:
    """
    Загрузить Spectrum1D из .npz.
    Ожидает массивы 'omega' и 'power'.
    
    Args:
        path: Путь к файлу .npz.
    
    Returns:
        Загруженный Spectrum1D.
    
    Raises:
        ValueError: Если файл не найден или отсутствуют ожидаемые ключи.
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise ValueError(f"File not found: {path}")
    
    try:
        data = np.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load .npz file {path}: {e}")
    
    # Проверка наличия необходимых ключей
    if 'omega' not in data or 'power' not in data:
        raise ValueError(
            f"File {path} missing required keys. "
            f"Expected 'omega' and 'power', found: {list(data.keys())}"
        )
    
    omega = data['omega']
    power = data['power']
    
    return Spectrum1D(omega=omega, power=power)
