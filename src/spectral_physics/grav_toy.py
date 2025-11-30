from .spectrum import Spectrum1D
import numpy as np

def spectral_pressure_difference(spec_left: Spectrum1D, spec_right: Spectrum1D) -> float:
    """
    Оценка 'градиента давления' между двумя точками как разности
    их полной спектральной мощности.

    Возвращает скаляр:
        p_left - p_right
    где p = total_power().
    """
    p_left = spec_left.total_power()
    p_right = spec_right.total_power()
    return float(p_left - p_right)
