import numpy as np

def ldos_from_modes(
    modes: np.ndarray,
    omegas: np.ndarray,
    freq_window: tuple[float, float],
) -> np.ndarray:
    """
    Оценка локальной спектральной плотности (LDOS) на решётке 2D.

    Параметры:
        modes: массив формы (N_points, N_modes),
               как вернул OscillatorGrid2D.eigenmodes()
        omegas: массив (N_modes,) — частоты мод
        freq_window: (omega_min, omega_max) — частотное окно

    Возвращает:
        ldos: массив (N_points,) — "сила" локального спектра
              в заданном окне частот.
    Идея:
        суммировать |mode_i|^2 по тем модам, у которых
        omega_i попадает в окно.
    """
    w_min, w_max = freq_window
    
    # Маска частот, попадающих в окно
    mask = (omegas >= w_min) & (omegas <= w_max)
    
    if not np.any(mask):
        return np.zeros(modes.shape[0])
        
    # Выбираем нужные моды: (N_points, K_selected)
    selected_modes = modes[:, mask]
    
    # Суммируем квадраты амплитуд по модам (вдоль оси 1)
    # LDOS(x) = sum_i |psi_i(x)|^2
    ldos = np.sum(selected_modes**2, axis=1)
    
    return ldos
