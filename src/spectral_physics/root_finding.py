import numpy as np
from typing import Callable, Tuple


def symmetric_newton(
    f: Callable[[float], float],
    x0: float,
    h0: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-10,
    tol_step: float = 1e-12,
) -> Tuple[float, int]:
    """
    Найти корень уравнения f(x) = 0 симметричным методом Ньютона,
    не используя аналитическую производную.

    Использует симметричную разностную производную:
        f'(x) ≈ (f(x + h) - f(x - h)) / (2*h)

    и адаптивно уменьшает h, если шаг получается слишком большим
    или метод начинает расходиться.

    Параметры:
        f: вызываемая функция f(x) -> float
        x0: начальное приближение
        h0: начальный шаг для симметричной разности
        max_iter: максимальное число итераций
        tol: допуск по |f(x)|
        tol_step: допуск по величине шага |delta|

    Возвращает:
        x_root: найденное значение x, для которого f(x) ≈ 0
        n_iter: число выполненных итераций
    
    Raises:
        ValueError: если метод расходится (|x| становится слишком большим)
        
    Notes:
        Функция останавливается, если выполнено ЛЮБОЕ из условий:
        - |f(x)| < tol
        - |delta| < tol_step (полезно на плоских участках)
        - достигнуто max_iter
    """
    x = float(x0)
    h = float(h0)
    
    for i in range(max_iter):
        fx = f(x)
        
        # Check convergence by function value
        if abs(fx) < tol:
            return x, i
        
        # Check for divergence
        if abs(x) > 1e10:
            raise ValueError(f"Method diverged: |x| = {abs(x)} > 1e10")
        
        # Symmetric difference derivative
        # f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
        try:
            df = (f(x + h) - f(x - h)) / (2 * h)
        except Exception as e:
            raise ValueError(f"Failed to compute derivative at x={x}: {e}")
        
        # Protect against division by near-zero derivative
        if abs(df) < 1e-14:
            # Try reducing h
            h = h / 2
            if h < 1e-15:
                # Give up
                return x, i
            continue
        
        # Newton step
        delta = fx / df
        
        # Check convergence by step size
        if abs(delta) < tol_step:
            return x, i
        
        # Adaptive step: if step is too large, reduce it
        if abs(delta) > 100:
            delta = 100 * np.sign(delta)
            h = h / 2  # Also reduce h for next iteration
        
        x_new = x - delta
        
        # Simple backtracking: if new value is worse, reduce step
        try:
            fx_new = f(x_new)
            if abs(fx_new) > abs(fx) * 2:  # Getting worse
                delta = delta / 2
                x_new = x - delta
        except Exception:
            # If evaluation fails, reduce step
            delta = delta / 2
            x_new = x - delta
        
        x = x_new
    
    # Reached max iterations
    return x, max_iter


