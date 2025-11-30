import numpy as np

def symmetric_newton(
    f,
    x0,
    max_iter=50,
    tol=1e-8,
    h=1e-6,
    verbose=False,
):
    """
    Находит корень уравнения f(x) = 0, используя симметричный метод Ньютона
    с численным приближением производной.

    Параметры:
        f: вызываемая функция f(x) -> float
        x0: начальное приближение
        max_iter: максимальное число итераций
        tol: допуск по |f(x)|
        h: шаг для симметричной разности
        verbose: если True, печатать прогресс

    Возвращает:
        x_root: найденное значение x, для которого f(x) ≈ 0
    """
    x = float(x0)
    
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            if verbose:
                print(f"Converged at iter {i}: x={x}, f(x)={fx}")
            return x
        
        # Symmetric difference derivative
        # f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
        df = (f(x + h) - f(x - h)) / (2 * h)
        
        if abs(df) < 1e-12:
            if verbose:
                print(f"Derivative too small at iter {i}: df={df}")
            # Simple fallback or break? 
            # Let's try to perturb x slightly or just break
            # For robustness, let's just break to avoid division by zero
            # Or maybe return current x with a warning?
            # The task says: "Защититься от деления на очень маленькую производную"
            # Let's stop here.
            break
            
        # Newton step
        delta = fx / df
        x_new = x - delta
        
        # Simple backtracking line search to prevent divergence
        # If the new function value is worse (larger magnitude), reduce the step.
        # This helps with functions like x^(1/3) where full Newton step overshoots.
        for _ in range(5):
            try:
                fx_new = f(x_new)
                if abs(fx_new) < abs(fx):
                    break
            except Exception:
                pass # If f(x_new) fails, we definitely want to reduce step
            
            delta /= 2
            x_new = x - delta
        
        if verbose:
            print(f"Iter {i}: x={x}, f(x)={fx}, df={df}, x_new={x_new}")
            
        x = x_new
        
    return x
