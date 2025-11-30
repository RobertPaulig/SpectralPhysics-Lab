from dataclasses import dataclass
import numpy as np
from .medium_1d import OscillatorChain1D
from .root_finding import symmetric_newton

@dataclass
class Layer:
    thickness: float
    density: float
    stiffness: float

@dataclass
class LayeredMedium1D:
    layers: list[Layer]
    dx: float  # шаг дискретизации по глубине

    def to_oscillator_chain(self) -> OscillatorChain1D:
        """
        Построить цепочку осцилляторов, где каждый узел
        наследует параметры слоя, в который он попадает.
        """
        total_depth = sum(layer.thickness for layer in self.layers)
        n_nodes = int(np.ceil(total_depth / self.dx))
        
        masses = np.zeros(n_nodes)
        k_springs = np.zeros(n_nodes - 1)
        
        current_depth = 0.0
        
        # Заполняем массы узлов
        for i in range(n_nodes):
            depth = i * self.dx
            
            # Найти слой
            layer_idx = 0
            d_acc = 0.0
            for l_idx, layer in enumerate(self.layers):
                d_acc += layer.thickness
                if depth < d_acc:
                    layer_idx = l_idx
                    break
                # Если глубина больше полной (последний узел), берем последний слой
                layer_idx = len(self.layers) - 1
            
            layer = self.layers[layer_idx]
            
            # Масса узла ~ плотность * dx (объем 1D элемента)
            masses[i] = layer.density * self.dx
            
        # Заполняем пружины
        for i in range(n_nodes - 1):
            depth = (i + 0.5) * self.dx
            
            # Найти слой для пружины
            layer_idx = 0
            d_acc = 0.0
            for l_idx, layer in enumerate(self.layers):
                d_acc += layer.thickness
                if depth < d_acc:
                    layer_idx = l_idx
                    break
                layer_idx = len(self.layers) - 1
            
            layer = self.layers[layer_idx]
            
            # Жесткость пружины ~ stiffness / dx
            k_springs[i] = layer.stiffness / self.dx
            
        return OscillatorChain1D(
            n=n_nodes,
            m=masses,     # Передаем массив масс
            k=k_springs   # Передаем массив жесткостей
        )


def simulate_pulse_response(
    medium: LayeredMedium1D,
    t_max: float | None = None,
    dt: float = 0.1,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Имитация отклика на удар по поверхности:
    возвращает (t, signal) — сигнал на поверхности во времени.
    """
    chain = medium.to_oscillator_chain()
    
    if n_steps is None:
        if t_max is None:
            raise ValueError("Must provide either t_max or n_steps")
        n_steps = int(np.ceil(t_max / dt))
    
    # t = np.linspace(0, t_max, n_steps) # This was causing issues
    t = np.arange(n_steps) * dt
    
    # Интегрирование во времени
    # M x'' + K x = F
    # F(t) = импульс в t=0 на узле 0
    
    # Состояние: x (смещения), v (скорости)
    x = np.zeros(chain.n)
    v = np.zeros(chain.n)
    
    # Начальный импульс: мгновенная скорость узла 0
    # Импульс P -> v[0] = P / m[0]
    # Пусть P = 1.0
    v[0] = 1.0 / chain.m[0]
    
    surface_signal = np.zeros(n_steps)
    
    # Явная схема Эйлера-Кромера (полунеявная) или Верле
    # v(t+dt) = v(t) + a(t)*dt
    # x(t+dt) = x(t) + v(t+dt)*dt
    
    # Для ускорения можно собрать K матрицу один раз, но для 1D цепочки
    # силы считаются быстро: F_i = k_i*(x_{i+1}-x_i) - k_{i-1}*(x_i - x_{i-1})
    
    # Предварительно распарсим k и m для скорости
    k = chain.k
    m_inv = 1.0 / chain.m
    
    # Handle scalar k
    if np.ndim(k) == 0:
        k = np.full(chain.n - 1, float(k))
    
    for step in range(n_steps):
        surface_signal[step] = x[0]
        
        # Расчет сил
        # F = -K * x
        # Внутренние силы
        forces = np.zeros(chain.n)
        
        # Пружины
        # force on i from i+1: k[i] * (x[i+1] - x[i])
        # force on i+1 from i: -k[i] * (x[i+1] - x[i])
        
        dx_springs = x[1:] - x[:-1] # x[i+1] - x[i]
        f_springs = k * dx_springs
        
        forces[:-1] += f_springs
        forces[1:] -= f_springs
        
        # Обновление
        a = forces * m_inv
        v += a * dt
        x += v * dt
        
    return t, surface_signal


def invert_single_layer_thickness(
    target_signal: np.ndarray,
    t: np.ndarray,
    density: float,
    stiffness: float,
    thickness_guess: float,
    fixed_layers_below: list[Layer],
    dx: float = 0.1,
) -> float:
    """
    Игрушечная обратная задача:
    подбираем толщину первого слоя (с известными density/stiffness),
    чтобы сигнал совпал с target.
    
    fixed_layers_below: слои, которые идут ПОД первым слоем (известная подложка).
    """
    
    dt = t[1] - t[0]
    n_steps = len(target_signal)
    
    def objective(h: float) -> float:
        # Собираем среду
        layer1 = Layer(thickness=h, density=density, stiffness=stiffness)
        medium = LayeredMedium1D(layers=[layer1] + fixed_layers_below, dx=dx)
        
        # Симулируем
        _, signal = simulate_pulse_response(medium, dt=dt, n_steps=n_steps)
        
        # Сравниваем (L2 norm)
        diff = signal - target_signal
        return float(np.sum(diff**2))

    from scipy.optimize import minimize_scalar
    
    res = minimize_scalar(
        objective, 
        bounds=(0.5 * thickness_guess, 2.0 * thickness_guess), 
        method='bounded'
    )
    
    res = minimize_scalar(
        objective, 
        bounds=(0.5 * thickness_guess, 2.0 * thickness_guess), 
        method='bounded'
    )
    
    return float(res.x)

def build_geo1d_health_profile(signal: np.ndarray) -> "FeatureSignature":
    """
    Подпись нормального отклика (например, для конкретной скважины/участка).
    Вектор признаков: [energy, max_amplitude, mean_abs, std, zero_crossings]
    """
    from .material import FeatureSignature
    
    # Simple time-domain features
    energy = np.sum(signal**2)
    max_amp = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal))
    std_val = np.std(signal)
    
    # Zero crossings
    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    
    features = np.array([energy, max_amp, mean_abs, std_val, float(zero_crossings)])
    
    return FeatureSignature(reference_features=features)
