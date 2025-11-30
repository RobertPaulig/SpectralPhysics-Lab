Супер. Ниже — текст, который можно **целиком вставить в `AI_TASKS.md`**.
Он написан так, чтобы его читал именно ИИ-агент, а не человек: чёткие действия, файлы, функции, без лишней болтовни.

````markdown
# SpectralPhysics-Lab – AI_TASKS

## Общие требования

- Использовать Python ≥ 3.10.
- Везде писать type hints.
- Использовать только стандартную библиотеку и `numpy` (и `matplotlib` в примерах).
- Код в `src/spectral_physics/`, тесты в `tests/`, примеры в `examples/`.
- Все публичные функции и классы должны иметь понятные docstring (Google-style или NumPy-style).

---

## ETAP 1: Базовая структура проекта

- [ ] Создать директории:
      - `src/spectral_physics/`
      - `tests/`
      - `examples/`
- [ ] Создать файл `src/spectral_physics/__init__.py` с таким содержимым:
      - Определить `__all__ = ["root_finding", "spectrum", "medium_1d", "grav_toy"]`
      - Не импортировать тяжёлые зависимости на верхнем уровне.

---

## ETAP 2: Симметричный метод Ньютона (root_finding.py)

### 2.1. Реализация

- [ ] Создать файл `src/spectral_physics/root_finding.py`.
- [ ] Реализовать функцию:

```python
def symmetric_newton(
    f: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
    h0: float = 1e-3,
) -> tuple[float, dict]:
    """
    Find a root of f(x) = 0 using the derivative-free symmetric Newton method.

    Args:
        f: scalar function f(x).
        x0: initial guess for the root.
        tol: tolerance for |f(x_k)|.
        max_iter: maximum number of iterations.
        h0: initial symmetric step for finite differences.

    Returns:
        root: approximated root x*.
        info: dict with fields:
            - "converged": bool
            - "iterations": int
            - "history": list[float]  # sequence of |f(x_k)|
    """
````

* [ ] Реализовать именно **симметричный** шаг Ньютона: использовать центральную разность
  `f(x + h) - f(x - h)` для оценки производной и адаптивно уменьшать `h`, если шаг становится нестабильным.

### 2.2. Тесты

* [ ] Создать файл `tests/test_root_finding.py`.
* [ ] Добавить тесты:

  * [ ] Функция `f(x) = x**3 - 2`. Проверить, что корень ≈ `2**(1/3)` с точностью `1e-6`.
  * [ ] Функция с "острым" поведением, например `f(x) = np.cbrt(x)` (x^(1/3)) в окрестности нуля.
    Проверить, что метод сходится, а обычный Ньютон (с аналитической производной) легко уходит в NaN
    при том же `x0`.
  * [ ] Тест на ограничение числа итераций: при `max_iter=3` метод должен вернуть `converged=False`.

---

## ETAP 3: Спектр одномерного сигнала (spectrum.py)

### 3.1. Реализация

* [ ] Создать файл `src/spectral_physics/spectrum.py`.
* [ ] Реализовать класс:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Spectrum1D:
    """One-dimensional discrete spectrum of a real signal."""
    omega: np.ndarray        # angular frequencies
    amplitude: np.ndarray    # complex amplitudes (or real for power)
    phase: np.ndarray        # phases in radians

    @classmethod
    def from_time_signal(cls, t: np.ndarray, x: np.ndarray) -> "Spectrum1D":
        """
        Build spectrum from real-valued time signal x(t) using FFT.

        Args:
            t: time grid (1D array, uniform step).
            x: signal values on grid t.

        Returns:
            Spectrum1D instance with positive frequencies only.
        """

    def power(self) -> np.ndarray:
        """Return spectral power density |amplitude|^2."""

    def band_power(self, omega_min: float, omega_max: float) -> float:
        """
        Integrate power over the band [omega_min, omega_max].

        Returns:
            Scalar total power in the given band.
        """
```

### 3.2. Тесты

* [ ] Создать файл `tests/test_spectrum.py`.
* [ ] Добавить тесты:

  * [ ] Сигнал `x(t) = sin(ω0 t)` → в `Spectrum1D` должен быть один ярко выраженный пик близко к `ω0`.
  * [ ] Сигнал `x(t) = sin(ω1 t) + 0.5 * sin(ω2 t)` → два пика, и отношение мощностей
    в полосах около `ω1` и `ω2` должно быть примерно `1 : 0.25`.
  * [ ] Проверить корректность работы `band_power` на узкой полосе вокруг доминирующей частоты.

---

## ETAP 4: Простая одномерная среда осцилляторов (medium_1d.py)

### 4.1. Реализация

* [ ] Создать файл `src/spectral_physics/medium_1d.py`.
* [ ] Реализовать класс:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class OscillatorChain1D:
    """
    1D chain of coupled oscillators as a toy model of a medium.

    Grid: N masses connected by springs with stiffness k.
    Free or fixed boundary conditions (start with fixed).
    """
    N: int
    m: float
    k: float
    damping: float = 0.0

    def __post_init__(self) -> None:
        # allocate x, v arrays
        ...

    def step(self, dt: float) -> None:
        """
        Advance the system by one time step dt using a simple
        explicit integrator (e.g., Velocity-Verlet or leapfrog).
        """

    def energy(self) -> float:
        """
        Return total energy (kinetic + potential) of the chain.
        """
```

### 4.2. Тесты

* [ ] Создать файл `tests/test_medium_1d.py`.
* [ ] Добавить тесты:

  * [ ] Возбудить один узел цепочки (x[0] = 1.0, все остальные = 0) и выполнить несколько шагов.
    Проверить, что энергия не возрастает при `damping = 0` (допускается небольшая численная ошибка).
  * [ ] При `damping > 0` энергия должна монотонно уменьшаться.
  * [ ] Проверить, что сигнал действительно "расползается" по цепочке: дисперсия положения по индексу растёт.

---

## ETAP 5: Игрушечная модель гравитации через спектральную тень (grav_toy.py)

### 5.1. Реализация

* [ ] Создать файл `src/spectral_physics/grav_toy.py`.
* [ ] Реализовать функции:

```python
from .medium_1d import OscillatorChain1D
from .spectrum import Spectrum1D
import numpy as np

def compute_local_spectrum(chain: OscillatorChain1D, idx_range: slice) -> Spectrum1D:
    """
    Record time series of oscillations for a subset of nodes (idx_range),
    then build Spectrum1D for the averaged signal in this region.
    """

def spectral_pressure_difference(
    chain: OscillatorChain1D,
    left_range: slice,
    right_range: slice,
    *,
    t_total: float,
    dt: float,
    band: tuple[float, float] | None = None,
) -> float:
    """
    Compute toy-model 'force' as difference of spectral power between left and right regions.

    Steps:
        1. Run simulation of chain for time t_total with step dt.
        2. Build spectra for left_range and right_range via compute_local_spectrum.
        3. Compute power (or band power) for both sides.
        4. Return P_left - P_right as effective spectral pressure difference.
    """
```

### 5.2. Тесты

* [ ] Создать файл `tests/test_grav_toy.py`.
* [ ] Добавить тесты:

  * [ ] Без "препятствия" (однородная цепочка, одинаковое возбуждение с обеих сторон) →
    `spectral_pressure_difference(...)` ≈ 0.
  * [ ] Ввести "препятствие" в виде более тяжёлой массы или изменённой жёсткости в области между
    left_range и right_range → полученное значение должно быть статистически отличимо от нуля
    и иметь знак, соответствующий "приталкиванию" к области с тенью.

---

## ETAP 6: Примеры (examples/)

### 6.1. Демо симметричного Ньютона

* [ ] Создать `examples/symmetric_newton_demo.ipynb` (или `.py`, если удобнее).
* [ ] Показать:

  * Решение `x**3 - 2 = 0`.
  * Решение задачи с "острым" корнем.
  * Графики `|f(x_k)|` по итерациям.

### 6.2. Демо волновой среды и спектральной тени

* [ ] Создать `examples/medium_oscillators_demo.ipynb`.
* [ ] Показать:

  * Возбуждение одного узла и распространение волны по цепочке.
  * Вычисление спектров слева и справа от "препятствия".
  * График "спектрального давления" (P_left и P_right) и разности между ними.

---

```

Скопируй этот блок в `AI_TASKS.md`, сохрани — и дай своему локальному ИИ отработать весь список. Когда он закончит генерить код, просто покажи мне файлы, и будем уже шлифовать математику и API.
::contentReference[oaicite:0]{index=0}
```
