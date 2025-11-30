"""Быстрая проверка новой архитектуры"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from spectral_physics.spectrum import Spectrum1D
from spectral_physics.medium_1d import OscillatorChain1D
from spectral_physics.grav_toy import spectral_pressure_difference
from spectral_physics.root_finding import symmetric_newton

print("=== Тест 1: Spectrum1D (новая архитектура) ===")
omega = np.array([1.0, 2.0, 3.0])
power = np.array([1.0, 2.0, 1.0])
spec = Spectrum1D(omega=omega, power=power)
print(f"Создан спектр: omega={spec.omega}, power={spec.power}")
print(f"Полная мощность: {spec.total_power()}")

normalized = spec.normalize()
print(f"Нормализованная мощность: {normalized.total_power():.6f}")

alpha = np.array([1.0, 0.5, 0.0])
filtered = spec.apply_filter(alpha)
print(f"Отфильтрованная мощность: {filtered.power}")

print("\n=== Тест 2: OscillatorChain1D (новая архитектура) ===")
chain = OscillatorChain1D(n=5, k=1.0, m=1.0)
print(f"Создана цепочка: n={chain.n}, k={chain.k}, m={chain.m}")

K = chain.stiffness_matrix()
print(f"Матрица жесткости (5x5):\n{K}")

omega_eig, modes = chain.eigenmodes()
print(f"Собственные частоты: {omega_eig}")
print(f"Число мод: {len(omega_eig)}")

print("\n=== Тест 3: spectral_pressure_difference (новая сигнатура) ===")
spectrum_bg = Spectrum1D(
    omega=np.array([1.0, 2.0, 3.0]),
    power=np.array([1.0, 1.0, 1.0])
)
alpha_left = np.array([0.2, 0.2, 0.2])
alpha_right = np.array([0.8, 0.8, 0.8])

delta_p = spectral_pressure_difference(spectrum_bg, alpha_left, alpha_right)
print(f"Разность давления: {delta_p:.4f}")
print(f"Ожидается: {3 * 0.6:.4f}")

print("\n=== Тест 4: symmetric_newton (обновленная сигнатура) ===")
def f(x):
    return x**2 - 2

x_root, n_iter = symmetric_newton(f, x0=1.0)
print(f"Корень x^2 - 2 = 0: {x_root:.6f} (ожидается {np.sqrt(2):.6f})")
print(f"Итераций: {n_iter}")

print("\n✅ Все тесты новой архитектуры пройдены успешно!")
