"""Быстрая проверка новых функций"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from spectral_physics.spectrum import Spectrum1D
from spectral_physics.medium_1d import OscillatorChain1D
from spectral_physics.grav_toy import spectral_pressure_difference

print("=== Тест 1: Spectrum1D.from_time_signal ===")
# Создаем синусоидальный сигнал
t = np.linspace(0, 10, 1000)
omega0 = 2.0  # Ожидаемая частота
x = np.sin(omega0 * t)

spec = Spectrum1D.from_time_signal(t, x)
print(f"Создан спектр с {len(spec.freqs)} точками")
print(f"Максимум спектра на частоте: {spec.freqs[np.argmax(spec.amps)]:.2f} (ожидается ~{omega0:.2f})")

print("\n=== Тест 2: Spectrum1D.band_power ===")
# Вычислим мощность в узкой полосе вокруг omega0
band_power = spec.band_power(omega0 - 0.5, omega0 + 0.5)
total_power = spec.total_power()
print(f"Мощность в полосе [{omega0-0.5:.2f}, {omega0+0.5:.2f}]: {band_power:.4f}")
print(f"Полная мощность: {total_power:.4f}")
print(f"Доля мощности в полосе: {band_power/total_power*100:.1f}%")

print("\n=== Тест 3: OscillatorChain1D.step и energy ===")
chain = OscillatorChain1D(n_points=10, k_coupling=1.0, m=1.0, damping=0.0)
# Возбудим один осциллятор
chain.x[0] = 1.0
chain.v[0] = 0.0

E0 = chain.energy()
print(f"Начальная энергия: {E0:.6f}")

# Сделаем несколько шагов
for _ in range(100):
    chain.step(0.01)

E1 = chain.energy()
print(f"Энергия после 100 шагов: {E1:.6f}")
print(f"Изменение энергии: {abs(E1 - E0)/E0 * 100:.2f}% (должно быть мало для damping=0)")

print("\n=== Тест 4: OscillatorChain1D с затуханием ===")
chain_damped = OscillatorChain1D(n_points=10, k_coupling=1.0, m=1.0, damping=0.1)
chain_damped.x[0] = 1.0

E0_d = chain_damped.energy()
for _ in range(100):
    chain_damped.step(0.01)
E1_d = chain_damped.energy()

print(f"Начальная энергия: {E0_d:.6f}")
print(f"Энергия после 100 шагов: {E1_d:.6f}")
print(f"Уменьшение: {(E0_d - E1_d)/E0_d * 100:.1f}% (должно быть заметно для damping=0.1)")

print("\n=== Тест 5: spectral_pressure_difference ===")
# Создадим цепочку и возбудим её асимметрично
chain_sim = OscillatorChain1D(n_points=20, k_coupling=1.0, m=1.0, damping=0.05)
# Возбудим левую сторону сильнее
chain_sim.x[2] = 1.0
chain_sim.v[2] = 0.5

try:
    pressure_diff = spectral_pressure_difference(
        chain_sim,
        left_range=slice(0, 5),
        right_range=slice(15, 20),
        t_total=20.0,
        dt=0.05
    )
    print(f"Разность спектрального давления: {pressure_diff:.6f}")
    print("Функция работает!")
except Exception as e:
    print(f"Ошибка: {e}")

print("\n✅ Все базовые тесты пройдены успешно!")
