# МЕГА-ПРОЕКТ: spectral-health v1.0

Цель: превратить текущий код в **полноценный инструмент диагностики машин**:

- Вход: CSV с вибрацией/сигналами.
- Выход: CLI-утилита `spectral-health`, которая:
  - обучает "здоровый профиль" по историческим данным;
  - анализирует новые данные;
  - даёт статус по каждому каналу (OK / ANOMALY);
  - пишет красивый Markdown-отчёт.

---

## ЭТАП 20. Доводим ядро и упаковку до боевого состояния

**Файлы:** `pyproject.toml`, `README.md`, `src/spectral_physics/cli.py`

- [ ] Исправить `pyproject.toml`:
  - корректный блок `[project]` с:
    - `name`, `version`, `description`, `readme`, `requires-python`;
    - `dependencies = ["numpy", "scipy", "matplotlib", "pyyaml"]` и т.п.
  - рабочий `[project.scripts]`:
    ```toml
    [project.scripts]
    spectral-health = "spectral_physics.cli:main"
    ```

- [ ] Поправить импорт `HealthProfile` в `cli.py`:
  - импортировать `HealthProfile` из `spectral_physics.material`.

- [ ] Прочистить `README.md`:
  - убрать дублирующийся кусок текста;
  - добавить раздел:
    ```markdown
    ## Installation

    ```bash
    pip install -e .
    spectral-health --help
    ```
    ```

---

## ЭТАП 21. YAML-конфиги и демонстрационный pipeline

**Новые файлы:**
- `configs/pump_train.yaml`
- `configs/pump_score.yaml`
- `configs/pump_thresholds.yaml`
- `examples/generate_synthetic_pump_data.py`

### 21.1. Пример конфигов

- [ ] Создать папку `configs/`.

- [ ] `configs/pump_train.yaml` — пример конфигурации для обучения:

  ```yaml
  dt: 0.001          # шаг дискретизации
  window: hann

  channels:
    motor_vibration:
      column: 1
      freq_min: 0.0
      freq_max: 500.0
      files:
        - data/pump/train_motor_1.csv
        - data/pump/train_motor_2.csv

    pump_vibration:
      column: 2
      freq_min: 0.0
      freq_max: 500.0
      files:
        - data/pump/train_pump_1.csv
        - data/pump/train_pump_2.csv
  ```

- [ ] `configs/pump_score.yaml` — аналогично, но для **текущего состояния**:

  ```yaml
  dt: 0.001
  window: hann

  channels:
    motor_vibration:
      column: 1
      freq_min: 0.0
      freq_max: 500.0
      files:
        - data/pump/current_motor.csv

    pump_vibration:
      column: 2
      freq_min: 0.0
      freq_max: 500.0
      files:
        - data/pump/current_pump.csv
  ```

- [ ] `configs/pump_thresholds.yaml` — пороги:

  ```yaml
  motor_vibration: 0.15
  pump_vibration: 0.20
  ```

### 21.2. Генератор синтетических данных

**Файл:** `examples/generate_synthetic_pump_data.py`

- [ ] Написать скрипт, который:

  * создаёт папку `data/pump/`;
  * генерирует тренировочные CSV:

    * `train_motor_1.csv`, `train_motor_2.csv`;
    * `train_pump_1.csv`, `train_pump_2.csv`;
  * генерирует "текущие" CSV:

    * `current_motor.csv` (нормальный);
    * `current_pump.csv` (с аномалией: добавлен новый пик, шум или дрейф).

- [ ] Формат CSV:

  ```text
  time, motor_vibration, pump_vibration
  0.000, ...
  0.001, ...
  ...
  ```

  * первая строка — заголовок;
  * столбец 1 — время, столбец 2 — мотор, столбец 3 — насос.

- [ ] Внутри использовать `numpy` для генерации сигналов:

  * Healthy: сумма нескольких синусов + немного шума;
  * Anomaly: усиленный один из пиков + дополнительный шум.

---

## ЭТАП 22. Умный health-профиль с фичами (band-power + энтропия)

Сейчас `MaterialSignature.distance_l2` сравнивает **нормированные спектры целиком** 

Хочется добавить ещё один уровень: сравнение по набору фич.

**Файлы:** `src/spectral_physics/material.py`, `src/spectral_physics/diagnostics.py`, `tests/test_material.py`, `tests/test_diagnostics.py`

### 22.1. Вектор фич по спектру

- [ ] Добавить в `diagnostics.py` функцию:

  ```python
  import numpy as np
  from .spectrum import Spectrum1D
  from .diagnostics import spectral_band_power, spectral_entropy

  def extract_features(
      spectrum: Spectrum1D,
      bands_hz: list[tuple[float, float]],
  ) -> np.ndarray:
      """
      Построить вектор фич:
      [ band_power_1, ..., band_power_N, spectral_entropy ]
      """
      features = []
      for fmin, fmax in bands_hz:
          features.append(spectral_band_power(spectrum, fmin, fmax))
      features.append(spectral_entropy(spectrum))
      return np.asarray(features, dtype=float)
  ```

- [ ] Добавить тест в `tests/test_diagnostics.py`:

  * создать простой спектр с двумя частотами;
  * проверить, что `extract_features` даёт ожидаемые значения band power;
  * проверить, что размерность вектора = `len(bands) + 1`.

### 22.2. Фичевая сигнатура материала

- [ ] В `material.py` добавить новый dataclass:

  ```python
  @dataclass
  class FeatureSignature:
      """
      Спектральная сигнатура на пространстве фич.
      """
      reference_features: np.ndarray

      def distance_l2(self, other_features: np.ndarray) -> float:
          if other_features.shape != self.reference_features.shape:
              raise ValueError("Feature vector shape mismatch")
          diff = self.reference_features - other_features
          return float(np.sqrt(np.sum(diff**2)))
  ```

- [ ] Добавить тесты в `tests/test_material.py`:

  * `test_feature_signature_zero_distance_for_identical`;
  * `test_feature_signature_shape_mismatch_raises`.

### 22.3. HealthProfile с двумя уровнями

- [ ] Обновить `HealthProfile` так, чтобы он мог хранить **оба типа** сигнатур:

  ```python
  @dataclass
  class HealthProfile:
      signatures: dict[str, MaterialSignature]
      feature_signatures: dict[str, FeatureSignature] | None = None
  ```

- [ ] Добавить метод:

  ```python
  def score_features(
      self,
      current: dict[str, Spectrum1D],
      bands_hz: dict[str, list[tuple[float, float]]],
  ) -> dict[str, float]:
      """
      Для каждого канала:
      - извлечь фичи,
      - посчитать L2-дистанцию в пространстве фич.
      """
  ```

- [ ] Добавить тест в `tests/test_material.py`, который:

  * строит простые спектры;
  * создаёт `FeatureSignature`;
  * проверяет, что `score_features` возвращает корректный словарь.

---

## ЭТАП 23. Полный демо-кейс: от генерации данных до отчёта

**Цель:** Один сценарий, который можно описать в README: *«запусти эти команды — и получишь отчёт о состоянии виртуального насоса».*

**Файлы:** `README.md`, `examples/health_monitor_demo.ipynb`, возможно новый `examples/pump_health_demo.ipynb`.

- [ ] Обновить или создать ноутбук `examples/health_monitor_demo.ipynb` так, чтобы он делал:

  1. `!python examples/generate_synthetic_pump_data.py`
  2. `!spectral-health train --config configs/pump_train.yaml --out data/pump/profile.npz`
  3. `!spectral-health score --config configs/pump_score.yaml --profile data/pump/profile.npz --thresholds configs/pump_thresholds.yaml --report data/pump/report.md`
  4. В конце ноутбука открыть и показать `report.md`.

- [ ] Дополнить `README.md` разделом **"Quick start: Pump demo"**:

  ```markdown
  ## Quick start: Pump health demo

  ```bash
  pip install -e .

  # 1. Сгенерировать синтетические данные
  python examples/generate_synthetic_pump_data.py

  # 2. Обучить профиль "здорового" состояния
  spectral-health train \
    --config configs/pump_train.yaml \
    --out data/pump/profile.npz

  # 3. Оценить текущее состояние и получить отчёт
  spectral-health score \
    --config configs/pump_score.yaml \
    --profile data/pump/profile.npz \
    --thresholds configs/pump_thresholds.yaml \
    --report data/pump/report.md
  ```
  ```

- [ ] Убедиться, что в отчёте есть как минимум:

  * таблица с каналами, distance, threshold, статусом (уже делает `generate_markdown_report`) 
  * понятный текст: "All systems nominal" или "Anomalies detected!"
