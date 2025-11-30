import numpy as np
from dataclasses import dataclass
from .spectrum import Spectrum1D

@dataclass
class AtomicResonator:
    """
    Игрушечная модель атома как набора резонансных частот на поверхности.
    """
    name: str
    omega: np.ndarray        # собственные частоты (рад/с или условные единицы)
    power: np.ndarray        # "интенсивность" / веса линий
    max_bonds: int           # сколько связей атом "выдерживает"
    core_modes: list[tuple[float, float]] = None    # диапазоны частот ядра (стабильные)
    valence_modes: list[tuple[float, float]] = None # диапазоны частот валентности (активные)

    def __post_init__(self):
        if self.core_modes is None:
            self.core_modes = []
        if self.valence_modes is None:
            self.valence_modes = []

    @classmethod
    def from_lines(
        cls,
        name: str,
        lines: list[tuple[float, float]],
        max_bonds: int,
        core_modes: list[tuple[float, float]] = None,
        valence_modes: list[tuple[float, float]] = None,
    ) -> "AtomicResonator":
        """
        lines: список (частота, интенсивность)
        """
        if not lines:
            return cls(name=name, omega=np.array([]), power=np.array([]), max_bonds=max_bonds,
                       core_modes=core_modes, valence_modes=valence_modes)
            
        omega = np.array([f for f, a in lines], dtype=float)
        power = np.array([a for f, a in lines], dtype=float)
        return cls(name=name, omega=omega, power=power, max_bonds=max_bonds,
                   core_modes=core_modes, valence_modes=valence_modes)

    def spectrum(self) -> Spectrum1D:
        """
        Вернуть Spectrum1D с заданными линиями.
        """
        return Spectrum1D(omega=self.omega, power=self.power)

    def normalized_spectrum(self) -> Spectrum1D:
        """
        Нормированный спектр (сумма power = 1).
        """
        return self.spectrum().normalize()


def spectral_overlap(
    atom_a: AtomicResonator,
    atom_b: AtomicResonator,
    freq_tol: float = 0.05,
) -> float:
    """
    Оценка "резонансной совместимости" двух атомов.

    Идея:
    - берём линии из A и B
    - если разность частот |ω_a - ω_b| < freq_tol * ω_avg,
      считаем, что эти линии могут "сцепиться"
    - суммаем веса совпадающих линий как меру силы связи.

    Возвращает число [0..1] примерно: чем больше, тем "лучше" связь.
    """
    if len(atom_a.omega) == 0 or len(atom_b.omega) == 0:
        return 0.0
        
    overlap_score = 0.0
    
    # Нормируем веса, чтобы оценка не зависела от абсолютной амплитуды
    # (или считаем, что power уже отражает "важность")
    # Давайте работать с нормированными спектрами для честности
    spec_a = atom_a.normalized_spectrum()
    spec_b = atom_b.normalized_spectrum()
    
    # Простой перебор всех пар линий (O(Na * Nb)) - для атомов это мало
    for i, wa in enumerate(spec_a.omega):
        pa = spec_a.power[i]
        
        for j, wb in enumerate(spec_b.omega):
            pb = spec_b.power[j]
            
            w_avg = (wa + wb) / 2.0
            if w_avg == 0:
                continue
                
            diff = abs(wa - wb)
            
            if diff < freq_tol * w_avg:
                # Резонанс!
                # Вклад в связь пропорционален произведению интенсивностей (или минимуму?)
                # Пусть будет произведение, как вероятность совпадения
                # Или сумма?
                # ТЗ: "суммаем веса совпадающих линий"
                # Возьмем min(pa, pb) как "общая энергия резонанса"
                overlap_score += min(pa, pb)
                
    # Ограничим 1.0 (хотя может быть и больше, если много линий совпадают с одной)
    return min(overlap_score, 1.0)


def can_form_bond(
    atom_a: AtomicResonator,
    atom_b: AtomicResonator,
    freq_tol: float,
    threshold: float,
) -> bool:
    """
    Решаем, возможна ли устойчивое "сцепление":
    overlap >= threshold и не превышены max_bonds у обоих.
    """
    if atom_a.max_bonds <= 0 or atom_b.max_bonds <= 0:
        return False
        
    score = spectral_overlap(atom_a, atom_b, freq_tol)
    return score >= threshold


def load_default_atom_db() -> dict[str, AtomicResonator]:
    """
    Load a 'toy' database of atoms with spectral properties.
    """
    db = {}
    
    # Hydrogen: Simple, one valence mode
    db['H'] = AtomicResonator.from_lines(
        name="H",
        lines=[(1.0, 1.0)],
        max_bonds=1,
        valence_modes=[(0.9, 1.1)]
    )
    
    # Carbon: Complex, core modes + valence
    db['C'] = AtomicResonator.from_lines(
        name="C",
        lines=[(0.5, 0.4), (1.0, 0.8), (1.5, 0.6)], # 0.5 core, 1.0/1.5 valence
        max_bonds=4,
        core_modes=[(0.4, 0.6)],
        valence_modes=[(0.9, 1.6)]
    )
    
    # Oxygen: Electronegative, distinct modes
    db['O'] = AtomicResonator.from_lines(
        name="O",
        lines=[(0.8, 0.6), (1.2, 0.9)],
        max_bonds=2,
        core_modes=[(0.7, 0.9)],
        valence_modes=[(1.1, 1.3)]
    )
    
    # Iron: Heavy, low frequency core, many lines
    db['Fe'] = AtomicResonator.from_lines(
        name="Fe",
        lines=[(0.2, 0.8), (0.3, 0.7), (0.8, 0.5)],
        max_bonds=6,
        core_modes=[(0.1, 0.4)],
        valence_modes=[(0.7, 0.9)]
    )
    
    # Silicon: Like Carbon but heavier (lower freq)
    db['Si'] = AtomicResonator.from_lines(
        name="Si",
        lines=[(0.3, 0.5), (0.7, 0.8), (1.1, 0.5)],
        max_bonds=4,
        core_modes=[(0.2, 0.4)],
        valence_modes=[(0.6, 1.2)]
    )
    
    return db
