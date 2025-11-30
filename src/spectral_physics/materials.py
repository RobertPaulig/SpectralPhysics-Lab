from dataclasses import dataclass
import numpy as np
from .atoms import AtomicResonator
from .spectrum import Spectrum1D

@dataclass
class MaterialPatch:
    """
    Участок материала как "атомная смесь" плюс среда.
    """
    atoms: list[AtomicResonator]
    weights: np.ndarray  # доли каждого атома (сумма = 1)

    def surface_spectrum(self) -> Spectrum1D:
        """
        Суммарный спектр поверхности:
        взвешенная сумма спектров атомов.
        """
        if not self.atoms:
            return Spectrum1D(omega=np.array([]), power=np.array([]))
            
        # Для простоты объединим все частоты и просуммируем мощности
        # (предполагаем дискретные линии)
        
        all_freqs = []
        all_powers = []
        
        for atom, w in zip(self.atoms, self.weights):
            spec = atom.spectrum()
            all_freqs.extend(spec.omega)
            all_powers.extend(spec.power * w)
            
        all_freqs = np.array(all_freqs)
        all_powers = np.array(all_powers)
        
        # Сортируем по частоте
        idx = np.argsort(all_freqs)
        sorted_freqs = all_freqs[idx]
        sorted_powers = all_powers[idx]
        
        # Можно объединить совпадающие частоты, но Spectrum1D этого не требует
        return Spectrum1D(omega=sorted_freqs, power=sorted_powers)


def effective_coupling(
    ldos: np.ndarray,
    patch: MaterialPatch,
    freq_window: tuple[float, float],
) -> float:
    """
    Оценка того, насколько данный участок материала "звучит"
    в данном частотном окне Среды.

    Идея:
      - из patch.surface_spectrum() взять Spectrum1D
      - ограничить его freq_window
      - умножить интегральную мощность на средний уровень LDOS
        (например, <ldos> по области интереса)
    
    Args:
        ldos: массив LDOS значений (например, карта или срез).
        patch: MaterialPatch.
        freq_window: (w_min, w_max).
    """
    spec = patch.surface_spectrum()
    
    # Фильтруем спектр патча по окну
    w_min, w_max = freq_window
    mask = (spec.omega >= w_min) & (spec.omega <= w_max)
    
    if not np.any(mask):
        return 0.0
        
    patch_power = np.sum(spec.power[mask])
    
    # Средний LDOS
    avg_ldos = np.mean(ldos)
    
    return float(patch_power * avg_ldos)

def build_material_health_profile(ldos_map: np.ndarray) -> "FeatureSignature":
    """
    Строим "здоровый" профиль для материала:
    - flatten ldos_map
    - извлекаем статистики (mean, std, PCA и т.д. — см. spectral-health)
    - получаем компактный вектор health-подписи.
    
    Для простоты: вектор = [mean, std, max, min, median]
    """
    from .material import FeatureSignature
    
    flat = ldos_map.flatten()
    features = np.array([
        np.mean(flat),
        np.std(flat),
        np.max(flat),
        np.min(flat),
        np.median(flat)
    ])
    
    return FeatureSignature(reference_features=features)
