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


@dataclass
class MoleculeGraph:
    """
    Graph representation of a molecule.
    Nodes are AtomicResonators, Edges are bonds.
    """
    atoms: list[AtomicResonator]
    bonds: list[tuple[int, int, str]] # (atom_idx1, atom_idx2, bond_type)
    
    def combined_valence_spectrum(self) -> Spectrum1D:
        """
        Compute the spectrum of the molecule based on its atoms and bonds.
        Simple model: sum of atomic spectra, perhaps modified by bonds.
        """
        all_freqs = []
        all_powers = []
        
        # Base atomic spectra
        for atom in self.atoms:
            spec = atom.spectrum()
            all_freqs.extend(spec.omega)
            all_powers.extend(spec.power)
            
        # Bond effects (toy model):
        # Bonds might shift frequencies or add new modes.
        # Here we just add a "bond mode" for each bond
        for idx1, idx2, btype in self.bonds:
            # Bond frequency depends on bond type
            if btype == 'single':
                freq = 1.0
            elif btype == 'double':
                freq = 1.5
            else:
                freq = 0.5
                
            all_freqs.append(freq)
            all_powers.append(0.5) # Arbitrary bond strength
            
        return Spectrum1D(
            omega=np.array(all_freqs),
            power=np.array(all_powers)
        )

@dataclass
class CandidateMaterial:
    name: str
    confidence: float
    predicted_properties: dict

def infer_material_from_ldos(
    ldos_map: np.ndarray,
    atom_db: dict[str, AtomicResonator]
) -> list[CandidateMaterial]:
    """
    Infer potential material composition from LDOS map features.
    
    This is a 'toy' inference engine.
    Real logic would involve matching spectral peaks to atomic/molecular signatures.
    """
    # Extract features from LDOS
    mean_val = np.mean(ldos_map)
    std_val = np.std(ldos_map)
    
    candidates = []
    
    # Toy logic:
    # High mean -> Light atoms (H, C) - higher activity?
    # Low mean -> Heavy atoms (Fe) - lower activity?
    
    # Check against known "materials" (hardcoded for demo)
    
    # 1. "Steel" (Iron-based)
    # Expect low mean (heavy), specific variance
    score_steel = 0.0
    if mean_val < 0.05:
        score_steel += 0.8
    elif mean_val < 0.1:
        score_steel += 0.4
        
    candidates.append(CandidateMaterial(
        name="Steel (Fe-C)",
        confidence=score_steel,
        predicted_properties={"density": "high", "stiffness": "high"}
    ))
    
    # 2. "Water" (H2O)
    # Expect high mean (light atoms), high variance
    score_water = 0.0
    if mean_val > 0.1:
        score_water += 0.7
    if std_val > 0.02:
        score_water += 0.2
        
    candidates.append(CandidateMaterial(
        name="Water (H2O)",
        confidence=score_water,
        predicted_properties={"density": "low", "stiffness": "low"}
    ))
    
    # 3. "Concrete" (Si-O based)
    # Medium mean
    score_concrete = 0.0
    if 0.05 <= mean_val <= 0.15:
        score_concrete += 0.6
        
    candidates.append(CandidateMaterial(
        name="Concrete (Si-O)",
        confidence=score_concrete,
        predicted_properties={"density": "medium", "stiffness": "high"}
    ))
    
    # Sort by confidence
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    
    return candidates

