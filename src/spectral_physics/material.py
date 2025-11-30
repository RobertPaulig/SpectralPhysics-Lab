import numpy as np
from dataclasses import dataclass
from .spectrum import Spectrum1D


@dataclass
class MaterialSignature:
    """
    Спектральная подпись материала/узла.
    
    Содержит:
    - опорный спектр (reference) в нормальном состоянии,
    - простую метрику "расстояния" до нового спектра.
    
    Используется для детекции дефектов/аномалий по изменению
    спектральных характеристик.
    """
    reference: Spectrum1D
    
    def distance_l2(self, other: Spectrum1D) -> float:
        """
        L2-норма между нормированными спектрами.
        
        Args:
            other: Спектр для сравнения с reference.
        
        Returns:
            L2-расстояние между нормированными спектрами.
        
        Raises:
            ValueError: Если сетки по частоте не совпадают.
        
        Notes:
            Спектры нормируются перед сравнением, чтобы различия
            в амплитуде не влияли на детекцию изменения формы спектра.
        """
        # Проверка совпадения частотных сеток
        if not np.array_equal(self.reference.omega, other.omega):
            raise ValueError(
                "Frequency grids do not match. "
                f"Reference has {len(self.reference.omega)} points, "
                f"other has {len(other.omega)} points. "
                "Cannot compute distance."
            )
        
        # Нормируем оба спектра
        ref_normalized = self.reference.normalize()
        other_normalized = other.normalize()
        
        # Вычисляем L2-расстояние
        diff = ref_normalized.power - other_normalized.power
        distance = np.sqrt(np.sum(diff ** 2))
        
        return float(distance)
    
    def is_anomalous(self, other: Spectrum1D, threshold: float) -> bool:
        """
        Вернуть True, если distance_l2(other) > threshold.
        
        Args:
            other: Спектр для проверки.
            threshold: Порог аномальности.
        
        Returns:
            True если спектр аномален, False иначе.
        
        Raises:
            ValueError: Если сетки по частоте не совпадают.
        
        Example:
            >>> sig = MaterialSignature(reference=normal_spectrum)
            >>> if sig.is_anomalous(test_spectrum, threshold=0.1):
            ...     print("Обнаружена аномалия!")
        """
        distance = self.distance_l2(other)
        return distance > threshold
