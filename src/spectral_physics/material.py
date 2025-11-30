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
    
    def distance_cosine(self, other: Spectrum1D) -> float:
        """
        Косинусная "дистанция" между нормированными спектрами:
            1 - (⟨a,b⟩ / (||a|| * ||b||))
        
        Args:
            other: Спектр для сравнения.
        
        Returns:
            Косинусное расстояние (0..1 для неотрицательных спектров).
        """
        if not np.array_equal(self.reference.omega, other.omega):
            raise ValueError("Frequency grids do not match")
        
        # Работаем с векторами мощности
        a = self.reference.power
        b = other.power
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            # Если один из векторов нулевой, расстояние неопределено (или макс)
            return 1.0
            
        cosine_similarity = np.dot(a, b) / (norm_a * norm_b)
        
        # Ограничиваем [0, 1] для стабильности
        cosine_similarity = np.clip(cosine_similarity, 0.0, 1.0)
        
        return 1.0 - cosine_similarity

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


@dataclass
class HealthProfile:
    """
    Эталонный профиль "здорового" состояния для нескольких каналов.
    """
    signatures: dict[str, MaterialSignature]  # имя канала -> подпись
    feature_signatures: dict[str, FeatureSignature] | None = None

    def score(self, current: dict[str, Spectrum1D]) -> dict[str, float]:
        """
        Вернуть словарь name -> distance_l2 для каждого канала.
        Канал присутствует в signatures и в current.
        """
        scores = {}
        for name, signature in self.signatures.items():
            if name in current:
                scores[name] = signature.distance_l2(current[name])
        return scores

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
        from .diagnostics import extract_features
        
        if self.feature_signatures is None:
            return {}
            
        scores = {}
        for name, feat_sig in self.feature_signatures.items():
            if name in current and name in bands_hz:
                spec = current[name]
                bands = bands_hz[name]
                feats = extract_features(spec, bands)
                scores[name] = feat_sig.distance_l2(feats)
        return scores

    def is_anomalous(
        self,
        current: dict[str, Spectrum1D],
        thresholds: dict[str, float],
    ) -> dict[str, bool]:
        """
        Вернуть словарь name -> bool (аномален / нет)
        по индивидуальным порогам для каждого канала.
        """
        results = {}
        for name, signature in self.signatures.items():
            if name in current and name in thresholds:
                results[name] = signature.is_anomalous(current[name], thresholds[name])
        return results


