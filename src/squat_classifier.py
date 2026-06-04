"""
Clasificador de forma en sentadilla basado en reglas biomecánicas.

El clasificador evalúa la calidad de una sentadilla usando umbrales sobre
ángulos y métricas extraídas del video. No requiere datos etiquetados para
entrenar — las reglas están basadas en la literatura de biomecánica.

La clase es compatible con `joblib` (implementa predict / predict_proba)
para que pueda serializarse y usarse como un modelo tradicional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CriterionResult:
    """Resultado de un criterio individual de evaluación."""

    name: str
    score: float  # 0–100
    weight: float = 1.0
    detail: str = ""
    passed: bool = True


# Umbrales por defecto (basados en literatura de biomecánica)
DEFAULT_THRESHOLDS: dict[str, Any] = {
    # --- Rodilla ---
    "knee_angle_min_range": (70.0, 110.0),  # rango aceptable para el ángulo mínimo
    "knee_angle_ideal": 90.0,               # valor ideal (paralelo)
    # --- Espalda ---
    "back_angle_mean_range": (15.0, 50.0),  # inclinación del torso aceptable
    "back_angle_ideal": 35.0,
    # --- Knee-over-toe (USA VALOR ABSOLUTO MÁXIMO, no el promedio) ---
    "knee_toe_max_abs": 40.0,  # máxima distancia horizontal (en px)
    # --- Simetría ---
    "symmetry_max": 12.0,  # diferencia máxima aceptable entre lados
    # --- Puntaje de corte ---
    "good_form_threshold": 60.0,  # mínimo para considerar "buena forma"
}


class SquatFormClassifier:
    """
    Clasificador de forma en sentadilla basado en reglas.

    Métricas que evalúa:
    - **Depth**: profundidad de la sentadilla (ángulo mínimo de rodilla)
    - **Back angle**: inclinación del torso
    - **Knee tracking**: desplazamiento horizontal de rodilla
    - **Symmetry**: diferencia entre lado izquierdo y derecho
    - **Stability**: variabilidad (desviación estándar) de las métricas

    Ejemplo de uso:
        classifier = SquatFormClassifier()
        features = aggregate_video_features(frame_angles)
        pred = classifier.predict([features])  # 0=good, 1=needs work
        scores, overall = classifier.score_squat(features)
    """

    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ------------------------------------------------------------------
    # Evaluación por criterio
    # ------------------------------------------------------------------

    def _score_depth(self, features: dict[str, float]) -> CriterionResult:
        """Evalúa la profundidad basada en el ángulo mínimo de rodilla."""
        left_min = features.get("left_knee_angle_min")
        right_min = features.get("right_knee_angle_min")
        values = [v for v in [left_min, right_min] if v is not None]
        if not values:
            return CriterionResult(name="depth", score=0, weight=2, detail="No se detectaron rodillas", passed=False)

        min_knee = min(values)
        lo, hi = self.thresholds["knee_angle_min_range"]
        ideal = self.thresholds["knee_angle_ideal"]

        # Distancia normalizada desde el ideal (0 en ideal, 1 en el borde)
        dist = abs(min_knee - ideal) / (hi - ideal) if min_knee > ideal else abs(min_knee - ideal) / (ideal - lo)

        if lo <= min_knee <= hi:
            # Score: 100 en ideal, baja linealmente a 40 en los bordes
            score = max(40, 100 - dist * 60)
            detail = f"Ángulo mínimo rodilla: {min_knee:.0f}° — dentro del rango ({lo}–{hi}°)"
        elif min_knee < lo:
            # Score: 40 en el borde inferior, baja a 0 en min_knee = lo - 40
            score = max(0, 40 - (lo - min_knee) / 40 * 40)
            detail = f"Ángulo mínimo rodilla: {min_knee:.0f}° — menor a {lo}° (hiperflexión / butt wink)"
        else:
            # Score: 40 en el borde superior, baja a 0 en min_knee = hi + 40
            score = max(0, 40 - (min_knee - hi) / 40 * 40)
            detail = f"Ángulo mínimo rodilla: {min_knee:.0f}° — mayor a {hi}° (poca profundidad)"

        passed = score >= self.thresholds["good_form_threshold"]
        return CriterionResult(
            name="depth", score=round(score, 1), weight=2, detail=detail, passed=passed
        )

    def _score_back_angle(self, features: dict[str, float]) -> CriterionResult:
        """Evalúa la inclinación del torso."""
        back_mean = features.get("back_angle_mean")
        if back_mean is None:
            return CriterionResult(name="back_angle", score=0, weight=1, detail="No se detectó torso", passed=False)

        lo, hi = self.thresholds["back_angle_mean_range"]
        ideal = self.thresholds["back_angle_ideal"]

        if lo <= back_mean <= hi:
            dist = abs(back_mean - ideal) / max(hi - ideal, ideal - lo, 1)
            score = max(40, 100 - dist * 60)
            detail = f"Inclinación torso: {back_mean:.0f}° — dentro del rango ({lo}–{hi}°)"
        elif back_mean < lo:
            score = max(0, 40 - (lo - back_mean) / 20 * 40)
            detail = f"Inclinación torso: {back_mean:.0f}° — muy vertical (<{lo}°)"
        else:
            # hi=50, si back=75 → score=40-(25/30)*40=40-33=7 (MUY inclinado)
            score = max(0, 40 - (back_mean - hi) / 30 * 40)
            detail = f"Inclinación torso: {back_mean:.0f}° — muy inclinado (>{hi}°), riesgo lumbar"

        passed = score >= self.thresholds["good_form_threshold"]
        return CriterionResult(
            name="back_angle", score=round(score, 1), weight=1, detail=detail, passed=passed
        )

    def _score_knee_tracking(self, features: dict[str, float]) -> CriterionResult:
        """Evalúa el desplazamiento de la rodilla sobre el tobillo.
        
        Usa el VALOR ABSOLUTO MÁXIMO (no el promedio) para capturar
        momentos puntuales donde la rodilla se va de控制.
        """
        # Usamos max de valores absolutos para capturar picos
        left_max = features.get("left_knee_toe_x_max")
        left_min = features.get("left_knee_toe_x_min")
        right_max = features.get("right_knee_toe_x_max")
        right_min = features.get("right_knee_toe_x_min")

        peak_values = []
        for v in [left_max, left_min, right_max, right_min]:
            if v is not None:
                peak_values.append(abs(v))

        if not peak_values:
            return CriterionResult(name="knee_tracking", score=100, weight=1, detail="No se pudo evaluar", passed=True)

        max_abs = self.thresholds["knee_toe_max_abs"]
        peak_abs = max(peak_values)

        if peak_abs <= max_abs:
            score = max(70, 100 - (peak_abs / max_abs) * 30)
            detail = f"Máx desplazamiento rodilla: {peak_abs:.0f} px — dentro del rango (±{max_abs} px)"
        elif peak_abs <= max_abs * 1.5:
            score = max(30, 70 - (peak_abs - max_abs) / (max_abs * 0.5) * 40)
            detail = f"Máx desplazamiento rodilla: {peak_abs:.0f} px — excede un poco el límite (±{max_abs} px)"
        else:
            score = max(0, 30 - (peak_abs - max_abs * 1.5) / (max_abs * 0.5) * 30)
            detail = f"Máx desplazamiento rodilla: {peak_abs:.0f} px — EXCEDE el límite (±{max_abs} px)"

        passed = score >= self.thresholds["good_form_threshold"]
        return CriterionResult(
            name="knee_tracking", score=round(score, 1), weight=1, detail=detail, passed=passed
        )

    def _score_symmetry(self, features: dict[str, float]) -> CriterionResult:
        """Evalúa la simetría entre lado izquierdo y derecho."""
        max_sym = self.thresholds["symmetry_max"]
        details: list[str] = []
        scores: list[float] = []

        for key, label in [("knee_symmetry", "rodilla"), ("hip_symmetry", "cadera")]:
            val = features.get(f"{key}_mean")
            if val is not None:
                if val <= max_sym:
                    s = 100 - (val / max_sym) * 30
                    details.append(f"Asimetría {label}: {val:.1f}° (ok)")
                else:
                    s = max(0, 70 - (val - max_sym) * 3)
                    details.append(f"Asimetría {label}: {val:.1f}° — excede {max_sym}°")
                scores.append(s)

        if not scores:
            return CriterionResult(name="symmetry", score=100, weight=1, detail="Simetría no evaluable", passed=True)

        score = float(np.mean(scores))
        detail = "; ".join(details) if details else "Simetría dentro de rango"
        passed = score >= self.thresholds["good_form_threshold"]
        return CriterionResult(
            name="symmetry", score=round(score, 1), weight=1, detail=detail, passed=passed
        )

    def _score_stability(self, features: dict[str, float]) -> CriterionResult:
        """Evalúa control del movimiento.
        
        Mide la suavidad basándose en el rango de movimiento de la rodilla.
        Una sentadilla controlada tiene:
        - Rango de movimiento adecuado (diferencia entre max y min)
        - Sin temblores ni movimientos bruscos
        """
        # Rango de movimiento de rodilla (diferencia entre ángulo de pie y fondo)
        ranges: list[float] = []
        for key in ["left_knee_angle_max", "right_knee_angle_max"]:
            mx = features.get(key)
            mn = features.get(key.replace("_max", "_min"))
            if mx is not None and mn is not None:
                ranges.append(mx - mn)

        if not ranges:
            return CriterionResult(name="stability", score=100, weight=0.5, detail="Estabilidad no evaluable", passed=True)

        mean_range = float(np.mean(ranges))

        # Rango de movimiento para una sentadilla completa: ~70-100° (de ~170° a ~80°)
        if mean_range >= 60:
            score = 100
            detail = f"Buen rango de movimiento ({mean_range:.0f}°) — sentadilla completa"
        elif mean_range >= 40:
            score = 70 + (mean_range - 40) / 20 * 30
            detail = f"Rango de movimiento aceptable ({mean_range:.0f}°), puedes bajar un poco más"
        elif mean_range >= 20:
            score = 30 + (mean_range - 20) / 20 * 40
            detail = f"Rango de movimiento limitado ({mean_range:.0f}°), intenta bajar más"
        else:
            score = max(0, 30 - (20 - mean_range) / 20 * 30)
            detail = f"Rango de movimiento muy limitado ({mean_range:.0f}°), casi no hay flexión de rodilla"

        passed = score >= self.thresholds["good_form_threshold"]
        return CriterionResult(
            name="stability", score=round(score, 1), weight=0.5, detail=detail, passed=passed
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def score_squat(
        self, features: dict[str, float]
    ) -> tuple[list[CriterionResult], float]:
        """
        Evalúa las características de un video y devuelve puntuaciones
        detalladas + puntuación global.

        Returns
        -------
        criteria : list[CriterionResult]
            Resultados individuales por criterio.
        overall : float
            Puntuación global 0–100.
        """
        criteria = [
            self._score_depth(features),
            self._score_back_angle(features),
            self._score_knee_tracking(features),
            self._score_symmetry(features),
            self._score_stability(features),
        ]

        total_weight = sum(c.weight for c in criteria if c.score > 0)
        weighted_sum = sum(c.score * c.weight for c in criteria if c.score > 0)
        overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

        return criteria, overall

    def predict(self, X: np.ndarray | list) -> np.ndarray:
        """
        Interfaz compatible con scikit-learn.
        Retorna 0 (good form) si overall >= threshold, 1 (needs work) en caso contrario.

        Parameters
        ----------
        X : array-like of dicts
            Lista de feature dicts (uno por video/sample).

        Returns
        -------
        np.ndarray
            Array de 0s y 1s.
        """
        threshold = self.thresholds["good_form_threshold"]
        results = []
        for features in X:
            _, overall = self.score_squat(features)
            results.append(0 if overall >= threshold else 1)
        return np.array(results, dtype=np.int32)

    def predict_proba(self, X: np.ndarray | list) -> np.ndarray:
        """
        Interfaz compatible con scikit-learn.
        Retorna probabilidades aproximadas [good_prob, bad_prob].

        Parameters
        ----------
        X : array-like of dicts
            Lista de feature dicts (uno por video/sample).

        Returns
        -------
        np.ndarray
            Array de shape (n_samples, 2).
        """
        probas = []
        for features in X:
            _, overall = self.score_squat(features)
            good_prob = overall / 100.0
            probas.append([good_prob, 1.0 - good_prob])
        return np.array(probas, dtype=np.float64)

    def get_feedback(self, features: dict[str, float]) -> list[str]:
        """
        Genera una lista de consejos legibles basados en la evaluación.

        Parameters
        ----------
        features : dict
            Feature dict de un video.

        Returns
        -------
        list[str]
            Consejos prioritarios (primero los más críticos).
        """
        criteria, overall = self.score_squat(features)
        feedback: list[str] = []

        # Primero los que no pasaron
        for c in sorted(criteria, key=lambda x: x.score):
            if not c.passed:
                feedback.append(f"⚠️  {c.detail}")
            elif c.score < 80:
                feedback.append(f"ℹ️  {c.detail}")

        if overall >= 80:
            feedback.append(f"✅  ¡Buena forma! Puntuación global: {overall:.0f}/100")
        elif overall >= self.thresholds["good_form_threshold"]:
            feedback.append(f"📈  Forma aceptable ({overall:.0f}/100) — hay margen de mejora")
        else:
            feedback.append(f"🔧  Forma necesita trabajo ({overall:.0f}/100) — revisa los puntos críticos")

        return feedback


# Alias para mantener compatibilidad con el notebook original
SquatFormModel = SquatFormClassifier
