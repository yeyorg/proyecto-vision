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
            detail = f"Rango de movimiento aceptable ({mean_range:.0f}°), podés bajar un poco más"
        elif mean_range >= 20:
            score = 30 + (mean_range - 20) / 20 * 40
            detail = f"Rango de movimiento limitado ({mean_range:.0f}°), intentá bajar más"
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
            feedback.append(f"🔧  Forma necesita trabajo ({overall:.0f}/100) — revisá los puntos críticos")

        return feedback


# Alias para mantener compatibilidad con el notebook original
SquatFormModel = SquatFormClassifier


# ═══════════════════════════════════════════════════════════════
# SquatDetector — máquina de estados para detectar sentadillas
# ═══════════════════════════════════════════════════════════════

from enum import Enum


class SquatPhase(Enum):
    """Fase actual de una sentadilla detectada en tiempo real."""

    UNKNOWN = "---"
    STANDING = "DE PIE"
    DESCENDING = "BAJANDO"
    BOTTOM = "FONDO"
    ASCENDING = "SUBIENDO"
    TRANSITION = "TRANSICION"


# Umbrales para la detección de fase (ángulos de rodilla)
# Basados en biomecánica: de pie ~170°, paralelo ~90°.
_THRESHOLD_DESCEND = 145.0  # por debajo de esto → está bajando
_THRESHOLD_BOTTOM = 115.0  # por debajo de esto → está en el fondo
_THRESHOLD_ASCEND = 140.0  # por encima de esto → casi de pie
_MIN_KNEE_CHANGE = 8.0  # cambio mínimo para considerar movimiento
_WINDOW_SIZE = 5  # frames para suavizar


def _smooth(values: list[float]) -> float:
    """Promedio de los últimos N valores."""
    return float(np.mean(values[-_WINDOW_SIZE:])) if values else 0.0


class SquatDetector:
    """
    Detecta en tiempo real si una persona está haciendo una sentadilla
    y en qué fase se encuentra.

    Usa el ángulo de rodilla como señal principal. La idea es simple:
      - De pie   → rodilla extendida (~170°)
      - Bajando  → rodilla se flexiona (< _THRESHOLD_DESCEND)
      - Fondo    → rodilla flexionada (< _THRESHOLD_BOTTOM)
      - Subiendo → rodilla se extiende (> _THRESHOLD_ASCEND)

    Solo cuando se detecta la fase BOTTOM se activa el análisis de forma.

    Ejemplo de uso:
        detector = SquatDetector()
        for angles in frame_stream:
            phase = detector.update(angles)
            if phase == SquatPhase.BOTTOM:
                score = classifier.score_squat(features)
    """

    def __init__(self):
        self.phase = SquatPhase.UNKNOWN
        self._prev_phase = SquatPhase.UNKNOWN
        self._knee_history: list[float] = []
        self._rep_count = 0
        self._in_rep = False  # True entre DESCENDING y ASCENDING
        self._frames_in_phase = 0

        # Para tracking de forma por repetición
        self.current_rep_angles: list[dict] = []
        self.completed_reps: list[float] = []  # scores de reps completadas
        self.last_rep_score: float | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def update(self, angles: dict[str, float]) -> SquatPhase:
        """
        Procesa un frame de ángulos y actualiza el estado.

        Parameters
        ----------
        angles : dict
            Output de get_squat_angles() para un frame.

        Returns
        -------
        SquatPhase
            Fase actual detectada.
        """
        # Extraer el ángulo de rodilla (promedio left/right)
        knee = self._get_knee_angle(angles)
        if knee is None:
            return self.phase

        self._knee_history.append(knee)
        if len(self._knee_history) > 30:
            self._knee_history.pop(0)

        smooth_knee = _smooth(self._knee_history)

        # Cambio reciente (para detectar movimiento)
        change = self._get_recent_change()
        moving = abs(change) > _MIN_KNEE_CHANGE

        # Máquina de estados (orden importa: evaluar de más específico a menos)
        self._prev_phase = self.phase

        if smooth_knee > _THRESHOLD_DESCEND and not moving:
            self.phase = SquatPhase.STANDING
        elif smooth_knee > _THRESHOLD_ASCEND and change > _MIN_KNEE_CHANGE:
            # Está subiendo pero aún no llegó a standing
            self.phase = SquatPhase.ASCENDING
        elif smooth_knee > _THRESHOLD_DESCEND and change < -_MIN_KNEE_CHANGE:
            # Está bajando desde la posición de pie
            self.phase = SquatPhase.DESCENDING
        elif smooth_knee <= _THRESHOLD_BOTTOM:
            self.phase = SquatPhase.BOTTOM
        elif (
            smooth_knee < _THRESHOLD_ASCEND
            and self._prev_phase == SquatPhase.DESCENDING
        ):
            self.phase = SquatPhase.DESCENDING
        elif (
            smooth_knee < _THRESHOLD_DESCEND
            and self._prev_phase == SquatPhase.ASCENDING
        ):
            self.phase = SquatPhase.ASCENDING
        else:
            self.phase = SquatPhase.TRANSITION

        # Detectar transiciones para contar reps
        self._track_rep(angles)

        # Contar frames en esta fase
        if self.phase == self._prev_phase:
            self._frames_in_phase += 1
        else:
            self._frames_in_phase = 1

        return self.phase

    def reset(self) -> None:
        """Reinicia el detector (nueva sesión)."""
        self.phase = SquatPhase.UNKNOWN
        self._prev_phase = SquatPhase.UNKNOWN
        self._knee_history.clear()
        self._rep_count = 0
        self._in_rep = False
        self._frames_in_phase = 0
        self.current_rep_angles.clear()
        self.completed_reps.clear()
        self.last_rep_score = None

    @property
    def is_active(self) -> bool:
        """Hay una sentadilla en curso? (DESCENDING, BOTTOM, ASCENDING)"""
        return self.phase in (
            SquatPhase.DESCENDING,
            SquatPhase.BOTTOM,
            SquatPhase.ASCENDING,
        )

    @property
    def should_analyze(self) -> bool:
        """
        Momento óptimo para evaluar la forma?
        True cuando estamos en BOTTOM con suficientes frames acumulados.
        """
        if self.phase != SquatPhase.BOTTOM:
            return False
        # Esperar al menos 3 frames en el fondo para estabilizar
        return self._frames_in_phase >= 3

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    @staticmethod
    def _get_knee_angle(angles: dict[str, float]) -> float | None:
        """Promedia left/right knee angle."""
        left = angles.get("left_knee_angle")
        right = angles.get("right_knee_angle")
        vals = [v for v in (left, right) if v is not None]
        return float(np.mean(vals)) if vals else None

    def _get_recent_change(self) -> float:
        """Cambio del ángulo de rodilla en los últimos frames."""
        if len(self._knee_history) < _WINDOW_SIZE + 1:
            return 0.0
        recent = self._knee_history[-_WINDOW_SIZE:]
        return recent[-1] - recent[0]

    def _track_rep(self, angles: dict[str, float]) -> None:
        """
        Cuenta repeticiones basándose en el ángulo de rodilla.

        Una rep = rodilla se flexionó (< umbral fondo) y volvió a extenderse
        (> umbral de pie). Esto es más robusto que depender de transiciones
        exactas de fase.
        """
        knee = self._get_knee_angle(angles)
        if knee is None:
            return

        # Estaba de pie y empezó a bajar
        if not self._in_rep and knee < _THRESHOLD_DESCEND:
            self._in_rep = True
            self.current_rep_angles = [angles]
            return

        # Acumular ángulos durante la rep
        if self._in_rep:
            self.current_rep_angles.append(angles)

        # Terminó la rep: estaba en una rep y volvió a > THRESHOLD_DESCEND
        if self._in_rep and knee > _THRESHOLD_DESCEND and self._prev_phase == SquatPhase.ASCENDING:
            self._in_rep = False
            self._rep_count += 1
        # También terminó si vuelve a > THRESHOLD_DESCEND sin movimiento
        elif self._in_rep and knee > _THRESHOLD_DESCEND + 10:
            self._in_rep = False
            self._rep_count += 1

    def score_current_rep(
        self, classifier: SquatFormClassifier
    ) -> float | None:
        """
        Evalúa la forma de la repetición actual con el clasificador.

        Returns
        -------
        score : float or None
            Puntaje 0-100, o None si no hay suficientes datos.
        """
        if len(self.current_rep_angles) < 3:
            return None

        from src.angle_utils import aggregate_video_features

        features = aggregate_video_features(self.current_rep_angles)
        _, score = classifier.score_squat(features)
        self.last_rep_score = score
        self.completed_reps.append(score)
        return score
