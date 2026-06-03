"""
Tests para el clasificador de sentadillas basado en reglas biomecánicas.

Verifica que cada criterio (depth, back_angle, knee_tracking, symmetry, stability)
produce scores coherentes para casos buenos y malos, y que el clasificador
completo funciona correctamente (predict, predict_proba, serialización).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

# Asegurar que podemos importar desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.angle_utils import aggregate_video_features
from src.squat_classifier import SquatFormClassifier


# ── Helpers ──────────────────────────────────────────────────────────────────


def gen_synthetic_squat(is_good: bool, n_frames: int = 50) -> list[dict]:
    """Generar ángulos sintéticos para una sentadilla.

    Parameters
    ----------
    is_good : bool
        ``True`` → simula buena forma (depth, back angle, simetría, knee tracking
        dentro de rangos óptimos).
        ``False`` → simula mala forma (todo fuera de rango).
    n_frames : int
        Cantidad de frames a generar.

    Returns
    -------
    list[dict]
        Lista de diccionarios con ángulos por frame, compatible con
        :func:`aggregate_video_features`.
    """
    frames: list[dict] = []
    for i in range(n_frames):
        phase = i / n_frames
        depth = np.sin(phase * np.pi)

        if is_good:
            knee_min = 85 + np.random.uniform(-5, 5)
            back = 30 + depth * 8 + np.random.uniform(-2, 2)
            sym = np.random.uniform(0, 4)
            ktoe = np.random.uniform(-5, 15)
        else:
            knee_min = 125 + np.random.uniform(-5, 5)
            back = 50 + depth * 25 + np.random.uniform(-5, 5)
            sym = np.random.uniform(12, 22)
            ktoe = np.random.uniform(-25, 45)

        knee = 170 - (170 - knee_min) * depth + np.random.uniform(-3, 3)

        frames.append({
            "left_knee_angle": knee,
            "right_knee_angle": knee + sym,
            "left_hip_angle": 120 - depth * 40 + np.random.uniform(-2, 2),
            "right_hip_angle": 120 - depth * 40 + sym * 0.5,
            "back_angle": back,
            "left_knee_toe_x": ktoe,
            "right_knee_toe_x": ktoe + np.random.uniform(-3, 3),
            "knee_symmetry": abs(sym),
            "hip_symmetry": abs(sym * 0.7),
        })

    return frames


def gen_both() -> tuple[dict, dict]:
    """Generar features para un caso bueno y uno malo.

    Returns
    -------
    good_features, bad_features : tuple[dict, dict]
    """
    np.random.seed(42)
    good_features = aggregate_video_features(gen_synthetic_squat(is_good=True))
    bad_features = aggregate_video_features(gen_synthetic_squat(is_good=False))
    return good_features, bad_features


# ── Tests del clasificador completo ──────────────────────────────────────────


class TestEndToEnd:
    """Tests de integración: el clasificador completo con datos sintéticos."""

    def test_good_form_predicts_0(self):
        """Buena forma sintética debe clasificarse como 0 (good)."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        assert clf.predict([good_f])[0] == 0

    def test_bad_form_predicts_1(self):
        """Mala forma sintética debe clasificarse como 1 (needs work)."""
        _, bad_f = gen_both()
        clf = SquatFormClassifier()
        assert clf.predict([bad_f])[0] == 1

    def test_good_form_overall_above_threshold(self):
        """Score global de buena forma debe superar el threshold."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        _, overall = clf.score_squat(good_f)
        assert overall >= clf.thresholds["good_form_threshold"]

    def test_bad_form_overall_below_threshold(self):
        """Score global de mala forma debe estar por debajo del threshold."""
        _, bad_f = gen_both()
        clf = SquatFormClassifier()
        _, overall = clf.score_squat(bad_f)
        assert overall < clf.thresholds["good_form_threshold"]

    def test_predict_proba_shape(self):
        """predict_proba debe devolver shape (n_samples, 2)."""
        good_f, bad_f = gen_both()
        clf = SquatFormClassifier()
        probas = clf.predict_proba([good_f, bad_f])
        assert probas.shape == (2, 2)
        # Las probabilidades deben sumar ~1
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_predict_proba_good_form_higher(self):
        """Buena forma debe tener proba[0] > proba[1] (good > needs work)."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        proba = clf.predict_proba([good_f])[0]
        assert proba[0] > proba[1]

    def test_predict_proba_bad_form_lower(self):
        """Mala forma debe tener proba[0] < proba[1] (good < needs work)."""
        _, bad_f = gen_both()
        clf = SquatFormClassifier()
        proba = clf.predict_proba([bad_f])[0]
        assert proba[0] < proba[1]

    def test_get_feedback_returns_list(self):
        """get_feedback debe devolver una lista con al menos 1 elemento."""
        good_f, bad_f = gen_both()
        clf = SquatFormClassifier()
        fb_good = clf.get_feedback(good_f)
        fb_bad = clf.get_feedback(bad_f)
        assert isinstance(fb_good, list) and len(fb_good) >= 1
        assert isinstance(fb_bad, list) and len(fb_bad) >= 1
        # El feedback malo debe mencionar problemas
        assert any("⚠️" in f or "🔧" in f for f in fb_bad)


# ── Tests por criterio individual ────────────────────────────────────────────


class TestCriteria:
    """Cada criterio del clasificador se comporta correctamente."""

    def test_depth_good_form_scores_high(self):
        """Buena profundidad (knee ~85°) debe dar score alto."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_depth(good_f)
        assert crit.score >= 60
        assert "dentro del rango" in crit.detail.lower()

    def test_depth_bad_form_scores_low(self):
        """Mala profundidad (knee ~125°) debe dar score bajo."""
        _, bad_f = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_depth(bad_f)
        assert crit.score < 60
        assert "poca profundidad" in crit.detail.lower()

    def test_back_angle_good_form_scores_high(self):
        """Espalda en rango normal debe dar score alto."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_back_angle(good_f)
        assert crit.score >= 60

    def test_back_angle_bad_form_scores_low(self):
        """Espalda muy inclinada debe dar score bajo."""
        _, bad_f = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_back_angle(bad_f)
        assert crit.score < 60

    def test_knee_tracking_good_form_scores_high(self):
        """Rodilla estable debe dar score alto."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_knee_tracking(good_f)
        assert crit.score >= 60

    def test_knee_tracking_bad_form_scores_low(self):
        """Rodilla muy adelantada debe dar score bajo."""
        _, bad_f = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_knee_tracking(bad_f)
        assert crit.score < 60

    def test_symmetry_good_form_scores_high(self):
        """Simetría baja (casi iguales) debe dar score alto."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_symmetry(good_f)
        assert crit.score >= 60

    def test_symmetry_bad_form_scores_lower_than_good(self):
        """Asimetría alta debe dar score menor que buena forma."""
        good_f, bad_f = gen_both()
        clf = SquatFormClassifier()
        good_crit = clf._score_symmetry(good_f)
        bad_crit = clf._score_symmetry(bad_f)
        assert bad_crit.score < good_crit.score
        assert "excede" in bad_crit.detail.lower()

    def test_stability_good_form_scores_high(self):
        """Buen rango de movimiento (≥60°) debe dar score alto."""
        good_f, _ = gen_both()
        clf = SquatFormClassifier()
        crit = clf._score_stability(good_f)
        assert crit.score >= 60
        assert "completa" in crit.detail.lower()

    def test_stability_bad_form_scores_lower_than_good(self):
        """Rango de movimiento de mala forma debe ser menor que buena forma."""
        good_f, bad_f = gen_both()
        clf = SquatFormClassifier()
        good_crit = clf._score_stability(good_f)
        bad_crit = clf._score_stability(bad_f)
        assert bad_crit.score < good_crit.score
        assert "Buen rango de movimiento" in good_crit.detail


# ── Tests de casos borde ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Comportamiento del clasificador ante entradas inesperadas."""

    def test_empty_features_does_not_crash(self):
        """Features vacío no debe lanzar excepción."""
        clf = SquatFormClassifier()
        features: dict = {}
        criteria, overall = clf.score_squat(features)
        assert isinstance(criteria, list)
        assert len(criteria) == 5
        assert isinstance(overall, float)
        # Sin datos, algunos criterios devuelven 100 por default (no evaluable)

    def test_predict_with_empty_features(self):
        """predict con features vacío no debe lanzar excepción."""
        clf = SquatFormClassifier()
        pred = clf.predict([{}])
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (1,)
        # Sin datos, predict devuelve 0 porque los fallbacks son generosos

    def test_partial_features_does_not_crash(self):
        """Solo algunas claves presentes no debe romperse."""
        clf = SquatFormClassifier()
        partial = {"left_knee_angle_min": 80.0, "left_knee_angle_max": 170.0}
        criteria, overall = clf.score_squat(partial)
        assert isinstance(criteria, list)
        assert isinstance(overall, float)

    def test_predict_empty_list(self):
        """Lista vacía en predict debe devolver array vacío."""
        clf = SquatFormClassifier()
        result = clf.predict([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_single_frame_squat_does_not_crash(self):
        """Un solo frame (sin variación) no debe romper aggregate."""
        frames = [{
            "left_knee_angle": 90.0,
            "right_knee_angle": 92.0,
            "left_hip_angle": 100.0,
            "right_hip_angle": 102.0,
            "back_angle": 35.0,
            "left_knee_toe_x": 5.0,
            "right_knee_toe_x": 3.0,
            "knee_symmetry": 2.0,
            "hip_symmetry": 2.0,
        }]
        features = aggregate_video_features(frames)
        clf = SquatFormClassifier()
        criteria, overall = clf.score_squat(features)
        assert len(criteria) == 5
        assert isinstance(overall, float)


# ── Tests de serialización ───────────────────────────────────────────────────


class TestSerialization:
    """El clasificador debe poder serializarse con joblib."""

    def test_joblib_roundtrip(self):
        """Guardar y cargar con joblib debe mantener el comportamiento."""
        good_f, bad_f = gen_both()
        clf = SquatFormClassifier()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            joblib.dump(clf, tmp_path)
            loaded = joblib.load(tmp_path)

            assert loaded.predict([good_f])[0] == 0
            assert loaded.predict([bad_f])[0] == 1

            _, overall_good = loaded.score_squat(good_f)
            _, overall_bad = loaded.score_squat(bad_f)
            assert overall_good >= clf.thresholds["good_form_threshold"]
            assert overall_bad < clf.thresholds["good_form_threshold"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_default_thresholds_preserved(self):
        """Los thresholds por defecto no se pierden al serializar."""
        clf = SquatFormClassifier()
        expected = clf.thresholds

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            joblib.dump(clf, tmp_path)
            loaded = joblib.load(tmp_path)
            assert loaded.thresholds == expected
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ── Tests del helper sintético ───────────────────────────────────────────────


class TestSyntheticHelper:
    """El helper gen_synthetic_squat genera datos coherentes."""

    def test_good_squat_generates_correct_number_of_frames(self):
        """gen_synthetic_squat debe generar exactamente n_frames frames."""
        frames = gen_synthetic_squat(is_good=True, n_frames=100)
        assert len(frames) == 100

    def test_frames_have_required_keys(self):
        """Cada frame debe tener todas las claves necesarias."""
        required_keys = {
            "left_knee_angle", "right_knee_angle", "left_hip_angle",
            "right_hip_angle", "back_angle", "left_knee_toe_x",
            "right_knee_toe_x", "knee_symmetry", "hip_symmetry",
        }
        frames = gen_synthetic_squat(is_good=True, n_frames=10)
        for f in frames:
            assert set(f.keys()) == required_keys

    def test_good_squat_knee_deeper_than_bad(self):
        """Buena forma debe tener menor ángulo de rodilla (más profunda)."""
        np.random.seed(42)
        good = gen_synthetic_squat(is_good=True)
        np.random.seed(42)
        bad = gen_synthetic_squat(is_good=False)
        good_min = min(f["left_knee_angle"] for f in good)
        bad_min = min(f["left_knee_angle"] for f in bad)
        assert good_min < bad_min

    def test_good_squat_symmetry_lower_than_bad(self):
        """Buena forma debe tener menos asimetría."""
        np.random.seed(42)
        good = gen_synthetic_squat(is_good=True)
        np.random.seed(42)
        bad = gen_synthetic_squat(is_good=False)
        good_sym = np.mean([f["knee_symmetry"] for f in good])
        bad_sym = np.mean([f["knee_symmetry"] for f in bad])
        assert good_sym < bad_sym

    def test_features_have_expected_keys(self):
        """aggregate_video_features debe producir las 36 features esperadas."""
        frames = gen_synthetic_squat(is_good=True, n_frames=50)
        features = aggregate_video_features(frames)
        # 9 métricas × 4 stats (mean, std, min, max) = 36
        assert len(features) == 36
