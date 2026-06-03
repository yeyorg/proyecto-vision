"""
Script para entrenar y guardar el clasificador de sentadillas.
"""
import sys
import os
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from src.squat_classifier import SquatFormClassifier
from src.angle_utils import aggregate_video_features


def gen_synthetic_squat(is_good: bool, n_frames: int = 50):
    """Generar ángulos sintéticos para una sentadilla."""
    frames = []
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


def main():
    np.random.seed(42)

    print("=== Squat Form Classifier Training ===")
    print()

    # Crear clasificador
    print("Creando clasificador basado en reglas biomecánicas...")
    classifier = SquatFormClassifier()
    print(f"  Umbrales: {len(classifier.thresholds)} parámetros")
    print()

    # Generar datos sintéticos
    print("Generando datos sintéticos de verificación...")
    good_frames = gen_synthetic_squat(is_good=True)
    bad_frames = gen_synthetic_squat(is_good=False)
    good_features = aggregate_video_features(good_frames)
    bad_features = aggregate_video_features(bad_frames)
    print(f"  Buena forma: {len(good_frames)} frames -> {len(good_features)} features")
    print(f"  Mala forma:  {len(bad_frames)} frames -> {len(bad_features)} features")
    print()

    # Verificar
    _, overall_good = classifier.score_squat(good_features)
    _, overall_bad = classifier.score_squat(bad_features)
    pred_good = classifier.predict([good_features])[0]
    pred_bad = classifier.predict([bad_features])[0]

    print("=== Verification ===")
    print(f"  Good form score: {overall_good:.1f}/100  predict={pred_good} (expected 0)")
    print(f"  Bad form score:  {overall_bad:.1f}/100  predict={pred_bad} (expected 1)")
    assert pred_good == 0, "Good form should predict 0"
    assert pred_bad == 1, "Bad form should predict 1"
    print("  Classification: OK")
    print()

    # Guardar
    os.makedirs("models", exist_ok=True)
    model_path = "models/squat_form_model.pkl"
    joblib.dump(classifier, model_path)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"Modelo guardado: {model_path} ({size_kb:.1f} KB)")
    print()

    # Verificar carga
    loaded = joblib.load(model_path)
    assert loaded.predict([good_features])[0] == 0
    assert loaded.predict([bad_features])[0] == 1
    print("Carga y verificacion: OK")
    print()
    print("Listo! Ya podes correr:")
    print("  uv run streamlit run app.py")


if __name__ == "__main__":
    main()
