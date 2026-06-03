"""
Verificación completa del proyecto Squat Form Analyzer.
"""
import sys
import os

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
from src.pose_extractor import PoseExtractor
from src.angle_utils import aggregate_video_features
from src.squat_classifier import SquatFormClassifier


def main():
    print("=" * 50)
    print("Squat Form Analyzer — Verificacion")
    print("=" * 50)

    # 1. Modelo
    m = joblib.load("models/squat_form_model.pkl")
    print(f"[OK] Modelo cargado: {type(m).__name__}")

    # 2. Clasificacion
    good = {
        "left_knee_angle_min": 85,
        "left_knee_angle_max": 170,
        "right_knee_angle_min": 88,
        "right_knee_angle_max": 169,
        "left_knee_toe_x_min": -5,
        "left_knee_toe_x_max": 15,
        "right_knee_toe_x_min": -3,
        "right_knee_toe_x_max": 12,
        "left_knee_angle_std": 30,
        "right_knee_angle_std": 28,
        "back_angle_mean": 35,
        "left_knee_angle_mean": 130,
        "right_knee_angle_mean": 132,
        "knee_symmetry_mean": 3,
        "hip_symmetry_mean": 2,
    }
    bad = dict(good)
    bad.update(
        {
            "left_knee_angle_min": 125,
            "right_knee_angle_min": 130,
            "back_angle_mean": 65,
            "left_knee_toe_x_max": 55,
        }
    )

    assert m.predict([good])[0] == 0, "Buena forma debe predecir 0"
    assert m.predict([bad])[0] == 1, "Mala forma debe predecir 1"
    _, gs = m.score_squat(good)
    _, bs = m.score_squat(bad)
    print(f"[OK] Buena forma: {gs:.1f}/100  -> predict 0")
    print(f"[OK] Mala forma:  {bs:.1f}/100  -> predict 1")

    # 3. Feedback
    fb = m.get_feedback(bad)
    print(f"[OK] Feedback generado: {len(fb)} items")
    for f in fb:
        print(f"     {f[:90]}")

    # 4. YOLO
    p = PoseExtractor()
    print(f"[OK] YOLO pose: {p.model.model_name}")

    # 5. Features
    fa = aggregate_video_features([{"left_knee_angle": 90, "right_knee_angle": 92}])
    print(f"[OK] Feature aggregation: {len(fa)} features")

    # 6. Notebook valido
    nb_path = os.path.join(os.path.dirname(__file__), "..", "squat_form.ipynb")
    import json

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    print(f"[OK] Notebook: {len(nb['cells'])} celdas")

    print()
    print("=" * 50)
    print("TODO OK — el proyecto esta listo para usar")
    print("=" * 50)


if __name__ == "__main__":
    main()
