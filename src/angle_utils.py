"""
Cálculos de ángulos biomecánicos específicos para sentadillas.

Usa los keypoints de COCO (17 keypoints) para computar:
  - Ángulo de rodilla (cadera → rodilla → tobillo)
  - Ángulo de cadera (hombro → cadera → rodilla)
  - Ángulo de espalda / inclinación del torso
  - Proyección de rodilla sobre tobillo (knee-over-toe)
  - Simetría entre lado izquierdo y derecho
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constantes de keypoints COCO
# ---------------------------------------------------------------------------
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


# ---------------------------------------------------------------------------
# Funciones geométricas base
# ---------------------------------------------------------------------------

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Ángulo (en grados) entre dos vectores."""
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-8:
        return 90.0
    cos_angle = max(-1.0, min(1.0, dot / norm))
    return float(np.degrees(np.arccos(cos_angle)))


def angle_3pt(p1: tuple[float, float],
              p2: tuple[float, float],
              p3: tuple[float, float]) -> float:
    """
    Ángulo P1–P2–P3 con vértice en P2 (en grados).

    Ejemplo: angle_3pt(cadera, rodilla, tobillo) → ángulo de rodilla.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    return angle_between(v1, v2)


# ---------------------------------------------------------------------------
# Extracción de ángulos específicos para sentadilla
# ---------------------------------------------------------------------------

SQUAT_ANGLE_KEYS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "back_angle",
    "left_knee_toe_x",
    "right_knee_toe_x",
    "knee_symmetry",
    "hip_symmetry",
]


def get_squat_angles(kps: dict[int, dict[str, float]]) -> dict[str, float]:
    """
    A partir de un dict {id: {x, y, confidence}} extrae todas las
    métricas relevantes para una sentadilla.

    Parameters
    ----------
    kps : dict
        Ejemplo: {5: {"x": 100, "y": 200, "confidence": 0.95}, ...}

    Returns
    -------
    dict[str, float]
        Métricas computadas. Solo incluye las que se pudieron calcular
        (depende de qué keypoints estén visibles).
    """
    angles: dict[str, float] = {}
    missing: list[str] = []

    def _pt(idx: int) -> np.ndarray | None:
        if idx not in kps:
            return None
        return np.array([kps[idx]["x"], kps[idx]["y"]])

    # ---------------------------------------------------------------
    # 1. Ángulo de rodilla (cadera → rodilla → tobillo)
    # ---------------------------------------------------------------
    for side, hip_id, knee_id, ankle_id in [
        ("left", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        ("right", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ]:
        hip = _pt(hip_id)
        knee = _pt(knee_id)
        ankle = _pt(ankle_id)
        if hip is not None and knee is not None and ankle is not None:
            angles[f"{side}_knee_angle"] = angle_3pt(
                (float(hip[0]), float(hip[1])),
                (float(knee[0]), float(knee[1])),
                (float(ankle[0]), float(ankle[1])),
            )
        else:
            missing.append(f"{side}_knee")

    # ---------------------------------------------------------------
    # 2. Ángulo de cadera (hombro → cadera → rodilla)
    # ---------------------------------------------------------------
    for side, shoulder_id, hip_id, knee_id in [
        ("left", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
        ("right", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ]:
        shoulder = _pt(shoulder_id)
        hip = _pt(hip_id)
        knee = _pt(knee_id)
        if shoulder is not None and hip is not None and knee is not None:
            angles[f"{side}_hip_angle"] = angle_3pt(
                (float(shoulder[0]), float(shoulder[1])),
                (float(hip[0]), float(hip[1])),
                (float(knee[0]), float(knee[1])),
            )
        else:
            missing.append(f"{side}_hip")

    # ---------------------------------------------------------------
    # 3. Ángulo de espalda (torso vs vertical)
    # ---------------------------------------------------------------
    left_shoulder = _pt(LEFT_SHOULDER)
    right_shoulder = _pt(RIGHT_SHOULDER)
    left_hip = _pt(LEFT_HIP)
    right_hip = _pt(RIGHT_HIP)

    if all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        mid_hip = (left_hip + right_hip) / 2.0
        torso_vec = mid_shoulder - mid_hip  # apunta hacia arriba
        vertical = np.array([0.0, -1.0])  # eje Y apunta hacia arriba en imagen
        angles["back_angle"] = angle_between(torso_vec, vertical)
    else:
        missing.append("back_angle")

    # ---------------------------------------------------------------
    # 4. Knee-over-toe (distancia horizontal rodilla → tobillo)
    # ---------------------------------------------------------------
    for side, knee_id, ankle_id in [
        ("left", LEFT_KNEE, LEFT_ANKLE),
        ("right", RIGHT_KNEE, RIGHT_ANKLE),
    ]:
        knee = _pt(knee_id)
        ankle = _pt(ankle_id)
        if knee is not None and ankle is not None:
            angles[f"{side}_knee_toe_x"] = float(knee[0] - ankle[0])
        else:
            missing.append(f"{side}_knee_toe")

    # ---------------------------------------------------------------
    # 5. Simetría
    # ---------------------------------------------------------------
    if "left_knee_angle" in angles and "right_knee_angle" in angles:
        angles["knee_symmetry"] = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
    if "left_hip_angle" in angles and "right_hip_angle" in angles:
        angles["hip_symmetry"] = abs(angles["left_hip_angle"] - angles["right_hip_angle"])

    return angles


# ---------------------------------------------------------------------------
# Features agregadas por video
# ---------------------------------------------------------------------------

def aggregate_video_features(
    frame_angles: list[dict[str, float]],
) -> dict[str, float]:
    """
    Toma una lista de dicts (uno por frame, output de get_squat_angles)
    y calcula estadísticos agregados para todo el video.

    Returns
    -------
    dict[str, float]
        features con sufijos _mean, _std, _min, _max para cada métrica.
    """
    if not frame_angles:
        return {}

    # Reunir todas las keys que aparecen en al menos un frame
    all_keys: set[str] = set()
    for fa in frame_angles:
        all_keys.update(fa.keys())

    features: dict[str, float] = {}
    for key in sorted(all_keys):
        values = [fa[key] for fa in frame_angles if key in fa]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        features[f"{key}_mean"] = float(np.mean(arr))
        features[f"{key}_std"] = float(np.std(arr))
        features[f"{key}_min"] = float(np.min(arr))
        features[f"{key}_max"] = float(np.max(arr))

    return features


# ---------------------------------------------------------------------------
# Ángulos de referencia para diagnóstico
# ---------------------------------------------------------------------------

ANGLE_EXPLANATIONS: dict[str, str] = {
    "left_knee_angle": (
        "Ángulo de la rodilla izquierda (cadera→rodilla→tobillo). "
        "En el punto más bajo de la sentadilla debería estar entre 70° y 110°. "
        ">110° = no llegaste a paralelo. <60° = hiperflexión / butt wink."
    ),
    "right_knee_angle": (
        "Ángulo de la rodilla derecha. Mismos rangos que la izquierda."
    ),
    "left_hip_angle": (
        "Ángulo de cadera izquierda (hombro→cadera→rodilla). "
        "Indica cuánto se abre la cadera en el descenso."
    ),
    "right_hip_angle": (
        "Ángulo de cadera derecha."
    ),
    "back_angle": (
        "Inclinación del torso respecto a la vertical. "
        "20°–40° es normal en sentadilla con barra alta; 40°–60° en barra baja. "
        ">60° = excesiva inclinación hacia adelante (riesgo lumbar)."
    ),
    "left_knee_toe_x": (
        "Distancia horizontal rodilla→tobillo (izquierda). "
        "Valores muy positivos = rodilla pasa mucho la punta del pie (carga excesiva en rodilla). "
        "Valores negativos = rodilla atrás del tobillo (sentadilla incompleta)."
    ),
    "right_knee_toe_x": (
        "Distancia horizontal rodilla→tobillo (derecha)."
    ),
    "knee_symmetry": (
        "Diferencia absoluta entre ángulo de rodilla izquierda y derecha. "
        ">15° indica asimetría significativa."
    ),
    "hip_symmetry": (
        "Diferencia absoluta entre ángulo de cadera izquierda y derecha. "
        ">15° indica asimetría en la cadera."
    ),
}
