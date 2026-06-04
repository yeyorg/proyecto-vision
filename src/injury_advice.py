"""
Riesgos de lesión y recomendaciones según los puntajes de la sentadilla.

OBJETIVO
--------
Traducir los puntajes biomecánicos (depth, back_angle, knee_tracking,
symmetry, stability) en información accionable para el usuario:
  - a qué tipo de lesiones lo expone cada problema detectado, y
  - una recomendación verificable para corregirlo.

No reemplaza la evaluación de un profesional de la salud; es orientativo.
"""

from __future__ import annotations

# Umbral por debajo del cual un criterio se considera "con riesgo".
RISK_THRESHOLD = 70.0

# Aviso obligatorio que la interfaz debe mostrar junto a las recomendaciones.
DISCLAIMER = (
    "Información orientativa basada en biomecánica. "
    "No reemplaza la evaluación de un profesional de la salud."
)


def _severity(score: float) -> str:
    """Clasifica la severidad del riesgo según el puntaje del criterio."""
    if score < 40:
        return "Alta"
    if score < 60:
        return "Moderada"
    return "Leve"


def get_injury_assessment(criteria, features: dict) -> list[dict]:
    """
    Genera la evaluación de riesgo de lesión a partir de los criterios.

    Parameters
    ----------
    criteria : list[CriterionResult]
        Resultado de ``SquatFormClassifier.score_squat`` (cada uno con
        atributos ``name`` y ``score``).
    features : dict
        Features agregadas del video (para distinguir sub-casos, p. ej.
        sentadilla demasiado profunda vs. poco profunda).

    Returns
    -------
    list[dict]
        Una entrada por criterio con riesgo, ordenada de mayor a menor
        severidad. Cada dict: {zona, criterio, score, severidad,
        lesiones, recomendacion}.
    """
    assessments: list[dict] = []

    for c in criteria:
        if c.score >= RISK_THRESHOLD:
            continue  # criterio sano, no genera advertencia

        entry = _assess_criterion(c, features)
        if entry:
            entry["severidad"] = _severity(c.score)
            entry["score"] = round(c.score, 0)
            assessments.append(entry)

    # Más severo primero (menor score primero)
    assessments.sort(key=lambda e: e["score"])
    return assessments


def _assess_criterion(c, features: dict) -> dict | None:
    """Devuelve el riesgo/recomendación para un criterio según su sub-caso."""
    name = c.name

    if name == "depth":
        left = features.get("left_knee_angle_min")
        right = features.get("right_knee_angle_min")
        vals = [v for v in (left, right) if v is not None]
        min_knee = min(vals) if vals else None
        if min_knee is not None and min_knee < 70:
            return {
                "zona": "Zona lumbar",
                "criterio": "Profundidad excesiva (butt wink)",
                "lesiones": "Flexión lumbar bajo carga: sobrecarga de los discos "
                            "intervertebrales y riesgo de lumbalgia/lesión de disco.",
                "recomendacion": "Controla la bajada y frena antes de que la pelvis "
                                 "se 'enrolle' en el fondo. Trabaja movilidad de "
                                 "tobillo y cadera para mantener la espalda neutra.",
            }
        return {
            "zona": "Rodilla / efectividad",
            "criterio": "Poca profundidad (no llega a paralelo)",
            "lesiones": "Sentadilla parcial: menor activación de glúteos e isquios "
                        "y posible sobrecarga de cuádriceps y rodilla por compensación.",
            "recomendacion": "Baja de forma controlada hasta que el muslo quede "
                             "paralelo al piso, si tu movilidad lo permite.",
        }

    if name == "back_angle":
        back = features.get("back_angle_mean")
        if back is not None and back < 15:
            return {
                "zona": "Rodilla",
                "criterio": "Torso demasiado vertical",
                "lesiones": "Torso muy erguido traslada carga excesiva a la rodilla.",
                "recomendacion": "Permite una ligera inclinación natural del torso "
                                 "llevando la cadera hacia atrás al bajar.",
            }
        return {
            "zona": "Zona lumbar",
            "criterio": "Inclinación excesiva del torso",
            "lesiones": "Mayor carga compresiva y de cizalla en la columna lumbar: "
                        "riesgo de lumbalgia y lesión de disco.",
            "recomendacion": "Mantén el pecho más alto y el core firme. Revisa que "
                             "el peso no te lleve hacia adelante; refuerza la "
                             "musculatura del tronco.",
        }

    if name == "knee_tracking":
        return {
            "zona": "Rodilla (femoropatelar)",
            "criterio": "Rodilla se desplaza demasiado adelante",
            "lesiones": "Estrés en la articulación femoropatelar y el tendón "
                        "rotuliano: riesgo de tendinitis y dolor anterior de rodilla.",
            "recomendacion": "Inicia el movimiento llevando la cadera hacia atrás "
                             "(bisagra) y reparte el peso en todo el pie, no en la "
                             "punta.",
        }

    if name == "symmetry":
        return {
            "zona": "Rodilla / cadera (lado dominante)",
            "criterio": "Asimetría entre piernas",
            "lesiones": "Descompensación que sobrecarga el lado dominante: riesgo "
                        "de lesión por carga desigual en rodilla o cadera.",
            "recomendacion": "Revisa fuerza y movilidad del lado más débil. Suma "
                             "trabajo unilateral (p. ej. sentadilla búlgara) para "
                             "equilibrar.",
        }

    if name == "stability":
        return {
            "zona": "Control / efectividad",
            "criterio": "Rango de movimiento incompleto o inestable",
            "lesiones": "Recorrido parcial y posibles compensaciones: menor "
                        "efectividad y control del movimiento bajo carga.",
            "recomendacion": "Trabaja el rango completo de forma lenta y "
                             "controlada. Refuerza el core y la estabilidad "
                             "antes de subir carga.",
        }

    return None
