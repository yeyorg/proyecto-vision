"""
Modo tiempo real — analizá tu sentadilla en vivo desde la webcam.

Usa SquatDetector para saber si estás haciendo una sentadilla y solo
entonces evalúa la forma con SquatFormClassifier.

Controles:
  q  → salir
  r  → resetear contador de reps
  p  → pausar/reanudar análisis

Uso:
  uv run python realtime.py
  uv run python realtime.py --camera 1      # cámara alternativa
  uv run python realtime.py --skip 3        # procesar 1 de cada 3 frames
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pose_extractor import PoseExtractor
from src.angle_utils import get_squat_angles, aggregate_video_features
from src.squat_classifier import SquatFormClassifier, SquatDetector, SquatPhase, CriterionResult

# ── Configuración visual ─────────────────────────────────────
_COLORS = {
    "green": (0, 200, 80),
    "yellow": (0, 200, 255),
    "red": (0, 50, 255),
    "white": (255, 255, 255),
    "gray": (180, 180, 180),
    "dark": (40, 40, 40),
    "bg": (25, 25, 25),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1


def _color_for_score(score: float) -> tuple[int, int, int]:
    if score >= 80:
        return _COLORS["green"]
    elif score >= 50:
        return _COLORS["yellow"]
    return _COLORS["red"]


def _color_for_phase(phase: SquatPhase) -> tuple[int, int, int]:
    mapping = {
        SquatPhase.STANDING: _COLORS["green"],
        SquatPhase.DESCENDING: _COLORS["yellow"],
        SquatPhase.BOTTOM: _COLORS["red"],
        SquatPhase.ASCENDING: _COLORS["yellow"],
        SquatPhase.TRANSITION: _COLORS["gray"],
        SquatPhase.UNKNOWN: _COLORS["gray"],
    }
    return mapping.get(phase, _COLORS["white"])


def _put_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = _COLORS["white"],
    scale: float = _FONT_SCALE,
    thick: int = _FONT_THICK,
    bg_alpha: float = 0.5,
) -> None:
    """Texto con fondo semitransparente."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    x, y = pos
    # Fondo
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + 4), _COLORS["dark"], -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    # Texto
    cv2.putText(frame, text, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _draw_gauge(
    frame: np.ndarray,
    label: str,
    value: float,
    max_val: float,
    pos: tuple[int, int],
    width: int = 140,
    height: int = 14,
) -> None:
    """Barra de progreso horizontal para una métrica."""
    x, y = pos
    pct = min(value / max_val, 1.0)
    color = _color_for_score(pct * 100)

    # Fondo
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Barra
    cv2.rectangle(frame, (x, y), (x + int(width * pct), y + height), color, -1)
    # Borde
    cv2.rectangle(frame, (x, y), (x + width, y + height), (120, 120, 120), 1)

    # Label y valor
    _put_text(frame, f"{label}: {value:.0f}%", (x, y - 6), color, 0.45, 1, 0.3)


def _draw_phase_badge(
    frame: np.ndarray,
    phase: SquatPhase,
    rep_count: int,
    score: float | None,
) -> None:
    """Panel superior izquierdo: fase actual + reps."""
    color = _color_for_phase(phase)
    phase_text = f"  {phase.value}  "

    # Fondo del badge
    (tw, th), _ = cv2.getTextSize(phase_text, _FONT, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (8 + tw + 16, 8 + th + 16), _COLORS["dark"], -1)
    cv2.rectangle(frame, (8, 8), (8 + tw + 16, 8 + th + 16), color, 1)

    cv2.putText(frame, phase_text, (16, 22 + th), _FONT, 0.7, color, 2, cv2.LINE_AA)

    # Reps
    _put_text(frame, f"Reps: {rep_count}", (16, 56 + th), _COLORS["white"], 0.5, 1)

    # Score si está disponible
    if score is not None:
        score_color = _color_for_score(score)
        _put_text(frame, f"Score: {score:.0f}/100", (16, 76 + th), score_color, 0.5, 1)


def _draw_metrics_panel(
    frame: np.ndarray,
    angles: dict[str, float],
    criteria: list[CriterionResult] | None,
    overall: float | None,
) -> None:
    """Panel derecho: métricas de ángulos y scores."""
    h, w = frame.shape[:2]
    panel_x = w - 220
    panel_y = 10
    panel_w = 210

    # Fondo del panel
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + 260),
        _COLORS["dark"],
        -1,
    )
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + 260),
        (60, 60, 60),
        1,
    )

    _put_text(frame, "METRICAS", (panel_x + 8, panel_y + 22), _COLORS["white"], 0.5, 1)

    y = panel_y + 40

    # Ángulos de rodilla
    for side, key in [("R Knee", "right_knee_angle"), ("L Knee", "left_knee_angle")]:
        val = angles.get(key)
        if val is not None:
            c = _COLORS["green"] if 70 <= val <= 110 else _COLORS["yellow"] if 60 <= val <= 130 else _COLORS["red"]
            _put_text(frame, f"{side}: {val:.0f}*", (panel_x + 8, y), c, 0.5, 1)
            y += 22

    # Back angle
    back = angles.get("back_angle")
    if back is not None:
        c = _COLORS["green"] if 20 <= back <= 50 else _COLORS["yellow"] if 15 <= back <= 60 else _COLORS["red"]
        _put_text(frame, f"Back: {back:.0f}*", (panel_x + 8, y), c, 0.5, 1)
        y += 22

    # Simetría
    sym = angles.get("knee_symmetry")
    if sym is not None:
        c = _COLORS["green"] if sym <= 10 else _COLORS["yellow"] if sym <= 20 else _COLORS["red"]
        _put_text(frame, f"Sym: {sym:.1f}*", (panel_x + 8, y), c, 0.5, 1)
        y += 22

    y += 10

    # Scores por criterio
    if criteria:
        _put_text(frame, "SCORES", (panel_x + 8, y), _COLORS["white"], 0.5, 1)
        y += 22
        for c in criteria:
            c_color = _color_for_score(c.score)
            _put_text(frame, f"{c.name[:6]}: {c.score:.0f}", (panel_x + 8, y), c_color, 0.45, 1)
            y += 20

    # Overall grande
    if overall is not None:
        y += 10
        o_color = _color_for_score(overall)
        cv2.putText(
            frame,
            f"{overall:.0f}/100",
            (panel_x + 40, y + 40),
            _FONT,
            1.2,
            o_color,
            2,
            cv2.LINE_AA,
        )
        _put_text(frame, "OVERALL", (panel_x + 40, y + 10), _COLORS["gray"], 0.45, 1)


def _draw_controls(frame: np.ndarray) -> None:
    """Barra inferior con controles."""
    h = frame.shape[0]
    _put_text(
        frame,
        "[q] salir  [r] reset reps  [p] pausa",
        (12, h - 12),
        _COLORS["gray"],
        0.4,
        1,
        0.3,
    )


def _draw_no_person(frame: np.ndarray) -> None:
    """Mensaje cuando no se detecta persona."""
    h, w = frame.shape[:2]
    _put_text(
        frame,
        "No se detecta persona — ponete frente a la camara",
        (w // 2 - 200, h // 2),
        _COLORS["yellow"],
        0.55,
        1,
    )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Squat Form Analyzer — Tiempo Real")
    parser.add_argument("--camera", type=int, default=0, help="ID de cámara (default: 0)")
    parser.add_argument("--skip", type=int, default=2, help="Procesar 1 de cada N frames (default: 2)")
    parser.add_argument("--width", type=int, default=800, help="Ancho del frame (default: 800)")
    parser.add_argument("--height", type=int, default=600, help="Alto del frame (default: 600)")
    args = parser.parse_args()

    print("=" * 56)
    print("  Squat Form Analyzer — MODO TIEMPO REAL")
    print("=" * 56)
    print(f"  Camara: {args.camera}")
    print(f"  Frame skip: 1 de cada {args.skip}")
    print(f"  Resolucion: {args.width}x{args.height}")
    print()
    print("  Controles:")
    print("    q → salir")
    print("    r → resetear reps")
    print("    p → pausar/reanudar")
    print()

    # ── Inicializar ───────────────────────────────────────────
    print("Cargando modelos...", end=" ", flush=True)
    pose = PoseExtractor()
    classifier = SquatFormClassifier()
    detector = SquatDetector()
    print("OK")

    # ── Webcam ────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Estado ────────────────────────────────────────────────
    paused = False
    frame_num = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0

    # Buffer de ángulos para el clasificador (acumula frames de una rep)
    rep_angles_buffer: list[dict] = []
    last_rep_score: float | None = None
    # Para no mostrar el score de la rep anterior después de varios frames
    score_fade_frames = 0
    prev_rep_count = 0  # Para detectar cuándo termina una rep

    print("Listo! Analizando en vivo... (q para salir)")
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        display_frame = frame.copy()

        # ── YOLO inference (cada N frames) ────────────────────
        if not paused and frame_num % args.skip == 0:
            kps, annotated = pose.extract_from_frame(frame)
        else:
            kps = []
            annotated = frame.copy()

        # ── Si hay persona ────────────────────────────────────
        if kps:
            kps_dict = pose.keypoints_to_dict(kps)
            angles = get_squat_angles(kps_dict)

            # Actualizar detector
            phase = detector.update(angles)

            # Acumular ángulos mientras está en una rep
            if detector.is_active:
                rep_angles_buffer.append(angles)
                # No dejar que crezca infinitamente
                if len(rep_angles_buffer) > 300:
                    rep_angles_buffer = rep_angles_buffer[-300:]

            # Evaluar forma cuando corresponde
            criteria: list[CriterionResult] | None = None
            overall: float | None = None

            if detector.should_analyze and len(rep_angles_buffer) >= 5:
                features = aggregate_video_features(rep_angles_buffer)
                criteria, overall = classifier.score_squat(features)
                last_rep_score = overall
                score_fade_frames = 15  # mostrar por 15 frames

            # Fade del score
            if score_fade_frames > 0:
                score_fade_frames -= 1
            else:
                criteria = None
                overall = None

            # Cuando termina la rep (detector incrementó contador), guardar score
            if detector._rep_count > prev_rep_count and len(rep_angles_buffer) >= 5:
                features = aggregate_video_features(rep_angles_buffer)
                _, rep_score = classifier.score_squat(features)
                last_rep_score = rep_score
                score_fade_frames = 30
                prev_rep_count = detector._rep_count
                rep_angles_buffer = []

            # ── Dibujar ────────────────────────────────────────
            display_frame = annotated.copy()

            # Badge de fase
            _draw_phase_badge(display_frame, phase, detector._rep_count, last_rep_score if score_fade_frames > 0 else None)

            # Panel de métricas
            show_score = overall if score_fade_frames > 0 else None
            show_criteria = criteria if score_fade_frames > 0 else None
            _draw_metrics_panel(display_frame, angles, show_criteria, show_score)

        else:
            # No hay persona
            _draw_no_person(display_frame)
            # Mostrar estado aunque no haya persona
            _draw_phase_badge(display_frame, detector.phase, detector._rep_count, None)

        # ── Controles y FPS ────────────────────────────────────
        _draw_controls(display_frame)

        # FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        _put_text(display_frame, f"{fps_display} FPS", (12, display_frame.shape[0] - 36), _COLORS["gray"], 0.4, 1, 0.3)

        # Indicador de pausa
        if paused:
            _put_text(
                display_frame,
                "PAUSADO",
                (display_frame.shape[1] // 2 - 50, 40),
                _COLORS["yellow"],
                0.7,
                2,
            )

        # ── Mostrar ────────────────────────────────────────────
        window_name = "Squat Form Analyzer — Tiempo Real"
        cv2.imshow(window_name, display_frame)

        # ── Controles ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset()
            rep_angles_buffer.clear()
            last_rep_score = None
            print("  Reset: contador de reps reiniciado")
        elif key == ord("p"):
            paused = not paused
            print(f"  {'Pausado' if paused else 'Reanudado'}")

    # ── Cierre ────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 56)
    print("  SESION FINALIZADA")
    print("=" * 56)
    print(f"  Reps detectadas: {detector._rep_count}")
    if detector.completed_reps:
        scores = detector.completed_reps
        print(f"  Scores: {[f'{s:.0f}/100' for s in scores]}")
        print(f"  Promedio: {np.mean(scores):.0f}/100")
        print(f"  Mejor: {max(scores):.0f}/100")
        print(f"  Peor: {min(scores):.0f}/100")
    else:
        print("  No se completaron repeticiones")
    print()


if __name__ == "__main__":
    main()
