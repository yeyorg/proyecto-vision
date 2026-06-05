"""
Streamlit App — Squat Form Analyzer 🏋️

Analiza la forma de sentadillas usando YOLOv8-pose + reglas biomecánicas.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Nuestros módulos ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pose_extractor import PoseExtractor
from src.angle_utils import (
    get_squat_angles,
    aggregate_video_features,
)
from src.squat_classifier import SquatFormClassifier
from src.injury_advice import get_injury_assessment, DISCLAIMER

# ── Constantes ────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/squat_form_model.pkl")
YOLO_MODEL = "yolov8n-pose.pt"
FRAME_SKIP = 3  # procesar 1 de cada N frames
MAX_VIDEO_SIZE_MB = 200

# ── Página ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Squat Form Analyzer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ───────────────────────────────────────────────────────────────


@st.cache_resource
def load_models():
    """Cargar modelos (cacheados en memoria)."""
    pose = PoseExtractor(YOLO_MODEL)
    if MODEL_PATH.exists():
        classifier = joblib_load_safe(MODEL_PATH)
    else:
        classifier = SquatFormClassifier()
    return pose, classifier


def joblib_load_safe(path: Path):
    """Cargar joblib manejando posibles errores de serialización."""
    import joblib

    return joblib.load(path)


def analyze_video_tracks(
    video_bytes: bytes,
    pose_extractor: PoseExtractor,
    classifier: SquatFormClassifier,
    progress_bar,
    status_text,
) -> dict:
    """
    Procesar un video detectando a TODAS las personas y evaluando a cada una.

    Returns
    -------
    dict con: tracks (una entrada por persona), annotated_bytes, total_frames,
    fps, error.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        status_text.text("Detectando y rastreando personas con YOLO pose...")

        def _progress(done, total):
            pct = int(done / max(1, total) * 100)
            progress_bar.progress(min(pct, 100))

        tracks_raw, annotated_path = pose_extractor.process_video_tracks(
            tmp_path,
            frame_skip=FRAME_SKIP,
            progress_callback=_progress,
        )

        status_text.text("Evaluando la forma de cada persona...")

        tracks: list[dict] = []
        for t in tracks_raw:
            frames_angles = [
                get_squat_angles(entry["kps"]) for entry in t["keypoints_per_frame"]
            ]
            frames_angles = [a for a in frames_angles if a]
            if not frames_angles:
                continue

            features = aggregate_video_features(frames_angles)
            criteria, overall = classifier.score_squat(features)
            pred = int(classifier.predict([features])[0])
            proba = classifier.predict_proba([features])[0].tolist()
            feedback = classifier.get_feedback(features)
            thumb_rgb = cv2.cvtColor(t["thumb"], cv2.COLOR_BGR2RGB)

            tracks.append({
                "track_id": t["track_id"],
                "thumb": thumb_rgb,
                "n_angles_frames": len(frames_angles),
                "features": features,
                "criteria": criteria,
                "overall": overall,
                "prediction": pred,
                "probabilities": proba,
                "feedback": feedback,
            })

        annotated_bytes = None
        if os.path.exists(annotated_path):
            with open(annotated_path, "rb") as f:
                annotated_bytes = f.read()
            try:
                os.unlink(annotated_path)
            except Exception:
                pass

        if not tracks:
            return {
                "error": "No se detectaron sentadillas evaluables en el video. "
                         "Asegúrate de que la persona se vea de perfil y completa.",
                "annotated_bytes": annotated_bytes,
                "tracks": [],
            }

        return {
            "tracks": tracks,
            "annotated_bytes": annotated_bytes,
            "total_frames": total_frames,
            "fps": fps,
            "error": None,
        }

    except Exception as e:
        return {"error": str(e), "tracks": []}

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def style_metric(value, max_val=100):
    """Código CSS condicional para el color del score."""
    if value >= 80:
        return f'<span style="color:#28a745;font-weight:bold;font-size:1.2em">{value:.0f}</span>'
    elif value >= 50:
        return f'<span style="color:#e67e22;font-weight:bold;font-size:1.2em">{value:.0f}</span>'
    else:
        return f'<span style="color:#dc3545;font-weight:bold;font-size:1.2em">{value:.0f}</span>'


# ── Interfaz ──────────────────────────────────────────────────────────────


def main():
    st.title("🏋️ Squat Form Analyzer")
    st.markdown(
        "Analiza la técnica de tu sentadilla usando **visión por computadora** "
        "con YOLOv8-pose + reglas biomecánicas."
    )

    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        pose_extractor, classifier = load_models()
    st.sidebar.success("✅ Modelos cargados")

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("📤 Input")

    uploaded = st.sidebar.file_uploader(
        "Seleccionar video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Formatos soportados: MP4, AVI, MOV, MKV",
    )

    video_bytes = None
    if uploaded:
        if uploaded.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
            st.sidebar.error(f"❌ El video es muy grande (máx {MAX_VIDEO_SIZE_MB} MB)")
        else:
            video_bytes = uploaded.getvalue()
            st.sidebar.success(f"✅ {uploaded.name} subido ({uploaded.size / 1024 / 1024:.1f} MB)")

    # ── Sin video ─────────────────────────────────────────────────────────
    if video_bytes is None:
        st.info("👈 Sube un video MP4 desde la barra lateral para analizar tu sentadilla.")

        st.markdown("### ¿Cómo funciona?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Pose Estimation**")
            st.markdown(
                "YOLOv8-pose detecta 17 keypoints del cuerpo en cada frame."
            )
        with col2:
            st.markdown("**2. Ángulos Biomecánicos**")
            st.markdown(
                "Calculamos ángulos de rodilla, cadera, espalda y simetría."
            )
        with col3:
            st.markdown("**3. Evaluación**")
            st.markdown(
                "Comparamos contra rangos óptimos y generamos feedback personalizado."
            )

        st.markdown("---")
        st.markdown(
            "**💡 Tip:** Grábate de perfil haciendo 3-5 sentadillas para mejor análisis."
        )
        return

    # ── Procesamiento (cacheado por archivo en session_state) ─────────────
    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("analysis_key") != file_key:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Iniciando analisis...")

        result = analyze_video_tracks(
            video_bytes, pose_extractor, classifier, progress_bar, status_text
        )

        progress_bar.empty()
        status_text.empty()

        st.session_state["analysis_key"] = file_key
        st.session_state["analysis_result"] = result

    result = st.session_state.get("analysis_result", {})

    if result.get("error"):
        st.error(f"❌ {result['error']}")
        if result.get("annotated_bytes"):
            st.subheader("🎥 Video procesado")
            st.video(result["annotated_bytes"])
        return

    tracks = result["tracks"]

    # ── Selector de persona ───────────────────────────────────────────────
    st.subheader("🧍 Personas detectadas")
    st.caption(
        "Se detecto mas de una persona. Elige a quien quieres analizar — "
        "el numero coincide con la etiqueta sobre cada persona en el video."
        if len(tracks) > 1 else
        "Se analizo la persona detectada en el video."
    )

    thumb_cols = st.columns(max(len(tracks), 1))
    for col, t in zip(thumb_cols, tracks):
        col.image(
            t["thumb"],
            caption=f"Persona {t['track_id']} · {t['overall']:.0f}/100",
            use_container_width=True,
        )

    if len(tracks) > 1:
        options = [t["track_id"] for t in tracks]
        selected_id = st.radio(
            "¿Que persona quieres evaluar?",
            options,
            format_func=lambda tid: f"Persona {tid}",
            horizontal=True,
        )
    else:
        selected_id = tracks[0]["track_id"]

    selected = next(t for t in tracks if t["track_id"] == selected_id)

    st.markdown("---")
    render_track_result(
        selected,
        result.get("annotated_bytes"),
        total_frames=result.get("total_frames", 0),
        fps=result.get("fps", 0),
    )


def render_track_result(track: dict, annotated_bytes, total_frames: int, fps: float):
    """Renderiza el resultado de una persona (track) seleccionada."""
    overall = track["overall"]
    criteria = track["criteria"]
    pred = track["prediction"]
    proba = track["probabilities"]
    feedback = track["feedback"]

    pred_label = "✅ Buena forma" if pred == 0 else "⚠️  Necesita trabajo"
    overall_color = "#28a745" if overall >= 80 else "#e67e22" if overall >= 50 else "#dc3545"

    st.markdown(
        f"""
        <div style="text-align:center;padding:1.5rem;border-radius:10px;background:#f8f9fa;margin-bottom:1rem">
            <h2 style="margin:0;color:{overall_color}">Persona {track['track_id']} — {pred_label}</h2>
            <div style="font-size:3.5rem;font-weight:bold;color:{overall_color}">{overall:.0f}/100</div>
            <p style="color:#666">Buena forma: {proba[0]:.0%} / Necesita trabajo: {proba[1]:.0%}</p>
            <p style="font-size:0.9em;color:#999">Frames analizados: {track['n_angles_frames']} de {total_frames} | FPS: {fps:.0f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_metrics, col_video = st.columns([2, 3])

    with col_video:
        st.subheader("🎥 Video Anotado")
        if annotated_bytes:
            st.video(annotated_bytes)
            st.download_button(
                label="⬇️ Descargar video anotado",
                data=annotated_bytes,
                file_name="squat_analysis.mp4",
                mime="video/mp4",
            )
        else:
            st.warning("Video anotado no disponible")

    with col_metrics:
        st.subheader("📊 Métricas")
        st.markdown("**Evaluación por criterio**")
        for c in criteria:
            val = c.score
            if val >= 80:
                icon = "✅"
                bar_color = "green"
            elif val >= 50:
                icon = "⚠️"
                bar_color = "orange"
            else:
                icon = "❌"
                bar_color = "red"

            st.markdown(
                f"""
                <div style="margin-bottom:0.8rem">
                    <div style="display:flex;justify-content:space-between;font-size:0.9em">
                        <span>{icon} {c.name.replace('_', ' ').title()}</span>
                        <span style="font-weight:bold">{c.score:.0f}/100</span>
                    </div>
                    <div style="background:#e0e0e0;border-radius:4px;height:8px;width:100%">
                        <div style="background:{bar_color};border-radius:4px;height:8px;width:{c.score}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("💡 Feedback")
        for tip in feedback:
            if tip.startswith("⚠️") or tip.startswith("🔧"):
                st.warning(tip)
            elif tip.startswith("ℹ️"):
                st.info(tip)
            elif tip.startswith("✅"):
                st.success(tip)
            else:
                st.markdown(tip)

        # --- Riesgo de lesion y recomendaciones ---
        assessments = get_injury_assessment(criteria, track.get("features", {}))
        if assessments:
            st.markdown("---")
            st.subheader("🩺 Riesgo de lesión y recomendaciones")
            sev_color = {"Alta": "#dc3545", "Moderada": "#e67e22", "Leve": "#f1c40f"}
            for a in assessments:
                color = sev_color.get(a["severidad"], "#888888")
                st.markdown(
                    f"""
                    <div style="border-left:4px solid {color};padding:0.5rem 0.8rem;margin-bottom:0.6rem;background:#f8f9fa;border-radius:4px">
                        <div style="font-weight:bold;color:#222">{a['zona']} · <span style="color:{color}">Riesgo {a['severidad'].lower()}</span></div>
                        <div style="font-size:0.9em;color:#444;margin-top:2px"><b>{a['criterio']}</b></div>
                        <div style="font-size:0.9em;color:#666;margin-top:2px">⚠️ Exposición: {a['lesiones']}</div>
                        <div style="font-size:0.9em;color:#1f6f3f;margin-top:2px">✅ Recomendación: {a['recomendacion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(DISCLAIMER)

        features = track.get("features")
        if features:
            st.markdown("---")
            st.subheader("📐 Ángulos detectados")
            angle_keys = [k for k in features if k.endswith("_mean") and "knee_angle" in k]
            for key in angle_keys:
                mean_val = features.get(key, 0)
                std_val = features.get(key.replace("_mean", "_std"), 0)
                min_val = features.get(key.replace("_mean", "_min"), 0)
                max_val = features.get(key.replace("_mean", "_max"), 0)
                label = key.replace("_mean", "").replace("_", " ").title()
                st.markdown(
                    f"**{label}:** media={mean_val:.1f}°, "
                    f"min={min_val:.1f}°, max={max_val:.1f}°, "
                    f"σ={std_val:.1f}°"
                )


if __name__ == "__main__":
    main()
