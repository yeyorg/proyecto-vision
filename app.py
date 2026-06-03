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
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ── Nuestros módulos ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pose_extractor import PoseExtractor, KEYPOINT_NAMES
from src.angle_utils import (
    get_squat_angles,
    aggregate_video_features,
    ANGLE_EXPLANATIONS,
    SQUAT_ANGLE_KEYS,
)
from src.squat_classifier import SquatFormClassifier

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


def process_uploaded_video(
    video_bytes: bytes,
    pose_extractor: PoseExtractor,
    classifier: SquatFormClassifier,
    progress_bar,
    status_text,
) -> dict:
    """
    Procesar un video subido y devolver resultados.

    Returns
    -------
    dict con:
      - annotated_video_path: str
      - metrics: dict con ángulos agregados
      - criteria: lista de CriterionResult
      - overall: float
      - feedback: list[str]
      - frame_samples: list[np.ndarray] (frames ilustrativos)
    """
    # Guardar a archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Output video
        out_path = tmp_path.replace(".mp4", "_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

        cap = cv2.VideoCapture(tmp_path)
        frame_num = 0
        all_angles: list[dict] = []
        sample_frames: list[np.ndarray] = []
        sample_indices = set(
            np.linspace(0, total_frames - 1, min(6, total_frames), dtype=int)
        )

        status_text.text("⏳ Procesando video con YOLO pose...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % FRAME_SKIP == 0:
                kps, annotated = pose_extractor.extract_from_frame(frame)

                if kps:
                    kps_dict = pose_extractor.keypoints_to_dict(kps)
                    angles = get_squat_angles(kps_dict)
                    if angles:
                        all_angles.append(angles)

                        # Anotar ángulos en el frame
                        y = 30
                        for key, val in angles.items():
                            if "angle" in key:
                                color = (0, 255, 0) if 70 <= val <= 110 else (0, 165, 255)
                                cv2.putText(
                                    annotated,
                                    f"{key.split('_')[0]} {key.split('_')[1]}: {val:.0f}°",
                                    (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                )
                                y += 20
                else:
                    annotated = frame.copy()

                # Guardar frames de muestra
                if frame_num in sample_indices:
                    sample_frames.append(annotated.copy())
            else:
                annotated = frame.copy()

            writer.write(annotated)

            # Progress
            pct = int((frame_num + 1) / total_frames * 100)
            progress_bar.progress(min(pct, 100))
            if pct % 25 == 0:
                status_text.text(f"⏳ Procesando... {pct}% ({frame_num + 1}/{total_frames})")

            frame_num += 1

        cap.release()
        writer.release()

        status_text.text("📊 Evaluando forma...")

        # Agregar features y clasificar
        if not all_angles:
            return {
                "error": "No se detectaron personas en el video. Asegurate de estar frente a la cámara.",
                "annotated_video_path": out_path,
                "sample_frames": sample_frames or [],
            }

        features = aggregate_video_features(all_angles)
        criteria, overall = classifier.score_squat(features)
        pred = int(classifier.predict([features])[0])
        proba = classifier.predict_proba([features])[0].tolist()
        feedback = classifier.get_feedback(features)

        return {
            "annotated_video_path": out_path,
            "features": features,
            "criteria": criteria,
            "overall": overall,
            "prediction": pred,
            "probabilities": proba,
            "feedback": feedback,
            "sample_frames": sample_frames,
            "total_frames": total_frames,
            "fps": fps,
            "n_angles_frames": len(all_angles),
            "error": None,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Limpiar el archivo subido original (no el anotado, el usuario lo descarga)
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


def draw_gauge(score, label, max_val=100, size=120):
    """Dibujar un gauge circular simple con matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Fondo
    ax.barh(0, np.pi, height=0.8, color="#e0e0e0", linewidth=0)
    # Score
    angle = score / max_val * np.pi
    color = "#28a745" if score >= 80 else "#e67e22" if score >= 50 else "#dc3545"
    ax.barh(0, angle, height=0.8, color=color, linewidth=0)

    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.text(0, 0.5, f"{score:.0f}", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0, -0.3, label, ha="center", va="center", fontsize=8)

    fig.tight_layout(pad=0)
    return fig


# ── Interfaz ──────────────────────────────────────────────────────────────


def main():
    st.title("🏋️ Squat Form Analyzer")
    st.markdown(
        "Analizá la técnica de tu sentadilla usando **visión por computadora** "
        "con YOLOv8-pose + reglas biomecánicas."
    )

    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        pose_extractor, classifier = load_models()
    st.sidebar.success("✅ Modelos cargados")

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("📤 Input")

    input_mode = st.sidebar.radio(
        "Fuente de entrada",
        ["Subir video", "Webcam (foto)"],
        help="Subí un video grabado o usá la webcam para una foto.",
    )

    video_bytes = None
    use_webcam = input_mode == "Webcam (foto)"

    if use_webcam:
        camera_image = st.sidebar.camera_input("Tomar foto")
        if camera_image:
            video_bytes = camera_image.getvalue()
            st.sidebar.info("📸 Foto capturada. Procesando...")
    else:
        uploaded = st.sidebar.file_uploader(
            "Seleccionar video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Formatos soportados: MP4, AVI, MOV, MKV",
        )
        if uploaded:
            if uploaded.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
                st.sidebar.error(f"❌ El video es muy grande (máx {MAX_VIDEO_SIZE_MB} MB)")
            else:
                video_bytes = uploaded.getvalue()
                st.sidebar.success(f"✅ {uploaded.name} subido ({uploaded.size / 1024 / 1024:.1f} MB)")

    # Opciones de análisis
    st.sidebar.header("⚙️ Opciones")
    show_angles_on_video = st.sidebar.checkbox("Mostrar ángulos en video", value=True)

    # ── Modo Tiempo Real ──────────────────────────────────────────────────
    st.sidebar.header("🎥 Tiempo Real")
    st.sidebar.markdown(
        "Analizá tu sentadilla **en vivo** desde la webcam con detección "
        "automática de fases (DE PIE → BAJANDO → FONDO → SUBIENDO)."
    )

    if st.sidebar.button("▶️ Abrir Tiempo Real", use_container_width=True):
        import subprocess
        import sys as _sys

        st.sidebar.info("Abriendo ventana de tiempo real...")
        # Lanzar en un proceso separado para no bloquear Streamlit
        script = Path(__file__).resolve().parent / "realtime.py"
        subprocess.Popen(
            [_sys.executable, str(script)],
            creationflags=subprocess.CREATE_NO_WINDOW if _sys.platform == "win32" else 0,
        )
        st.sidebar.success(
            "Ventana de tiempo real abierta. "
            "Presioná **q** sobre la ventana para salir."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💡 *También podés correrlo desde la terminal:*  \n"
        "`uv run python realtime.py`"
    )

    # ── Procesar ──────────────────────────────────────────────────────────
    if video_bytes is None:
        st.info(
            "👈 Subí un video, usá la webcam o activá el **Modo Tiempo Real** "
            "desde la barra lateral."
        )

        # Botón rápido para tiempo real también acá
        col_rt1, col_rt2 = st.columns([1, 4])
        with col_rt1:
            if st.button("🎥 Tiempo Real", use_container_width=True):
                import subprocess
                import sys as _sys

                script = Path(__file__).resolve().parent / "realtime.py"
                subprocess.Popen(
                    [_sys.executable, str(script)],
                    creationflags=subprocess.CREATE_NO_WINDOW if _sys.platform == "win32" else 0,
                )
                st.success("Ventana de tiempo real abierta. Presioná **q** para salir.")

        st.markdown("---")
        st.markdown("### ¿Cómo funciona?")
        col1, col2, col3, col4 = st.columns(4)
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
            st.markdown("**3. Detección de Fase**")
            st.markdown(
                "Solo analizamos cuando detectamos movimiento "
                "de sentadilla — ignoramos cuando estás de pie."
            )
        with col4:
            st.markdown("**4. Evaluación**")
            st.markdown(
                "Comparamos contra rangos óptimos y generamos feedback."
            )

        st.markdown("---")
        st.markdown(
            "**💡 Tip:** Grabate de perfil haciendo 3-5 sentadillas para mejor análisis."
        )
        return

    # ── Procesamiento ────────────────────────────────────────────────────
    progress_bar = st.progress(0)
    status_text = st.empty()

    if use_webcam:
        # Foto individual — procesar un solo frame
        status_text.text("📸 Analizando foto...")

        nparr = np.frombuffer(video_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        kps, annotated = pose_extractor.extract_from_frame(frame)

        if not kps:
            st.error("❌ No se detectó ninguna persona en la foto.")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            progress_bar.empty()
            status_text.empty()
            return

        kps_dict = pose_extractor.keypoints_to_dict(kps)
        angles = get_squat_angles(kps_dict)
        features = aggregate_video_features([angles])
        criteria, overall = classifier.score_squat(features)
        pred = int(classifier.predict([features])[0])
        proba = classifier.predict_proba([features])[0].tolist()
        feedback = classifier.get_feedback(features)

        # Mostrar resultado
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("📷 Foto analizada")
            # Anotar ángulos si se pidió
            display_frame = annotated.copy()
            if show_angles_on_video and angles:
                y = 30
                for key, val in angles.items():
                    if "angle" in key:
                        color = (0, 255, 0) if 70 <= val <= 110 else (0, 165, 255)
                        cv2.putText(
                            display_frame,
                            f"{key}: {val:.0f}°",
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                        1,
                        )
                        y += 25
            st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_right:
            st.subheader("📊 Resultado")
            pred_label = "✅ Buena forma" if pred == 0 else "⚠️  Necesita trabajo"
            st.markdown(f"### {pred_label}")
            st.markdown(f"**Puntaje global:** {overall:.1f}/100")
            st.markdown(f"**Confianza:** buena {proba[0]:.0%} / mala {proba[1]:.0%}")

            # Métricas
            mcol1, mcol2 = st.columns(2)
            for c in criteria:
                with mcol1 if c.name in ("depth", "back_angle") else mcol2:
                    st.metric(
                        label=c.name.replace("_", " ").title(),
                        value=f"{c.score:.0f}/100",
                        delta=None,
                    )

        # Feedback
        st.subheader("💡 Feedback")
        for tip in feedback:
            st.markdown(tip)

        progress_bar.empty()
        status_text.empty()
        return

    # ── Video ─────────────────────────────────────────────────────────────
    status_text.text("⏳ Iniciando análisis...")
    result = process_uploaded_video(
        video_bytes, pose_extractor, classifier, progress_bar, status_text
    )

    progress_bar.empty()
    status_text.empty()

    if result.get("error"):
        st.error(f"❌ {result['error']}")
        if result.get("sample_frames"):
            st.subheader("📹 Frames del video")
            cols = st.columns(len(result["sample_frames"]))
            for i, (col, frame) in enumerate(zip(cols, result["sample_frames"])):
                col.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        return

    # ── Resultados ────────────────────────────────────────────────────────
    overall = result["overall"]
    criteria = result["criteria"]
    pred = result["prediction"]
    proba = result["probabilities"]
    feedback = result["feedback"]

    # --- Header ---
    pred_label = "✅ Buena forma" if pred == 0 else "⚠️  Necesita trabajo"
    overall_color = "#28a745" if overall >= 80 else "#e67e22" if overall >= 50 else "#dc3545"

    st.markdown(
        f"""
        <div style="text-align:center;padding:1.5rem;border-radius:10px;background:#f8f9fa;margin-bottom:1rem">
            <h2 style="margin:0;color:{overall_color}">{pred_label}</h2>
            <div style="font-size:3.5rem;font-weight:bold;color:{overall_color}">{overall:.0f}/100</div>
            <p style="color:#666">Buena forma: {proba[0]:.0%} / Necesita trabajo: {proba[1]:.0%}</p>
            <p style="font-size:0.9em;color:#999">Frames analizados: {result['n_angles_frames']} de {result['total_frames']} | FPS: {result['fps']:.0f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Columnas ---
    col_metrics, col_video = st.columns([2, 3])

    with col_video:
        st.subheader("🎥 Video Anotado")

        # Leer el video anotado para mostrarlo
        vid_path = result["annotated_video_path"]
        if os.path.exists(vid_path):
            with open(vid_path, "rb") as f:
                video_bytes_out = f.read()
            st.video(video_bytes_out)

            # Descarga
            st.download_button(
                label="⬇️ Descargar video anotado",
                data=video_bytes_out,
                file_name="squat_analysis.mp4",
                mime="video/mp4",
            )
        else:
            st.warning("Video anotado no disponible")

        # Frames de muestra
        if result.get("sample_frames"):
            st.markdown("**Frames ilustrativos:**")
            sample = result["sample_frames"]
            cols = st.columns(len(sample))
            for i, (col, frame) in enumerate(zip(cols, sample)):
                col.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=f"Frame {i + 1}",
                )

    with col_metrics:
        st.subheader("📊 Métricas")

        # --- Scores por criterio ---
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

            # Barra personalizada
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

        # --- Feedback ---
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

        # --- Detalle de ángulos ---
        if result.get("features"):
            st.markdown("---")
            st.subheader("📐 Ángulos detectados")
            features = result["features"]
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
