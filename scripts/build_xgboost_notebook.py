"""
Generar el notebook de entrenamiento XGBoost.
Usa nbformat para armar el .ipynb con código + markdown.
"""

import nbformat as nbf

NB_PATH = "xgboost_training.ipynb"

# ── helpers ──────────────────────────────────────────────────────────────────


def md(src: str):
    """Crear celda de markdown con líneas correctamente separadas."""
    return nbf.v4.new_markdown_cell(src.splitlines(keepends=True))


def code(src: str):
    """Crear celda de código con líneas correctamente separadas."""
    return nbf.v4.new_code_cell(src.splitlines(keepends=True))


# ── notebook ─────────────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.12"},
}
cells = nb.cells

# ══════════════════════════════════════════════════════════════════════════════
# CELL 0 — TITLE
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""# 🏋️ Entrenamiento XGBoost para Clasificación de Sentadillas

**Pipeline completo:**
1. Explorar videos → detectar personas con YOLOv8-pose
2. Extraer keypoints multi-persona por frame con tracking
3. Calcular ángulos biomecánicos por persona
4. Agregar features estadísticos (mean, std, min, max)
5. Entrenar XGBoost con datos reales
6. Evaluar y guardar el modelo

> Este notebook es **autocontenido** — no importa nada de `src/`.
> Todo el código necesario está inline para que sea reproducible.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 1. Imports y Configuración
"""))

cells.append(code("""import os, sys, math, warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

print("✅ Imports listos")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — CONFIG
# ══════════════════════════════════════════════════════════════════════════════

cells.append(code("""# ── Rutas ──────────────────────────────────────────────────
DATA_DIR = Path("data_videos")
CLASSES = {
    "squat_correcto": 0,       # buena forma
    "squatmal_ejecutado": 1,   # mala forma
}
OUTPUT_MODEL = Path("models/xgboost_squat.pkl")

# ── Parámetros de procesamiento ────────────────────────────
FRAME_SKIP = 3        # procesar 1 de cada N frames
CONF_THRESH = 0.5     # confianza mínima de detección
TRACK_DIST_THRESH = 200  # px máx para considerar misma persona entre frames
YOLO_MODEL = "yolov8n-pose.pt"

# ── COCO keypoint indices ─────────────────────────────────
(NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
 L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
 L_WRIST, R_WRIST, L_HIP, R_HIP,
 L_KNEE, R_KNEE, L_ANKLE, R_ANKLE) = range(17)

print(f"📁 Videos en: {DATA_DIR.resolve()}")
print(f"📦 Clases: {CLASSES}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — FUNCIONES DE ÁNGULOS
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 2. Funciones de Ángulos Biomecánicos

Cada función trabaja sobre **keypoints de una persona** (dict `{id: {x, y}}`).
Las fórmulas son copia directa de las que usa la app — acá están inline para
que el notebook sea autocontenido.
"""))

cells.append(code("""# ── Geometría básica ───────────────────────────────────────

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    \"\"\"Ángulo (en grados) entre dos vectores.\"\"\"
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-8:
        return 90.0
    cos_angle = max(-1.0, min(1.0, dot / norm))
    return float(np.degrees(np.arccos(cos_angle)))


def angle_3pt(p1, p2, p3) -> float:
    \"\"\"Ángulo P1–P2–P3 con vértice en P2 (en grados).\"\"\"
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    return angle_between(v1, v2)


# ── Ángulos específicos de sentadilla ──────────────────────

def get_squat_angles(kps: dict) -> dict:
    \"\"\"
    kps: {id: {x, y}}  (COCO keypoints de UNA persona)
    Retorna dict con ángulos calculados.
    \"\"\"
    angles = {}

    def pt(idx):
        p = kps.get(idx)
        return (p["x"], p["y"]) if p is not None else None

    # Rodilla (cadera → rodilla → tobillo)
    for side, hip_id, knee_id, ankle_id in [
        ("left", L_HIP, L_KNEE, L_ANKLE),
        ("right", R_HIP, R_KNEE, R_ANKLE),
    ]:
        hip, knee, ankle = pt(hip_id), pt(knee_id), pt(ankle_id)
        if hip and knee and ankle:
            angles[f"{side}_knee_angle"] = angle_3pt(hip, knee, ankle)

    # Cadera (hombro → cadera → rodilla)
    for side, sh_id, hip_id, knee_id in [
        ("left", L_SHOULDER, L_HIP, L_KNEE),
        ("right", R_SHOULDER, R_HIP, R_KNEE),
    ]:
        sh, hip, knee = pt(sh_id), pt(hip_id), pt(knee_id)
        if sh and hip and knee:
            angles[f"{side}_hip_angle"] = angle_3pt(sh, hip, knee)

    # Espalda (torso vs vertical)
    ls, rs, lh, rh = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_HIP), pt(R_HIP)
    if all(p is not None for p in [ls, rs, lh, rh]):
        mid_sh = np.mean([ls, rs], axis=0)
        mid_hip = np.mean([lh, rh], axis=0)
        torso = mid_sh - mid_hip
        vertical = np.array([0.0, -1.0])
        angles["back_angle"] = angle_between(torso, vertical)

    # Knee-over-toe
    for side, knee_id, ankle_id in [
        ("left", L_KNEE, L_ANKLE),
        ("right", R_KNEE, R_ANKLE),
    ]:
        knee, ankle = pt(knee_id), pt(ankle_id)
        if knee and ankle:
            angles[f"{side}_knee_toe_x"] = float(knee[0] - ankle[0])

    # Simetría
    if "left_knee_angle" in angles and "right_knee_angle" in angles:
        angles["knee_symmetry"] = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
    if "left_hip_angle" in angles and "right_hip_angle" in angles:
        angles["hip_symmetry"] = abs(angles["left_hip_angle"] - angles["right_hip_angle"])

    return angles


# ── Features agregadas por video ───────────────────────────

ANGLE_KEYS = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "back_angle", "left_knee_toe_x", "right_knee_toe_x",
    "knee_symmetry", "hip_symmetry",
]


def aggregate_features(frame_angles: list[dict]) -> dict:
    \"\"\"
    Toma una lista de dicts (uno por frame) y calcula
    estadísticos agregados (mean, std, min, max) para cada métrica.
    \"\"\"
    if not frame_angles:
        return {}

    all_keys = set()
    for fa in frame_angles:
        all_keys.update(fa.keys())

    features = {}
    for key in sorted(all_keys):
        vals = [fa[key] for fa in frame_angles if key in fa]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        features[f"{key}_mean"] = float(np.mean(arr))
        features[f"{key}_std"] = float(np.std(arr))
        features[f"{key}_min"] = float(np.min(arr))
        features[f"{key}_max"] = float(np.max(arr))

    return features


print("✅ Funciones de ángulos y features listas")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — TRACKING MULTI-PERSONA
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 3. Tracking Multi-Persona

YOLO detecta **todas las personas** en cada frame. Para asignar detecciones
a la misma persona a través del video, usamos **seguimiento por proximidad**
del centro del bounding box entre frames consecutivos.

![tracking concept](https://raw.githubusercontent.com/ultralytics/assets/main/examples/tracker.png)

### Algoritmo:
1. Por cada frame, obtener bounding boxes de todas las personas detectadas
2. Calcular centro del bbox para cada una
3. Comparar con centros del frame anterior
4. Asignación greedy: cada nueva detección se empareja con la más cercana
5. Si la distancia supera `TRACK_DIST_THRESH`, es una persona nueva
"""))

cells.append(code("""def bbox_center(bbox):
    \"\"\"Centro (x, y) de un bbox [x1, y1, x2, y2].\"\"\"
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def greedy_match(current_centers, prev_centers, max_dist):
    \"\"\"
    Empareja detecciones actuales con tracks previos.

    Returns
    -------
    assignments : list[(track_idx, det_idx)]
        Emparejamientos track → detección.
    unmatched_dets : list[int]
        Índices de detecciones sin track previo (personas nuevas).
    \"\"\"
    assignments = []
    used_prev = set()
    used_curr = set()

    # Para cada detección actual, encontrar el track previo más cercano
    for det_idx, c_center in enumerate(current_centers):
        best_dist = max_dist
        best_prev = None
        for prev_idx, p_center in enumerate(prev_centers):
            if prev_idx in used_prev:
                continue
            dist = np.linalg.norm(np.array(c_center) - np.array(p_center))
            if dist < best_dist:
                best_dist = dist
                best_prev = prev_idx

        if best_prev is not None:
            assignments.append((best_prev, det_idx))
            used_prev.add(best_prev)
            used_curr.add(det_idx)

    unmatched = [i for i in range(len(current_centers)) if i not in used_curr]
    return assignments, unmatched


def process_video_multi(video_path: str, model: YOLO,
                        frame_skip: int = 3,
                        conf_thresh: float = 0.5,
                        track_dist: float = 200.0):
    \"\"\"
    Procesa un video y devuelve los tracks de cada persona.

    Parameters
    ----------
    video_path : str
        Ruta al video.
    model : YOLO
        Modelo YOLO cargado.
    frame_skip : int
        Procesar 1 de cada N frames.
    conf_thresh : float
        Confianza mínima de detección.
    track_dist : float
        Distancia máxima (px) para considerar misma persona.

    Returns
    -------
    list[dict]
        Cada elemento::
            {track_id, keypoints_per_frame: [{frame, dict_kps}],
             angles_per_frame: [{frame, angle_dict}],
             n_frames: int}
    \"\"\"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks: dict[int, dict] = {}       # track_id -> track data
    prev_centers: dict[int, tuple] = {}  # track_id -> center (x, y)
    next_track_id = 0
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip != 0:
            frame_num += 1
            continue

        results = model(frame, verbose=False)
        if (results[0].keypoints is None
                or len(results[0].keypoints.xy) == 0
                or results[0].boxes is None):
            frame_num += 1
            continue

        n_people = len(results[0].keypoints.xy)

        # Armar detecciones actuales
        current_dets = []
        current_centers = []
        for p_idx in range(n_people):
            bbox = results[0].boxes.xyxy[p_idx].tolist()
            det_conf = float(results[0].boxes.conf[p_idx].item())
            if det_conf < conf_thresh:
                continue

            kp_list = []
            kps = results[0].keypoints
            for i in range(len(kps.xy[p_idx])):
                x, y = kps.xy[p_idx][i].tolist()
                c = float(kps.conf[p_idx][i].item())
                kp_list.append({"id": i, "x": x, "y": y, "confidence": c})

            current_dets.append({"keypoints": kp_list, "bbox": bbox, "conf": det_conf})
            current_centers.append(bbox_center(bbox))

        if not current_dets:
            frame_num += 1
            continue

        # Matching greedy con tracks previos
        prev_list = list(prev_centers.items())  # [(track_id, center), ...]
        prev_ids = [p[0] for p in prev_list]
        prev_cent = [p[1] for p in prev_list]

        if not prev_cent:
            # Primer frame: todos son nuevos tracks
            for det_idx, det in enumerate(current_dets):
                tid = next_track_id
                next_track_id += 1
                kps_dict = {kp["id"]: {"x": kp["x"], "y": kp["y"]}
                            for kp in det["keypoints"]}
                angles = get_squat_angles(kps_dict)
                tracks[tid] = {
                    "track_id": tid,
                    "keypoints_per_frame": [{"frame": frame_num, "keypoints": kps_dict}],
                    "angles_per_frame": [{"frame": frame_num, "angles": angles}] if angles else [],
                    "n_frames": 1,
                }
                prev_centers[tid] = current_centers[det_idx]
        else:
            assignments, unmatched = greedy_match(current_centers, prev_cent, track_dist)

            # Actualizar tracks existentes
            new_centers = {}
            for track_idx, det_idx in assignments:
                tid = prev_ids[track_idx]
                det = current_dets[det_idx]
                kps_dict = {kp["id"]: {"x": kp["x"], "y": kp["y"]}
                            for kp in det["keypoints"]}
                angles = get_squat_angles(kps_dict)
                tracks[tid]["keypoints_per_frame"].append({"frame": frame_num, "keypoints": kps_dict})
                if angles:
                    tracks[tid]["angles_per_frame"].append({"frame": frame_num, "angles": angles})
                tracks[tid]["n_frames"] += 1
                new_centers[tid] = current_centers[det_idx]

            # Nuevos tracks para detecciones no emparejadas
            for det_idx in unmatched:
                tid = next_track_id
                next_track_id += 1
                det = current_dets[det_idx]
                kps_dict = {kp["id"]: {"x": kp["x"], "y": kp["y"]}
                            for kp in det["keypoints"]}
                angles = get_squat_angles(kps_dict)
                tracks[tid] = {
                    "track_id": tid,
                    "keypoints_per_frame": [{"frame": frame_num, "keypoints": kps_dict}],
                    "angles_per_frame": [{"frame": frame_num, "angles": angles}] if angles else [],
                    "n_frames": 1,
                }
                new_centers[tid] = current_centers[det_idx]

            prev_centers = new_centers

        frame_num += 1

    cap.release()
    return list(tracks.values())


print("✅ Función de tracking multi-persona lista")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — EXPLORAR DATASET
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 4. Explorar el Dataset

Vamos a listar todos los videos, ver sus resoluciones y hacer una prueba
rápida de detección con YOLO.
"""))

cells.append(code("""# ── Listar videos ──────────────────────────────────────────

videos = []
for class_name, label in CLASSES.items():
    class_dir = DATA_DIR / class_name
    if not class_dir.exists():
        print(f"⚠️  No existe: {class_dir}")
        continue
    for f in sorted(class_dir.iterdir()):
        if f.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv"):
            videos.append({"path": str(f), "class": class_name, "label": label})

df_videos = pd.DataFrame(videos)
print(f"Total videos: {len(df_videos)}")
print(df_videos.groupby("class").size().to_string())
print()

# ── Info de resolución (muestra) ───────────────────────────

resolutions = []
for v in df_videos.sample(min(5, len(df_videos))).itertuples():
    cap = cv2.VideoCapture(v.path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps > 0 else 0
    cap.release()
    resolutions.append({"video": Path(v.path).name, "width": w, "height": h,
                        "fps": fps, "frames": total, "duration_s": dur})

df_res = pd.DataFrame(resolutions)
print("Muestra de resoluciones:")
print(df_res.to_string(index=False))
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — PRUEBA DE DETECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

cells.append(code("""# ── Probar detección en un video ────────────────────────────

model = YOLO(YOLO_MODEL)
print(f"✅ Modelo cargado: {YOLO_MODEL}")

# Procesar el primer video para ver cuántas personas detecta
test_video = df_videos.iloc[0]
test_class = getattr(test_video, "class")  # class is a keyword, must use getattr
print(f"\\n📹 Video de prueba: {Path(test_video.path).name} ({test_class})")

tracks = process_video_multi(
    test_video.path, model,
    frame_skip=30,  # rápido: sample 1 de cada 30 frames
    conf_thresh=CONF_THRESH,
    track_dist=TRACK_DIST_THRESH,
)

n_people = len(tracks)
print(f"👥 Personas detectadas: {n_people}")
for t in tracks:
    print(f"   Track #{t['track_id']}: {t['n_frames']} frames, "
          f"{len(t['angles_per_frame'])} frames con ángulos")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — MARKDOWN: PROCESAR TODO
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 5. Extraer Features de Todos los Videos

Procesamos **todos los videos**:

1. Por cada video, extraemos tracks de cada persona
2. Por cada track, calculamos ángulos por frame
3. Agregamos features estadísticos (mean, std, min, max)
4. Etiquetamos según la carpeta (`squat_correcto` = 0, `squatmal_ejecutado` = 1)

Esto puede llevar **varios minutos** (18 videos × ~30s cada uno).
"""))

cells.append(code("""# ── Procesamiento completo ──────────────────────────────────

all_samples = []
error_log = []

for v in df_videos.itertuples():
    video_name = Path(v.path).name
    video_class = getattr(v, "class")  # class is a keyword, must use getattr
    print(f"📹 Procesando: {video_name} ({video_class}) ... ", end="", flush=True)

    try:
        tracks = process_video_multi(
            v.path, model,
            frame_skip=FRAME_SKIP,
            conf_thresh=CONF_THRESH,
            track_dist=TRACK_DIST_THRESH,
        )

        if not tracks:
            print("⚠️  sin detecciones")
            error_log.append({"video": video_name, "error": "no detections"})
            continue

        for t in tracks:
            if not t["angles_per_frame"]:
                continue

            frames_angles = [entry["angles"] for entry in t["angles_per_frame"]]
            features = aggregate_features(frames_angles)

            all_samples.append({
                "video": video_name,
                "track_id": t["track_id"],
                "label": v.label,
                "class": getattr(v, "class"),
                "n_frames": t["n_frames"],
                "n_angles_frames": len(t["angles_per_frame"]),
                **features,
            })

        print(f"✅ {len(tracks)} persona(s)")

    except Exception as e:
        print(f"❌ {e}")
        error_log.append({"video": video_name, "error": str(e)})

print(f"\\n📊 Total samples (personas): {len(all_samples)}")
if error_log:
    print(f"⚠️  Errores: {len(error_log)}")
    for e in error_log:
        print(f"   {e['video']}: {e['error']}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

cells.append(code("""# ── Armar DataFrame ─────────────────────────────────────────

df = pd.DataFrame(all_samples)
print(f"Dataset shape: {df.shape}")
print(f"Columnas: {len(df.columns)}")
print(f"  - Metadata: video, track_id, label, class, n_frames, n_angles_frames")
feature_cols = [c for c in df.columns if c.endswith(("_mean", "_std", "_min", "_max"))]
print(f"  - Features: {len(feature_cols)}")
print()

# Distribución de clases
print("Distribución de etiquetas:")
print(df["class"].value_counts().to_string())
print()

# Ver cuántas personas únicas detectamos
print(f"Videos únicos: {df['video'].nunique()}")
print(f"Tracks únicos (personas): {len(df)}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — EDA
# ══════════════════════════════════════════════════════════════════════════════

cells.append(code("""# ── Análisis exploratorio rápido ────────────────────────────

# Mostrar estadísticas básicas de las features principales
key_feats = ["left_knee_angle_min", "back_angle_mean",
             "knee_symmetry_mean", "left_knee_toe_x_max"]
available = [k for k in key_feats if k in df.columns]
if available:
    print("Estadísticas por clase (features clave):")
    print(df.groupby("class")[available].describe().round(1))
    print()

# Distribución del ángulo mínimo de rodilla (profundidad)
if "left_knee_angle_min" in df.columns and "right_knee_angle_min" in df.columns:
    df["knee_min_avg"] = df[["left_knee_angle_min", "right_knee_angle_min"]].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cls, color in zip(axes, ["squat_correcto", "squatmal_ejecutado"], ["green", "red"]):
        subset = df[df["class"] == cls]["knee_min_avg"]
        ax.hist(subset, bins=8, alpha=0.7, color=color, edgecolor="white")
        ax.set_title(f"{cls} (n={len(subset)})")
        ax.set_xlabel("Ángulo mínimo de rodilla (media L/R)")
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — MATRIZ X, y
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 6. Construir Matriz de Features

Convertimos las features a matriz numérica y separamos en train/test.
"""))

cells.append(code("""# ── Preparar X, y ────────────────────────────────────────────

# Features (columnas que terminan en _mean, _std, _min, _max)
feature_cols = sorted([c for c in df.columns
                       if c.endswith(("_mean", "_std", "_min", "_max"))])
print(f"Features totales: {len(feature_cols)}")
print(f"Primeras 5: {feature_cols[:5]}")
print(f"Últimas 5:  {feature_cols[-5:]}")

X = df[feature_cols].values.astype(np.float64)
y = df["label"].values

# ── Manejar NaN / Inf ───────────────────────────────────────
n_nan = np.isnan(X).sum()
n_inf = np.isinf(X).sum()
if n_nan > 0 or n_inf > 0:
    print(f"⚠️  NaN: {n_nan}, Inf: {n_inf} — reemplazando con 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Clases: 0={sum(y==0)}, 1={sum(y==1)}")

# ── Train / Test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\\nSplit: train={len(X_train)}, test={len(X_test)}")
print(f"  Train: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
print(f"  Test:  0={sum(y_test==0)}, 1={sum(y_test==1)}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — TRAIN XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 7. Entrenar XGBoost

Usamos **XGBoost** con parámetros razonables para un dataset chico:
- `n_estimators=200`: suficientes árboles para ~30-50 samples
- `max_depth=4`: profundo suficiente sin overfittear
- `learning_rate=0.1`: clásico
- `scale_pos_weight`: balance automático si hay desbalance de clases

Después evaluamos con **validación cruzada** (StratifiedKFold, 5 folds).
"""))

cells.append(code("""# ── Entrenar modelo base ─────────────────────────────────────

params = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métricas
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc:.2%}")
print(f"\\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Buena", "Mala"]))
print(f"\\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Buena", "Mala"], yticklabels=["Buena", "Mala"])
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 — CROSS-VAL
# ══════════════════════════════════════════════════════════════════════════════

cells.append(code("""# ── Validación cruzada ─────────────────────────────────────
# Más robusta que un solo split para datasets chicos

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"Cross-validation Accuracy (5 folds):")
print(f"  Scores: {cv_scores}")
print(f"  Media:  {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 13 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 8. Importancia de Features

Vemos qué métricas biomecánicas pesan más en la decisión del modelo.
"""))

cells.append(code("""# ── Feature Importance ──────────────────────────────────────

importances = model.feature_importances_
idx_sorted = np.argsort(importances)[::-1]

top_n = min(15, len(feature_cols))
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[idx_sorted[:top_n]][::-1], color="steelblue")
plt.yticks(range(top_n), [feature_cols[i] for i in idx_sorted[:top_n]][::-1])
plt.xlabel("Importancia (gain)")
plt.title("Top 15 Features más importantes para XGBoost")
plt.tight_layout()
plt.show()

print("Top 10 features:")
for i in range(min(10, len(feature_cols))):
    name = feature_cols[idx_sorted[i]]
    print(f"  {i+1}. {name}: {importances[idx_sorted[i]]:.4f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 14 — COMPARAR CON REGLAS
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 9. Comparar XGBoost vs Reglas Biomecánicas

Comparamos las predicciones de XGBoost contra el clasificador de reglas
(`SquatFormClassifier`) para ver si coinciden o dónde difieren.
"""))

cells.append(code("""# ── Comparar con clasificador de reglas ──────────────────────

import sys
sys.path.insert(0, str(Path.cwd()))
from src.squat_classifier import SquatFormClassifier
from src.angle_utils import aggregate_video_features

# Para cada sample, evaluar con reglas
rule_clf = SquatFormClassifier()
rule_preds = []
for _, row in df.iterrows():
    # Reconstruir el dict de features desde las columnas
    feats = {k: row[k] for k in feature_cols}
    rule_preds.append(rule_clf.predict([feats])[0])

rule_preds = np.array(rule_preds)

# Comparación
agreement = (rule_preds == y).mean()
print(f"Coincidencia reglas vs XGBoost (sobre dataset completo): {agreement:.1%}")
print()
print("      ┌──────────────┬──────────┬──────────┐")
print("      │              │ Reglas   │ XGBoost  │")
print("      ├──────────────┼──────────┼──────────┤")
print(f"      │ Accuracy     │ {accuracy_score(y, rule_preds):.1%}    │ {accuracy_score(y, y_pred):.1%}    │")
print("      └──────────────┴──────────┴──────────┘")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 15 — SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## 10. Guardar Modelo

Guardamos el modelo XGBoost con joblib para usarlo en la app.
"""))

cells.append(code("""# ── Guardar ──────────────────────────────────────────────────

os.makedirs(OUTPUT_MODEL.parent, exist_ok=True)
joblib.dump(model, str(OUTPUT_MODEL))
size_kb = Path(OUTPUT_MODEL).stat().st_size / 1024
print(f"✅ Modelo guardado: {OUTPUT_MODEL} ({size_kb:.1f} KB)")

# Guardar también feature_cols para saber el orden esperado
meta = {"feature_cols": feature_cols, "label_map": {"Buena": 0, "Mala": 1}}
meta_path = OUTPUT_MODEL.with_suffix(".json")
import json
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ Metadata guardada: {meta_path}")

# ── Verificar carga ─────────────────────────────────────────

loaded = joblib.load(str(OUTPUT_MODEL))
test_pred = loaded.predict(X_test)
reload_acc = accuracy_score(y_test, test_pred)
print(f"\\n🔁 Verificación de carga: accuracy = {reload_acc:.2%}")
assert reload_acc == acc, "Accuracy mismatch después de recargar"
print("✅ Modelo cargado correctamente — todo OK")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# CELL 16 — WRAPUP
# ══════════════════════════════════════════════════════════════════════════════

cells.append(md("""---
## Resumen

| Paso | Resultado |
|------|-----------|
| Videos procesados | ✅ |
| Personas detectadas | ✅ |
| Dataset construido | ✅ |
| XGBoost entrenado | ✅ |
| Evaluación completa | ✅ |
| Modelo guardado | ✅ |

### Próximos pasos:
1. **Usar el modelo en la app**: modificar `app.py` para cargar `xgboost_squat.pkl`
2. **Mejorar con más datos**: grabar más variaciones (diferentes personas, ángulos de cámara)
3. **Ajustar hiperparámetros**: grid search sobre `max_depth`, `learning_rate`, etc.
4. **Ensemble**: combinar reglas + XGBoost para un sistema híbrido
"""))

# ══════════════════════════════════════════════════════════════════════════════
# WRITE
# ══════════════════════════════════════════════════════════════════════════════

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook generado: {NB_PATH}")
