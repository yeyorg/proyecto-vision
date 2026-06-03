# Squat Form Analyzer

[![GitHub](https://img.shields.io/badge/github-yeyorg/proyecto--vision-blue?logo=github)](https://github.com/yeyorg/proyecto-vision)

> Analizá la técnica de tus sentadillas subiendo un video —
> YOLOv8-pose detecta keypoints, calcula ángulos biomecánicos
> y los evalúa contra rangos óptimos con reglas interpretables.

---

## Tabla de contenidos

- [Descripcion](#descripcion)
- [Cambios recientes](#cambios-recientes)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalacion](#instalacion)
- [Uso](#uso)
- [Comandos utiles](#comandos-utiles)
- [Comandos utiles (Windows sin make)](#comandos-utiles-windows-sin-make)
- [Modelo](#modelo)
- [Extender el proyecto](#extender-el-proyecto)
- [Troubleshooting](#troubleshooting)
- [Licencia](#licencia)

---

## Descripcion

Este proyecto analiza la forma de sentadillas en video usando **YOLOv8-pose**
para detectar 17 keypoints del cuerpo, calcula **angulos biomecanicos**
especificos (rodilla, cadera, espalda, simetria) y los evalua contra rangos
optimos basados en la literatura de biomecanica.

A diferencia de enfoques puramente basados en ML que necesitan cientos de
videos etiquetados, este clasificador usa **reglas interpretables** que
funcionan desde el primer momento. Cada metrica tiene feedback especifico
("angulo de rodilla: 125 -- no llegaste a paralelo").

### Que evalua

| Criterio | Peso | Que mide |
| --- | --- | --- |
| Depth (profundidad) | 2x | Angulo minimo de rodilla, ideal ~90 |
| Back angle (espalda) | 1x | Inclinacion del torso respecto a vertical |
| Knee tracking | 1x | Desplazamiento horizontal rodilla sobre tobillo |
| Symmetry (simetria) | 1x | Diferencia entre lado izquierdo y derecho |
| Stability (rango) | 0.5x | Rango de movimiento completo |

---

## Cambios recientes

### Eliminado: modo tiempo real y detección por webcam

Se removió toda la funcionalidad de análisis en vivo para simplificar el proyecto
a un solo flujo: **subir video → analizar**.

| Lo que se fue | Por qué |
|---|---|
| `realtime.py` (443 lines) | Ventana OpenCV con webcam, detección de fases y scoring en vivo |
| `SquatDetector` + `SquatPhase` | Máquina de estados que clasificaba fases (de pie/bajando/fondo/subiendo) — solo la usaba `realtime.py` |
| Webcam en `app.py` | Opción "Webcam (foto)" y botones de "Tiempo Real" en la sidebar |
| `make realtime` | Target del Makefile |

El proyecto ahora se enfoca exclusivamente en el análisis de **videos grabados**
(subidos por el usuario), sin depender de la cámara ni de procesamiento en tiempo real.

### Flujo actual

```
Subís un video MP4 → YOLOv8-pose detecta keypoints →
se calculan ángulos biomecánicos por frame →
se agregan estadísticos (mean, std, min, max) →
SquatFormClassifier evalúa con reglas →
score 0-100 + feedback personalizado
```

---

## Arquitectura

```
     video (MP4 subido)
           |
           v
 +------------------+
 |  PoseExtractor   |  YOLOv8-pose -> 17 keypoints COCO
 +------------------+
           |
           v
 +------------------+
 |  get_squat_angles |  Calcula angulos por frame
 +------------------+   (rodilla, cadera, espalda, simetria)
           |
           v
 +------------------+
 | aggregate_video   |  Estadisticos: mean, std, min, max
 | _features         |
 +------------------+
           |
           v
 +------------------+
 | SquatForm         |  Reglas biomecanicas -> score + feedback
 | Classifier        |
 +------------------+
           |
           v
    score 0-100 + consejos
```

### Modulos

| Archivo | Responsabilidad |
| --- | --- |
| `src/pose_extractor.py` | Wrapper de YOLOv8-pose. Procesa frames y videos. |
| `src/angle_utils.py` | Calculos geometricos: angulos, features agregadas, explicaciones. |
| `src/squat_classifier.py` | Clasificador basado en reglas. Interfaz `predict()` / `predict_proba()`. |
| `app.py` | Aplicacion Streamlit con upload de video y visualizacion. |
| `squat_form.ipynb` | Notebook interactivo para exploracion y pruebas. |

---

## Reglas de evaluación

El `SquatFormClassifier` evalúa **5 criterios** con distintos pesos.
Cada criterio usa un estadístico distinto (`min`, `mean`, `max`) según
lo que tenga sentido biomecánico para esa métrica.

El **score global** es un promedio ponderado:

```
overall = sum(score_i * weight_i) / sum(weight_i)
```

Un score >= 60 = buena forma, < 60 = necesita trabajo.

### 1. Depth (profundidad) — peso 2x

Usa el **ángulo MÍNIMO** de rodilla alcanzado en todo el video
(`knee_angle_min`). Se usa el mínimo porque el momento más profundo
de la sentadilla es el que define si llegaste a paralelo o no.

| Ángulo mínimo | Score | Significado |
|---|---|---|
| ~90° (ideal) | 100 | Paralelo perfecto |
| 70°–110° | 40–100 | Dentro del rango aceptable |
| < 70° | 0–40 | Hiperflexión / butt wink |
| > 110° | 0–40 | No llegaste a paralelo |

```
Score
 100 │     ╱╲
  80 │    ╱  ╲
  60 │   ╱    ╲
  40 │  ╱      ╲
  20 │ ╱        ╲
   0 ╱╲__________╲╲__
     60  70  90  110 130
           └─ ideal ─┘
         └── rango ────┘
```

### 2. Back angle (espalda) — peso 1x

Usa el **PROMEDIO** de inclinación del torso (`back_angle_mean`).
Se usa el promedio porque la inclinación debería ser consistente
a lo largo del ejercicio; picos aislados no son representativos.

| Inclinación | Score | Significado |
|---|---|---|
| ~35° (ideal) | 100 | Rango normal para barra alta |
| 15°–50° | 40–100 | Aceptable |
| < 15° | 0–40 | Muy vertical, pérdida de ventaja mecánica |
| > 50° | 0–40 | Riesgo lumbar, excesiva inclinación |

### 3. Knee tracking (rodilla sobre tobillo) — peso 1x

Usa el **MÁXIMO ABSOLUTO** de desplazamiento horizontal
(`knee_toe_x_max` y `knee_toe_x_min`). Se usa el valor pico porque
una sola vez que la rodilla se vaya demasiado adelante ya es un
riesgo de lesión — el promedio no capturaría esto.

| Máx desplazamiento | Score | Significado |
|---|---|---|
| < 40 px | 70–100 | Controlado |
| 40–60 px | 30–70 | Se fue un poco |
| > 60 px | 0–30 | Rodilla en riesgo |

### 4. Symmetry (simetría) — peso 1x

Usa el **PROMEDIO** de la diferencia entre piernas
(`knee_symmetry_mean`, `hip_symmetry_mean`). Una diferencia sostenida
indica desbalance muscular o compensación.

| Asimetría | Score | Significado |
|---|---|---|
| < 5° | ~95 | Movimiento balanceado |
| 5°–12° | 70–95 | Asimetría leve |
| > 12° | 0–70 | Desbalance significativo |

### 5. Stability (rango de movimiento) — peso 0.5x

Usa el **RANGO** de movimiento de rodilla
(`knee_angle_max - knee_angle_min`). Mide si estás haciendo el
recorrido completo de una sentadilla o solo media repetición.

| Rango | Score | Significado |
|---|---|---|
| > 60° | 100 | Sentadilla completa |
| 40°–60° | 70–100 | Rango aceptable |
| 20°–40° | 30–70 | Media sentadilla |
| < 20° | 0–30 | Casi no hay flexión |

### Ejemplo concreto

Para un video donde:
- Rodilla mínima: 88° → **Depth = 95**
- Espalda promedio: 32° → **Back = 85**
- Máx desplazamiento: 25 px → **Knee = 80**
- Asimetría promedio: 4° → **Sym = 90**
- Rango de movimiento: 65° → **Stability = 100**

```
overall = (95×2 + 85×1 + 80×1 + 90×1 + 100×0.5) / (2 + 1 + 1 + 1 + 0.5)
       = (190 + 85 + 80 + 90 + 50) / 5.5
       = 495 / 5.5
       = 90  ✅ Buena forma
```

---

## Requisitos

- **Python** >= 3.12
- **uv** >= 0.5 (instalalo con `pip install uv` o desde <https://docs.astral.sh/uv/>)
- Opcional pero recomendado: **Jupyter** para el notebook
- Opcional: **make** (Linux/macOS) o **Chocolatey** (Windows)

---

## Instalacion

```bash
# 1. Clonar el repo
git clone https://github.com/yeyorg/proyecto-vision.git
cd proyecto-vision

# 2. Crear entorno virtual e instalar dependencias
uv venv
uv sync

# 3. Entrenar el modelo
uv run python train_model.py
```

Listo. No hace falta activar el venv -- `uv run` lo usa automagicamente.
Si preferis activarlo: `.venv\Scripts\activate` (Windows) o
`source .venv/bin/activate` (Linux/macOS).

---

## Uso

### App Streamlit

```bash
uv run streamlit run app.py
```

Se abre en el navegador. Subi un video MP4 para analizar.

### Notebook

```bash
uv run jupyter notebook squat_form.ipynb
```

Exploracion interactiva: proba la extraccion de pose, calcula angulos, evalua
sentadillas sinteticas.

### Entrenar modelo

```bash
uv run python train_model.py
```

Genera `models/squat_form_model.pkl`. El clasificador usa reglas biomecanicas,
no necesita datos de entrenamiento.

---

## Comandos utiles

```bash
make help       # Muestra todos los comandos disponibles
make run        # Inicia la app Streamlit
make notebook   # Abre el notebook
make train      # Entrena y guarda el modelo
make check      # Verifica que todo funciona
make clean      # Limpia archivos generados
make deep-clean # Limpia todo incluyendo .venv
```

---

## Comandos utiles (Windows sin make)

```powershell
# Iniciar la app
uv run streamlit run app.py

# Entrenar modelo
uv run python train_model.py

# Verificar instalacion
uv run python -c "from src.pose_extractor import PoseExtractor; print('OK')"

# Re-instalar dependencias si cambia pyproject.toml
uv sync

# Agregar una nueva dependencia
uv add <paquete>

# Limpiar
Remove-Item -Recurse -Force __pycache__, src\__pycache__, .ipynb_checkpoints -ErrorAction SilentlyContinue
```

---

## Modelo

El modelo guardado (`models/squat_form_model.pkl`) es un `SquatFormClassifier`
que implementa la interfaz scikit-learn:

```python
import joblib

model = joblib.load("models/squat_form_model.pkl")

# features: dict con estadisticos de angulos
prediccion = model.predict([features])       # 0 = buena, 1 = mala
probabilidad = model.predict_proba([features])  # [[good_prob, bad_prob]]
```

### Por que reglas en vez de XGBoost?

El notebook original (`exercise_form.ipynb`) usa XGBoost entrenado con datos
etiquetados. El problema: **sin datos, no hay modelo**. Nuestro enfoque:

| Ventaja | Reglas biomecanicas | XGBoost |
| --- | --- | --- |
| Funciona sin datos | SI | NO |
| Feedback interpretable | "rodilla a 125" | "clase 1" |
| Ajustable | Editar umbrales | Re-entrenar |
| Tamaño | 0.3 KB | ~MB |

**Proximo paso:** Si juntas videos etiquetados, las features de
`angle_utils.py` sirven directamente para entrenar un XGBoost. La interfaz
`predict()` es identica -- no cambia ni una linea de la app.

---

## Extender el proyecto

### Agregar un nuevo ejercicio (ej: peso muerto)

1. En `angle_utils.py`, crear `get_deadlift_angles()`
2. En `squat_classifier.py`, crear `_score_deadlift_*()` o un clasificador nuevo
3. En `app.py`, agregar un selector de ejercicio

### Ajustar umbrales

```python
from src.squat_classifier import SquatFormClassifier

clf = SquatFormClassifier(thresholds={
    "knee_angle_min_range": (75.0, 105.0),  # mas estricto
    "good_form_threshold": 70.0,            # corte mas alto
})
```

### Entrenar XGBoost con datos reales

```python
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Cargar features de tus videos etiquetados
# X = np.array([...])  # features de cada video
# y = np.array([...])  # 0 = buena, 1 = mala

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = XGBClassifier()
# model.fit(X_train, y_train)
# joblib.dump(model, "models/xgboost_model.pkl")
```

---

## Troubleshooting

### "No module named 'ultralytics'"

```bash
uv sync
```

Si el paquete no esta en `pyproject.toml`:

```bash
uv add ultralytics
```

### "No se detectaron personas en el video"

- Asegurate de estar de frente o de perfil a la camara
- El video debe tener buena iluminacion
- Probá con el notebook para ver si YOLO detecta keypoints

### El score de mala forma es muy alto/bajo

Ajusta los umbrales en `squat_classifier.py`:

```python
DEFAULT_THRESHOLDS = {
    "good_form_threshold": 60.0,  # subi a 70 para mas exigencia
    "knee_angle_min_range": (70.0, 110.0),
    "back_angle_mean_range": (15.0, 50.0),
}
```

Corre `uv run python train_model.py` para actualizar el modelo.

---

## Licencia

Proyecto basado en el notebook de Science Buddies
"[Using AI to Detect Proper Exercise Form](https://www.sciencebuddies.org/science-fair-projects/project-ideas/ArtificialIntelligence_p027/artificial-intelligence/exercise_form)".
Para uso personal y educativo.

---

*Ultima actualizacion: 2026-06-02*
