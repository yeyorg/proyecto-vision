# ─────────────────────────────────────────────────────────────
# Squat Form Analyzer — Makefile (uv edition)
# ─────────────────────────────────────────────────────────────
# Requiere: uv (https://docs.astral.sh/uv/)
# ─────────────────────────────────────────────────────────────

.PHONY: help setup install sync run notebook test check lint clean deep-clean

help: ## Mostrar ayuda con todos los comandos
	@type Makefile 2>nul | findstr /r "^[a-z].*:.*##" | powershell -Command \
		"$$input | ForEach-Object { \
			$$parts = $$_ -split '## '; \
			$$cmd = ($$parts[0] -split ':')[0].Trim(); \
			$$desc = $$parts[1].Trim(); \
			Write-Host (\"  make {0,-12} {1}\" -f $$cmd,$$desc) \
		}"
	@echo.

# ── Entorno ─────────────────────────────────────────────────

.venv:
	uv venv

setup: .venv sync ## Crear entorno virtual e instalar dependencias
	@echo ""
	@echo "Entorno listo. Activalo con:"
	@echo "  Windows:     .venv\Scripts\activate"
	@echo "  Linux/macOS: source .venv/bin/activate"
	@echo ""

sync: ## Sincronizar dependencias desde pyproject.toml
	uv sync

install: sync ## Alias para sync

# ── Ejecucion ───────────────────────────────────────────────

run: ## Correr la app Streamlit
	uv run streamlit run app.py

notebook: ## Iniciar Jupyter Notebook (exploración)
	uv run jupyter notebook squat_form.ipynb

train-xgb: ## Entrenar XGBoost desde el notebook
	uv run jupyter nbconvert --to notebook --execute --inplace xgboost_training.ipynb

# ── Tests ───────────────────────────────────────────────────

test: ## Correr tests unitarios del clasificador
	uv run --group dev pytest tests/ -v

# ── Mantenimiento ───────────────────────────────────────────

clean: ## Limpiar archivos generados
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -f test_videos/*_annotated.mp4
	@echo "Cache y archivos temporales eliminados"

deep-clean: clean ## Limpiar todo, incluyendo .venv y uv.lock
	rm -rf .venv
	rm -f uv.lock
	rm -rf models/
	@echo "Entorno y lock eliminados"
	@echo "ATENCION: corre 'make setup' para recrear todo"

# ── Calidad ─────────────────────────────────────────────────

lint: ## Verificar sintaxis de Python
	uv run python -m py_compile src/squat_classifier.py
	uv run python -m py_compile src/angle_utils.py
	uv run python -m py_compile src/pose_extractor.py
	uv run python -m py_compile app.py
	uv run python -m py_compile tests/test_classifier.py
	@echo "Todo compila sin errores"

check: ## Verificar que todo funciona
	uv run python scripts/check.py
