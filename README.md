```markdown
# Automatización de pipeline ML con GitHub Actions

Este repositorio contiene un pipeline reproducible de Machine Learning que:
- Descarga el dataset Wine Quality (UCI).
- Preprocesa los datos, entrena un RandomForestClassifier.
- Evalúa (Accuracy, F1).
- Registra parámetros, métricas, firma y ejemplo de entrada con MLflow.
- Guarda el modelo como artefacto.
- Automatiza la ejecución mediante GitHub Actions (CI/CD).

Estructura
- src/train.py         -> script principal del pipeline
- config.yaml          -> configuración de hiperparámetros y rutas
- Makefile             -> tareas (install, train, test)
- .github/workflows/ml.yml -> CI/CD workflow
- tests/               -> pruebas básicas con pytest
- mlruns/              -> directorio de tracking (generado al entrenar)
- models/              -> modelo serializado (generado al entrenar)

Instalación y ejecución local (Linux / macOS / WSL)
1. Clona el repo:
   git clone <TU-REPO-URL>
   cd <TU-REPO>

2. Crear un virtualenv e instalar dependencias:
   python -m venv .venv
   source .venv/bin/activate
   make install
   # make install ahora también instala el paquete del proyecto en modo editable (pip install -e .)

3. Ejecutar training (usa config.yaml):
   make train
   o
   python src/train.py --config config.yaml

4. Ejecutar tests:
   make test

Abrir MLflow UI (para ver runs y modelos registrados)
- Desde la raíz del repo:
  mlflow ui --backend-store-uri file://$(pwd)/mlruns --host 0.0.0.0 --port 5000
- Luego abre http://localhost:5000

Notas sobre CI/CD (GitHub Actions)
- El workflow .github/workflows/ml.yml hace:
  - checkout
  - setup Python
  - make install (ahora instala el paquete editable)
  - lint (flake8)
  - make test
  - make train
  - sube los directorios mlruns/ y models/ como artefactos del run

Sugerencias
- Si no quieres usar pip install -e ., puedes mantener tests/conftest.py; es un parche rápido que añade la raíz del repo a PYTHONPATH durante pytest.
- Para un entorno de producción o empaquetado, considera agregar metadata adicional en setup.py (author, license, classifiers).
```