What it does 

Classifies short text as POSITIVE, NEGATIVE, or NEUTRAL

Runs locally (CPU by default) – no external API calls after the first model download

Simple web UI (Gradio) + clean Python module you can import in other code

Unit tests with pytest and GitHub Actions that run on every push/PR



python -m pip install --upgrade pip

pip install -r requirements.txt


Project structure

.

├─ app.py                 # Gradio app (UI)

├─ sentiment.py           # Model loading + batch analysis helpers

├─ requirements.txt       # Python dependencies

├─ pytest.ini             # pytest config

├─ tests/

│  ├─ test_imports.py     # sanity checks

│  └─ test_sentiment.py   # simple positive/negative assertions

└─ .github/workflows/ci.yml   # GitHub Actions: install + run tests
