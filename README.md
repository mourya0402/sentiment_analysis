# Text Sentiment Analyzer (Local)

A **simple local** Gradio app that classifies text sentiment using a Hugging Face Transformers pipeline
(`distilbert-base-uncased-finetuned-sst-2-english`). No remote APIs.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the printed local URL and paste any text. Use the **Neutral margin** slider to make the
classifier return **NEUTRAL** when POS vs NEG is too close to call.

## Why this fits Deliverable 2

- **Runs locally** on your hardware (no `InferenceClient`, no external API).
- Clear contrast vs. your API-based product from Deliverable 1.
- Includes **tests** and a **GitHub Action** that runs `pytest` on push/PR.

## Project layout

```
sentiment_local/
├─ app.py
├─ sentiment.py
├─ requirements.txt
├─ README.md
├─ tests/
│  └─ test_sentiment.py
└─ .github/workflows/
   └─ ci.yml
```

## Notes

- On the first run, Transformers will **download the model weights** to your machine
  (cached under `~/.cache/huggingface/`). Subsequent runs are fully local.
- If you need strictly offline installs, pre-download the model:
  ```bash
  python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
             AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); \
             AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')"
  ```
push-trigger-test
