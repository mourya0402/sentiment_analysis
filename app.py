from prometheus_client import Counter, Gauge, Histogram, start_http_server
import threading
import time
import os
import gradio as gr
from sentiment import analyze_batch, load_pipeline

# Optional resource metrics 
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

INTRO = """
# Text Sentiment Analyzer (Local)
Runs **fully local** using a Hugging Face Transformers pipeline—no external APIs.
Default model: `distilbert-base-uncased-finetuned-sst-2-english`.
"""

pipe = None
_is_warm = False

# Prometheus metrics
REQ_COUNT = Counter("sentiment_requests_total", "Total number of predictions made")
REQ_LATENCY = Histogram("sentiment_request_latency_seconds", "Latency for prediction calls")
CPU_USAGE = Gauge("sentiment_cpu_usage_percent", "Current CPU usage percent")
MEM_USAGE = Gauge("sentiment_memory_mb", "Resident memory in MB")

@REQ_LATENCY.time()  # measure duration automatically
def predict(text, neutral_margin):
    global pipe, _is_warm
    if not text or not text.strip():
        return [{"label": "NEUTRAL", "score": 1.0}], "Please enter some text."

    cold = False
    if pipe is None:
        cold = True
        pipe = load_pipeline()

    REQ_COUNT.inc()

    t0 = time.perf_counter()
    results = analyze_batch([text], pipe=pipe, neutral_margin=neutral_margin)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Collect optional resource metrics text for the UI
    metrics = []
    if psutil is not None:
        try:
            p = psutil.Process()
            cpu_now = p.cpu_percent(interval=0.0)
            mem_now = p.memory_info().rss / (1024 ** 2)
            CPU_USAGE.set(cpu_now)
            MEM_USAGE.set(mem_now)
            metrics.append(f"CPU: {cpu_now:.1f}%")
            metrics.append(f"RSS: {mem_now:.1f} MB")
        except Exception:
            pass
    else:
        metrics.append("(Install 'psutil' to see CPU/RAM)")

    # pretty output (your original, plus timing)
    pretty_lines = [f"**{r['label']}** (conf: {r['score']:.3f})" for r in results]
    mode = "cold start" if cold and not _is_warm else "warm"
    pretty_lines.append(f"\n**Latency:** {dt_ms:.1f} ms ({mode})")
    if metrics:
        pretty_lines.append("  " + " | ".join(metrics))

    _is_warm = True
    return results, "\n".join(pretty_lines)

# --- UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input text", placeholder="Paste a review, tweet, etc.", lines=6)
            neutral_margin = gr.Slider(0.0, 0.5, value=0.15, step=0.01,
                                       label="Neutral margin (wider margin ⇒ more NEUTRAL)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_json = gr.JSON(label="Raw output")
            out_md = gr.Markdown(label="Friendly view")
    btn.click(predict, inputs=[inp, neutral_margin], outputs=[out_json, out_md])

# --- Main entry ---
if __name__ == "__main__":
    # Start Prometheus metrics server in background
    threading.Thread(target=start_http_server, args=(8000,), daemon=True).start()

    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
