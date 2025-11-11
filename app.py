# ... imports ...
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

# (unchanged INTRO / globals ...)
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
            CPU_USAGE.set(p.cpu_percent(interval=0.0))
            MEM_USAGE.set(p.memory_info().rss / (1024 ** 2))
            metrics.append(f"CPU: {p.cpu_percent(interval=0.0):.1f}%")
            metrics.append(f"RSS: {p.memory_info().rss / (1024 ** 2):.1f} MB")
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

with gr.Blocks() as demo:
    # ... your UI code unchanged ...
    pass  # keep your existing Blocks code here

if __name__ == "__main__":
    # start prometheus metrics server on port 8000 in background
    threading.Thread(target=start_http_server, args=(8000,), daemon=True).start()

    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.queue(concurrency_count=2).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
