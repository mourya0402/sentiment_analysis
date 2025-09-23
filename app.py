import time
import gradio as gr
from sentiment import analyze_batch, load_pipeline

# Optional resource metrics (install if you want: pip install psutil)
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
_is_warm = False  # track cold vs warm calls for latency reporting

def predict(text, neutral_margin):
    global pipe, _is_warm

    if not text or not text.strip():
        return [{"label": "NEUTRAL", "score": 1.0}], "Please enter some text."

    cold = False
    if pipe is None:
        cold = True
        pipe = load_pipeline()

    # timing start
    t0 = time.perf_counter()
    results = analyze_batch([text], pipe=pipe, neutral_margin=neutral_margin)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # optional resource metrics
    metrics = []
    if psutil is not None:
        p = psutil.Process()
        try:
            rss_mb = p.memory_info().rss / (1024 ** 2)
            # cpu_percent(None) returns the % since last call; fine for a quick sample
            cpu_pct = p.cpu_percent(interval=0.0)
            metrics.append(f"RSS: {rss_mb:.1f} MB")
            metrics.append(f"CPU: {cpu_pct:.1f}%")
        except Exception:
            pass
    else:
        metrics.append("(Install 'psutil' to see CPU/RAM)")

    # pretty output (your original, plus timing)
    pretty_lines = [f"**{r['label']}** (conf: {r['score']:.3f})" for r in results]
    mode = "cold start" if cold and not _is_warm else "warm"
    pretty_lines.append(f"\n**Latency:** {dt_ms:.1f} ms ({mode})")
    pretty_lines.append("  " + " | ".join(metrics))

    _is_warm = True  # future calls are warm
    return results, "\n".join(pretty_lines)

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

if __name__ == "__main__":
    demo.launch()
