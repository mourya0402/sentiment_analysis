
import gradio as gr
from sentiment import analyze_batch, load_pipeline

INTRO = """
# Text Sentiment Analyzer (Local)
Runs **fully local** using a Hugging Face Transformers pipeline—no external APIs.
Default model: `distilbert-base-uncased-finetuned-sst-2-english`.
"""

pipe = None

def predict(text, neutral_margin):
    global pipe
    if not text or not text.strip():
        return [{"label": "NEUTRAL", "score": 1.0}], "Please enter some text."
    if pipe is None:
        pipe = load_pipeline()
    results = analyze_batch([text], pipe=pipe, neutral_margin=neutral_margin)
    pretty = [f"**{r['label']}** (conf: {r['score']:.3f})" for r in results]
    return results, "\n".join(pretty)

with gr.Blocks() as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input text", placeholder="Paste a review, tweet, etc.", lines=6)
            neutral_margin = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="Neutral margin (wider margin => more NEUTRAL)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_json = gr.JSON(label="Raw output")
            out_md = gr.Markdown(label="Friendly view")
    btn.click(predict, inputs=[inp, neutral_margin], outputs=[out_json, out_md])

if __name__ == "__main__":
    demo.launch()
