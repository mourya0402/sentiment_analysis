
from sentiment import analyze_batch, load_pipeline

def test_positive():
    pipe = load_pipeline()
    res = analyze_batch(["I absolutely love this!"], pipe=pipe, neutral_margin=0.1)[0]
    assert res["label"] in ("POSITIVE", "NEUTRAL")
    assert 0.0 <= res["score"] <= 1.0

def test_negative():
    pipe = load_pipeline()
    res = analyze_batch(["This is terrible and disappointing."], pipe=pipe, neutral_margin=0.1)[0]
    assert res["label"] in ("NEGATIVE", "NEUTRAL")
    assert 0.0 <= res["score"] <= 1.0
