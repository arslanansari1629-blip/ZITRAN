"""ZITRAN Ensemble: weighted probability averaging of v1 (free-text),
v2 (real telemetry) and v3 (ultra, calibrated). Weights = each model's
validation accuracy from its metrics.json (fallback: equal). v3's
temperature is applied before averaging so its probabilities are calibrated.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.model import ZITRAN, softmax
from zitran.data import LABELS
from zitran.predict import _RISK, _ACTION

class ZitranEnsemble:
    def __init__(self, dirs=("artifacts", "artifacts_v2", "artifacts_v3")):
        self.members = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            model_file = next((f"{d}/{f}" for f in os.listdir(d) if f.endswith("_model.npz")), None)
            if not model_file:
                continue
            tok = BPETokenizer.load(f"{d}/tokenizer.json")
            model = ZITRAN.load(model_file)
            w, T = 1.0, 1.0
            mp = f"{d}/metrics.json"
            if os.path.exists(mp):
                m = json.load(open(mp)); w = float(m.get("accuracy", 1.0)); T = float(m.get("temperature", 1.0))
            self.members.append({"name": os.path.basename(d), "tok": tok, "model": model, "weight": w, "temperature": T})
        if not self.members:
            raise FileNotFoundError("no ZITRAN artifacts found")
        total = sum(m["weight"] for m in self.members)
        for m in self.members: m["weight"] /= total

    def predict(self, text):
        avg = np.zeros(len(LABELS)); votes = {}
        for m in self.members:
            x = np.array([m["tok"].encode(text)])
            logits, bl = m["model"].forward(x)
            p = softmax(logits / m["temperature"])[0]
            avg += m["weight"] * p
            votes[m["name"]] = {"class": LABELS[int(p.argmax())], "confidence": round(float(p.max()), 4),
                                "weight": round(m["weight"], 3)}
        i = int(avg.argmax()); label = LABELS[i]
        return {"input": text, "predicted_class": label, "confidence": round(float(avg[i]), 4),
                "is_threat": bool(label != "benign"), "risk_level": _RISK[label],
                "recommended_action": _ACTION[label], "model_votes": votes,
                "top3": [{"class": LABELS[j], "prob": round(float(avg[j]), 4)} for j in avg.argsort()[::-1][:3]]}

if __name__ == "__main__":
    e = ZitranEnsemble()
    for s in sys.argv[1:] or ["ransom note pay bitcoin files encrypted locked shadow copies deleted"]:
        print(json.dumps(e.predict(s), indent=2))
