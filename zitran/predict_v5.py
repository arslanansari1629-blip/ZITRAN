"""Inference for ZITRAN v5 (50M, PyTorch). Same output schema as v1-v4."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.model_torch import ZitranV5
from zitran.data import LABELS
from zitran.predict import _RISK, _ACTION

class ZitranV5Detector:
    def __init__(self, artifacts="artifacts_v5"):
        cfg = json.load(open(f"{artifacts}/config.json"))
        self.tok = BPETokenizer.load(f"{artifacts}/tokenizer.json")
        self.max_len = cfg["max_len"]
        self.model = ZitranV5(vocab=cfg["vocab"], max_len=cfg["max_len"], d=cfg["d"],
                              heads=cfg["heads"], ff=cfg["ff"], layers=cfg["layers"])
        path = f"{artifacts}/zitran_v5_50m.pt"
        if not os.path.exists(path): path = f"{artifacts}/zitran_v5_50m_fp16.pt"
        sd = torch.load(path, map_location="cpu")
        self.model.load_state_dict({k: v.float() for k, v in sd.items()})
        self.model.eval()
        self.params = cfg["params"]

    def predict(self, text):
        ids = self.tok.encode(text)[:self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))
        with torch.no_grad():
            logits, bl = self.model(torch.tensor([ids]))
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            bprobs = torch.softmax(bl, dim=-1)[0].numpy()
        i = int(probs.argmax()); label = LABELS[i]
        return {"input": text, "predicted_class": label, "confidence": round(float(probs[i]), 4),
                "is_threat": bool(label != "benign"), "threat_probability": round(float(bprobs[1]), 4),
                "risk_level": _RISK[label], "recommended_action": _ACTION[label],
                "model": f"ZITRAN v5 ({self.params:,} params)",
                "top3": [{"class": LABELS[j], "prob": round(float(probs[j]), 4)} for j in probs.argsort()[::-1][:3]]}

if __name__ == "__main__":
    d = ZitranV5Detector(sys.argv[1] if len(sys.argv) > 2 else "artifacts_v5")
    for s in sys.argv[2:] or ["ransom note pay bitcoin files encrypted locked shadow copies deleted",
                              "url https://secure-login-verify.top/account/update"]:
        print(json.dumps(d.predict(s), indent=2))
