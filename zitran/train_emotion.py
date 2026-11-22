"""Train ZITRAN-Emotion: from-scratch NumPy Transformer (same engine as v1)
on GoEmotions (Ekman: anger/disgust/fear/joy/neutral/sadness/surprise).
Official test.tsv is held out untouched for the final score.
"""
import os, sys, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.model import ZITRAN
from zitran.emotion_data import load_goemotions, EMOTION_LABELS
from collections import Counter

def evaluate_emotion(model, X, y, bs=256):
    correct = 0; conf = np.zeros((7, 7), dtype=int)
    for i in range(0, len(X), bs):
        pred, _, _ = model.predict(X[i:i+bs])
        correct += (pred == y[i:i+bs]).sum()
        for t, p in zip(y[i:i+bs], pred): conf[t, p] += 1
    per = {}
    for i, lab in enumerate(EMOTION_LABELS):
        tp = conf[i, i]; prec = tp/max(conf[:, i].sum(), 1); rec = tp/max(conf[i].sum(), 1)
        per[lab] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                    "f1": round(float(2*prec*rec/max(prec+rec, 1e-9)), 4), "support": int(conf[i].sum())}
    macro_f1 = float(np.mean([v["f1"] for v in per.values()]))
    return {"accuracy": round(float(correct/len(X)), 4), "macro_f1": round(macro_f1, 4), "per_class": per, "confusion_matrix": conf.tolist(), "labels": EMOTION_LABELS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--cap", type=int, default=7000)
    ap.add_argument("--vocab", type=int, default=2000); ap.add_argument("--out", default="artifacts_emotion")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    print("[1/4] Loading GoEmotions...", flush=True)
    train = load_goemotions("train", cap_per_class=a.cap)
    test = load_goemotions("test")
    print(f"      train={len(train)} {dict(Counter(l for _, l in train))} test={len(test)}", flush=True)
    texts = [t for t, _ in train]; y = np.array([l for _, l in train]); yb = (y != 4).astype(np.int64)
    print("[2/4] BPE tokenizer...", flush=True)
    tok = BPETokenizer(a.vocab, max_len=32); tok.train(texts); tok.save(f"{a.out}/tokenizer.json")
    X = np.array([tok.encode(t) for t in texts], dtype=np.int64)
    Xte = np.array([tok.encode(t) for t, _ in test], dtype=np.int64); yte = np.array([l for _, l in test])
    rng = np.random.default_rng(42); tr, va = [], []
    for c in range(7):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(0.9*len(idx))
        tr += idx[:k].tolist(); va += idx[k:].tolist()
    Xtr, ytr, ybtr, Xva, yva = X[tr], y[tr], yb[tr], X[va], y[va]
    print(f"[3/4] Training emotion Transformer (train={len(Xtr)})...", flush=True)
    model = ZITRAN(len(tok), n_classes=7, d=96, heads=4, ff=192, layers=2, max_len=32, seed=7)
    print(f"      params={model.num_params():,}", flush=True)
    history = []
    for epoch in range(a.epochs):
        perm = rng.permutation(len(Xtr)); el = ea = nb = 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]
            loss, acc, _ = model.loss_and_grads(Xtr[idx], ytr[idx], ybtr[idx], smooth=0.05)
            model.adam_update(lr=3e-3 * 0.5 * (1 + np.cos(np.pi * epoch / a.epochs)))
            el += loss; ea += acc; nb += 1
        m = evaluate_emotion(model, Xva, yva)
        history.append({"epoch": epoch+1, "train_loss": round(el/nb, 4), "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"]})
        print(f"      {history[-1]} ({time.time()-t0:.0f}s)", flush=True)
    print("[4/4] Official GoEmotions test + save...", flush=True)
    metrics = evaluate_emotion(model, Xte, yte)
    metrics.update({"split": "GoEmotions ekman test.tsv (held out)", "history": history,
                    "params": model.num_params(), "training_seconds": round(time.time()-t0, 1)})
    model.save(f"{a.out}/emotion_model.npz")
    with open(f"{a.out}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps({"test_accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "params": model.num_params()}, indent=2), flush=True)

if __name__ == "__main__":
    main()
