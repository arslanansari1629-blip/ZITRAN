"""Train ZITRAN from scratch: data gen -> BPE train -> Transformer train -> eval -> save."""
import os, sys, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.data import generate, LABELS
from zitran.model import ZITRAN

def encode_all(tok, texts):
    return np.array([tok.encode(t) for t in texts], dtype=np.int64)

def evaluate(model, X, y, yb, bs=256):
    correct = bin_correct = 0; conf = np.zeros((len(LABELS), len(LABELS)), dtype=int); loss_sum=0.0
    for i in range(0, len(X), bs):
        xb, yb_ = X[i:i+bs], y[i:i+bs]
        logits, bl = model.forward(xb)
        pred = logits.argmax(1)
        correct += (pred == yb_).sum(); bin_correct += (bl.argmax(1) == yb[i:i+bs]).sum()
        for t, p in zip(yb_, pred): conf[t, p] += 1
    per = {}
    for i, lab in enumerate(LABELS):
        tp = conf[i, i]; prec = tp / max(conf[:, i].sum(), 1); rec = tp / max(conf[i].sum(), 1)
        per[lab] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                    "f1": round(float(2*prec*rec/max(prec+rec, 1e-9)), 4), "support": int(conf[i].sum())}
    return {"accuracy": round(float(correct/len(X)), 4), "binary_threat_accuracy": round(float(bin_correct/len(X)), 4),
            "per_class": per, "confusion_matrix": conf.tolist(), "labels": LABELS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--per-class", type=int, default=2200)
    ap.add_argument("--vocab", type=int, default=2500); ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="artifacts"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("data", exist_ok=True)
    t0 = time.time()
    print("[1/5] Generating synthetic cybersecurity dataset from scratch...")
    data = generate(a.per_class, seed=a.seed)
    texts = [t for t, _ in data]; y = np.array([l for _, l in data]); yb = (y != 0).astype(np.int64)
    with open("data/dataset.jsonl", "w") as f:
        for t, l in data: f.write(json.dumps({"text": t, "label": LABELS[l]}) + "\n")
    print(f"      {len(data)} samples | classes={len(LABELS)}")
    print("[2/5] Training BPE tokenizer from scratch...")
    tok = BPETokenizer(a.vocab, max_len=40); tok.train(texts); tok.save(f"{a.out}/tokenizer.json")
    print(f"      vocab={len(tok)} merges={len(tok.merges)}")
    X = encode_all(tok, texts)
    # stratified split
    rng = np.random.default_rng(a.seed); tr, va = [], []
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(0.85 * len(idx))
        tr += idx[:k].tolist(); va += idx[k:].tolist()
    Xtr, ytr, ybtr, Xva, yva, ybva = X[tr], y[tr], yb[tr], X[va], y[va], yb[va]
    print(f"[3/5] Building Transformer from scratch (random init, no pretrained weights)...")
    model = ZITRAN(len(tok), d=a.d_model, heads=a.heads, layers=a.layers, seed=a.seed)
    print(f"      params={model.num_params():,} d={a.d_model} layers={a.layers} heads={a.heads}")
    print("[4/5] Training (pure NumPy backprop + Adam + cosine LR)...")
    history = []
    for epoch in range(a.epochs):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * epoch / a.epochs))
        perm = rng.permutation(len(Xtr)); ep_loss = ep_acc = nb = 0
        for i in range(0, len(Xtr), a.batch):
            idx = perm[i:i+a.batch]
            loss, acc, _ = model.loss_and_grads(Xtr[idx], ytr[idx], ybtr[idx])
            model.adam_update(lr=lr)
            ep_loss += loss; ep_acc += acc; nb += 1
        m = evaluate(model, Xva, yva, ybva)
        rec = {"epoch": epoch+1, "lr": round(float(lr), 6), "train_loss": round(ep_loss/nb, 4),
               "train_acc": round(ep_acc/nb, 4), "val_accuracy": m["accuracy"], "val_binary_acc": m["binary_threat_accuracy"]}
        history.append(rec); print(f"      epoch {epoch+1:2d}/{a.epochs} loss={rec['train_loss']:.4f} train_acc={rec['train_acc']:.3f} val_acc={m['accuracy']:.3f} binary={m['binary_threat_accuracy']:.3f} ({time.time()-t0:.0f}s)")
    print("[5/5] Final evaluation + saving artifacts...")
    metrics = evaluate(model, Xva, yva, ybva); metrics["history"] = history
    metrics["params"] = model.num_params(); metrics["training_seconds"] = round(time.time()-t0, 1)
    model.save(f"{a.out}/zitran_model.npz")
    with open(f"{a.out}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ["accuracy", "binary_threat_accuracy", "params"]}, indent=2))
    print("Saved: artifacts/zitran_model.npz, tokenizer.json, metrics.json, data/dataset.jsonl")

if __name__ == "__main__":
    main()
