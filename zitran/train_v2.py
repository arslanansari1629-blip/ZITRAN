"""ZITRAN v2 — train on REAL public data (NSL-KDD, CTU-13, Enron, JPCERT,
malicious URLs) + synthetic top-up. Upgrades over v1: larger Transformer
(d=160, 3 layers), BPE vocab 4000, seq 64, label smoothing, class-weighted
loss, LR warmup+cosine, EMA weights, early stopping, and evaluation on the
OFFICIAL NSL-KDD KDDTest+ set (contains attack types unseen in training).
"""
import os, sys, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.data import LABELS
from zitran.model import ZITRAN
from zitran.real_data import build_mixed, load_official_test
from zitran.train import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=144); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=288)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-len", type=int, default=56); ap.add_argument("--out", default="artifacts_v2")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-per-class", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    print("[1/6] Building unified real+synthetic corpus...", flush=True)
    data = build_mixed(seed=a.seed)
    from collections import Counter
    print(f"      total={len(data)} by_source={dict(Counter(s for *_, s in data))}", flush=True)
    print(f"      by_label={ {LABELS[k]: v for k, v in sorted(Counter(l for _, l, _ in data).items())} }", flush=True)
    with open("data/dataset_v2.jsonl", "w") as f:
        for t, l, s in data: f.write(json.dumps({"text": t, "label": LABELS[l], "source": s}) + "\n")
    texts = [t for t, _, _ in data]; y = np.array([l for _, l, _ in data]); yb = (y != 0).astype(np.int64)
    print("[2/6] Training BPE tokenizer (vocab 4000, seq 64)...", flush=True)
    tok = BPETokenizer(a.vocab, max_len=a.max_len); tok.train(texts); tok.save(f"{a.out}/tokenizer.json")
    X = np.array([tok.encode(t) for t in texts], dtype=np.int64)
    print(f"      vocab={len(tok)} encoded {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(a.seed)
    # cap per-class volume for CPU training time (stratified)
    keep_idx = []
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]; rng.shuffle(idx); keep_idx += idx[:a.max_per_class].tolist()
    X, y, yb = X[keep_idx], y[keep_idx], yb[keep_idx]
    print(f"      capped corpus for training: {len(y)}", flush=True)
    tr, va = [], []
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(0.85 * len(idx))
        tr += idx[:k].tolist(); va += idx[k:].tolist()
    Xtr, ytr, ybtr, Xva, yva, ybva = X[tr], y[tr], yb[tr], X[va], y[va], yb[va]
    counts = np.bincount(ytr, minlength=len(LABELS)).astype(np.float64)
    class_w = np.sqrt(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
    print(f"[3/6] Transformer v2: d={a.d_model} layers={a.layers} heads={a.heads} ff={a.ff} class_w={np.round(class_w,2)}", flush=True)
    model = ZITRAN(len(tok), d=a.d_model, heads=a.heads, ff=a.ff, layers=a.layers, max_len=a.max_len, seed=a.seed)
    model.init_ema()
    print(f"      params={model.num_params():,}", flush=True)
    print("[4/6] Training (label smoothing .05, class weights, warmup+cosine, EMA)...", flush=True)
    history, best_acc, best_state, bad = [], -1, None, 0
    steps_per_epoch = int(np.ceil(len(Xtr) / a.batch)); total_steps = steps_per_epoch * a.epochs; step = 0
    for epoch in range(a.epochs):
        perm = rng.permutation(len(Xtr)); ep_loss = ep_acc = nb = 0
        for i in range(0, len(Xtr), a.batch):
            warm = min(1.0, step / max(1, steps_per_epoch))
            lr = a.lr * warm * 0.5 * (1 + np.cos(np.pi * step / total_steps))
            idx = perm[i:i+a.batch]
            loss, acc, _ = model.loss_and_grads(Xtr[idx], ytr[idx], ybtr[idx], smooth=0.05, class_w=class_w)
            model.adam_update(lr=max(lr, 1e-5)); model.update_ema(0.999)
            ep_loss += loss; ep_acc += acc; nb += 1; step += 1
        model.apply_ema(); m = evaluate(model, Xva, yva, ybva); ema_state = {k: v.copy() for k, v in model.named_params().items()}
        model.restore_ema()
        rec = {"epoch": epoch+1, "train_loss": round(ep_loss/nb, 4), "train_acc": round(ep_acc/nb, 4),
               "val_accuracy_ema": m["accuracy"], "val_binary_ema": m["binary_threat_accuracy"]}
        history.append(rec); print(f"      {rec} ({time.time()-t0:.0f}s)", flush=True)
        if m["accuracy"] > best_acc: best_acc, best_state, bad = m["accuracy"], ema_state, 0
        else:
            bad += 1
            if bad >= a.patience: print("      early stopping", flush=True); break
    for k, v in best_state.items(): model.set_param(k, v)
    print("[5/6] Evaluating on OFFICIAL NSL-KDD KDDTest+ (novel attacks, never trained)...", flush=True)
    official = load_official_test()
    Xo = np.array([tok.encode(t) for t, _, _ in official], dtype=np.int64)
    yo = np.array([l for _, l, _ in official]); yob = (yo != 0).astype(np.int64)
    off_metrics = evaluate(model, Xo, yo, yob)
    print(f"      official KDDTest+ n={len(official)} acc={off_metrics['accuracy']} binary={off_metrics['binary_threat_accuracy']}", flush=True)
    print("[6/6] Saving artifacts_v2...", flush=True)
    metrics = evaluate(model, Xva, yva, ybva)
    metrics.update({"official_kddtest_accuracy": off_metrics["accuracy"],
                    "official_kddtest_binary": off_metrics["binary_threat_accuracy"],
                    "official_kddtest_per_class": off_metrics["per_class"],
                    "history": history, "params": model.num_params(),
                    "training_seconds": round(time.time()-t0, 1),
                    "config": {"d_model": a.d_model, "layers": a.layers, "heads": a.heads, "ff": a.ff,
                               "vocab": len(tok), "max_len": a.max_len, "epochs_run": len(history)}})
    model.save(f"{a.out}/zitran_v2_model.npz")
    with open(f"{a.out}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(f"{a.out}/official_kddtest_metrics.json", "w") as f: json.dump(off_metrics, f, indent=2)
    print(json.dumps({"val_accuracy": metrics["accuracy"], "official_kddtest_accuracy": off_metrics["accuracy"],
                      "official_binary": off_metrics["binary_threat_accuracy"], "params": model.num_params()}, indent=2), flush=True)

if __name__ == "__main__":
    main()
