"""ZITRAN v3 (ultra): real telemetry (v2 corpus) + paraphrase-augmented
free-text at full volume, MLP head, d=160/3 layers, BPE 4000/seq 56,
label smoothing, class weights, warmup+cosine, EMA(0.995), early stopping,
temperature calibration, official KDDTest+ evaluation.
"""
import os, sys, time, json, argparse, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.data import LABELS, generate
from zitran.model import ZITRAN, softmax
from zitran.real_data import build_mixed, load_official_test
from zitran.augment import paraphrase
from zitran.calibrate import collect_logits, fit_temperature, expected_calibration_error
from zitran.train import evaluate
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=160); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-len", type=int, default=56); ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-per-class", type=int, default=2600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time(); rng_py = random.Random(a.seed)
    print("[1/7] Corpus: real telemetry + augmented free-text...", flush=True)
    data = [(t, l, s) for t, l, s in build_mixed(seed=a.seed)]
    synth = generate(2200, seed=2024)
    for t, l in synth:
        data.append((t, l, "synthetic-text"))
        data.append((paraphrase(t, rng_py), l, "synthetic-augmented"))
    print(f"      total={len(data)} by_source={dict(Counter(s for *_, s in data))}", flush=True)
    print(f"      by_label={ {LABELS[k]: v for k, v in sorted(Counter(l for _, l, _ in data).items())} }", flush=True)
    with open("data/dataset_v3.jsonl", "w") as f:
        for t, l, s in data: f.write(json.dumps({"text": t, "label": LABELS[l], "source": s}) + "\n")
    texts = [t for t, _, _ in data]; y = np.array([l for _, l, _ in data]); yb = (y != 0).astype(np.int64)
    print("[2/7] BPE tokenizer (24k subsample — tokenizer needs coverage, not volume)...", flush=True)
    tok_texts = [texts[i] for i in np.random.default_rng(1).choice(len(texts), size=min(24000, len(texts)), replace=False)]
    tok = BPETokenizer(a.vocab, max_len=a.max_len); tok.train(tok_texts); tok.save(f"{a.out}/tokenizer.json")
    X = np.array([tok.encode(t) for t in texts], dtype=np.int64)
    print(f"      vocab={len(tok)} X={X.shape} ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(a.seed); keep = []
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]; rng.shuffle(idx); keep += idx[:a.max_per_class].tolist()
    X, y, yb = X[keep], y[keep], yb[keep]
    tr, va = [], []
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(0.85 * len(idx))
        tr += idx[:k].tolist(); va += idx[k:].tolist()
    Xtr, ytr, ybtr, Xva, yva, ybva = X[tr], y[tr], yb[tr], X[va], y[va], yb[va]
    counts = np.bincount(ytr, minlength=len(LABELS)).astype(np.float64)
    class_w = np.sqrt(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
    print(f"[3/7] Transformer v3 MLP-head d={a.d_model} L={a.layers} ff={a.ff} train={len(Xtr)} class_w={np.round(class_w,2)}", flush=True)
    model = ZITRAN(len(tok), d=a.d_model, heads=a.heads, ff=a.ff, layers=a.layers, max_len=a.max_len, seed=a.seed, mlp_head=True)
    model.init_ema(); print(f"      params={model.num_params():,}", flush=True)
    print("[4/7] Training...", flush=True)
    history, best_acc, best_state, bad = [], -1, None, 0
    spe = int(np.ceil(len(Xtr)/a.batch)); total_steps = spe*a.epochs; step = 0
    for epoch in range(a.epochs):
        perm = rng.permutation(len(Xtr)); el = ea = nb = 0
        for i in range(0, len(Xtr), a.batch):
            warm = min(1.0, step/max(1, spe)); lr = a.lr*warm*0.5*(1+np.cos(np.pi*step/total_steps))
            idx = perm[i:i+a.batch]
            loss, acc, _ = model.loss_and_grads(Xtr[idx], ytr[idx], ybtr[idx], smooth=0.05, class_w=class_w)
            model.adam_update(lr=max(lr, 1e-5)); model.update_ema(0.995); el += loss; ea += acc; nb += 1; step += 1
        model.apply_ema(); m = evaluate(model, Xva, yva, ybva)
        state = {k: v.copy() for k, v in model.named_params().items()}; model.restore_ema()
        rec = {"epoch": epoch+1, "train_loss": round(el/nb, 4), "train_acc": round(ea/nb, 4),
               "val_accuracy_ema": m["accuracy"], "val_binary_ema": m["binary_threat_accuracy"]}
        history.append(rec); print(f"      {rec} ({time.time()-t0:.0f}s)", flush=True)
        if m["accuracy"] > best_acc: best_acc, best_state, bad = m["accuracy"], state, 0
        else:
            bad += 1
            if bad >= a.patience: print("      early stopping", flush=True); break
    for k, v in best_state.items(): model.set_param(k, v)
    print("[5/7] Temperature calibration on validation...", flush=True)
    logits_va = collect_logits(model, Xva)
    ece_before = expected_calibration_error(softmax(logits_va), yva)
    T, nll = fit_temperature(logits_va, yva)
    ece_after = expected_calibration_error(softmax(logits_va/T), yva)
    print(f"      T={T:.2f} ECE {ece_before:.4f} -> {ece_after:.4f}", flush=True)
    print("[6/7] Official KDDTest+ (novel attacks)...", flush=True)
    official = load_official_test()
    Xo = np.array([tok.encode(t) for t, _, _ in official], dtype=np.int64)
    yo = np.array([l for _, l, _ in official]); yob = (yo != 0).astype(np.int64)
    off = evaluate(model, Xo, yo, yob)
    print(f"      n={len(official)} acc={off['accuracy']} binary={off['binary_threat_accuracy']}", flush=True)
    print("[7/7] Saving artifacts_v3...", flush=True)
    metrics = evaluate(model, Xva, yva, ybva)
    metrics.update({"official_kddtest_accuracy": off["accuracy"], "official_kddtest_binary": off["binary_threat_accuracy"],
                    "official_kddtest_per_class": off["per_class"], "temperature": T,
                    "ece_before_calibration": round(ece_before, 4), "ece_after_calibration": round(ece_after, 4),
                    "history": history, "params": model.num_params(), "training_seconds": round(time.time()-t0, 1),
                    "config": {"d_model": a.d_model, "layers": a.layers, "heads": a.heads, "ff": a.ff,
                               "vocab": len(tok), "max_len": a.max_len, "mlp_head": True, "epochs_run": len(history)}})
    model.save(f"{a.out}/zitran_v3_model.npz")
    with open(f"{a.out}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(f"{a.out}/official_kddtest_metrics.json", "w") as f: json.dump(off, f, indent=2)
    print(json.dumps({"val_accuracy": metrics["accuracy"], "official": off["accuracy"], "temperature": T, "params": model.num_params()}, indent=2), flush=True)

if __name__ == "__main__":
    main()
