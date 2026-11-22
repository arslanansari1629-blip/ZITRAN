"""ZITRAN v5 — 50M-parameter model trained on ALL data sources:
NSL-KDD, CTU-13, UNSW-NB15 (new), Enron, JPCERT phishing URLs, malicious-URL
corpus, refined user Attack_Dataset, synthetic + paraphrase augmentation.
PyTorch CPU (float32, AdamW, cosine, label smoothing, class weights).
Step-budgeted: 2 CPU cores cannot fully converge 50M params quickly, so the
run is time-boxed and checkpointed at the best validation point; metrics
report exactly how much training was done. Reuses v4 BPE tokenizer
(texts truncated to 48 tokens for speed).
"""
import os, sys, time, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.data import LABELS, generate
from zitran.real_data import build_mixed, load_official_test, load_unsw
from zitran.train_v4 import load_attack
from zitran.augment import paraphrase
from zitran.model_torch import build_50m
from collections import Counter

def evaluate_torch(model, X, y, bs=256, limit=None):
    model.eval(); correct = bcorrect = 0; conf = np.zeros((10, 10), dtype=int)
    idx = np.arange(len(X))[:limit] if limit else np.arange(len(X))
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = torch.tensor(X[idx[i:i+bs]]); logits, bl = model(xb)
            pred = logits.argmax(1).numpy()
            correct += (pred == y[idx[i:i+bs]]).sum(); bcorrect += (bl.argmax(1).numpy() == (y[idx[i:i+bs]] != 0)).sum()
            for t, p in zip(y[idx[i:i+bs]], pred): conf[t, p] += 1
    per = {}
    for i, lab in enumerate(LABELS):
        tp = conf[i, i]; prec = tp/max(conf[:, i].sum(), 1); rec = tp/max(conf[i].sum(), 1)
        per[lab] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                    "f1": round(float(2*prec*rec/max(prec+rec, 1e-9)), 4), "support": int(conf[i].sum())}
    return {"accuracy": round(float(correct/len(idx)), 4), "binary_threat_accuracy": round(float(bcorrect/len(idx)), 4),
            "per_class": per, "confusion_matrix": conf.tolist(), "labels": LABELS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=300); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=48); ap.add_argument("--out", default="artifacts_v5")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--eval-every", type=int, default=60)
    a = ap.parse_args()
    torch.set_num_threads(2); os.makedirs(a.out, exist_ok=True); t0 = time.time(); rng_py = random.Random(a.seed)
    print("[1/6] Assembling ALL data sources...", flush=True)
    data = [(t, l, s) for t, l, s in build_mixed(cap_real=5000, seed=a.seed)]
    data += load_unsw(cap_per_class=5000, seed=a.seed)
    for t, l in generate(2200, seed=2024):
        data.append((t, l, "synthetic-text")); data.append((paraphrase(t, rng_py), l, "synthetic-augmented"))
    atk_train, atk_test = load_attack(a.seed); cnt = Counter()
    for r in atk_train:
        if cnt[r["label"]] >= 1200: continue
        cnt[r["label"]] += 1; li = LABELS.index(r["label"])
        data.append((r["text"], li, "attack-dataset")); data.append((r["event_text"], li, "attack-dataset-event"))
    print(f"      total_pool={len(data)} by_source={dict(Counter(s for *_, s in data))}", flush=True)
    with open("data/dataset_v5.jsonl", "w") as f:
        for t, l, s in data: f.write(json.dumps({"text": t, "label": LABELS[l], "source": s}) + "\n")
    texts = [t for t, _, _ in data]; y_all = np.array([l for _, l, _ in data])
    print("[2/6] Encoding with v4 tokenizer (truncate 48)...", flush=True)
    tok = BPETokenizer.load("artifacts_v4/tokenizer.json"); tok.save(f"{a.out}/tokenizer.json")
    X_all = np.array([tok.encode(t)[:a.max_len] for t in texts], dtype=np.int64)
    print(f"      X={X_all.shape} ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(a.seed); keep = []
    for c in range(len(LABELS)):
        idx = np.where(y_all == c)[0]; rng.shuffle(idx); keep += idx[:a.max_per_class].tolist()
    X, y = X_all[keep], y_all[keep]
    del texts, data, y_all, X_all
    import gc; gc.collect()
    tr, va = [], []
    for c in range(len(LABELS)):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(0.85*len(idx))
        tr += idx[:k].tolist(); va += idx[k:].tolist()
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    counts = np.bincount(ytr, minlength=10).astype(np.float64)
    class_w = torch.tensor(np.sqrt(counts.sum()/(10*np.maximum(counts, 1))), dtype=torch.float32)
    print(f"[3/6] Building 50M model (train={len(Xtr)} val={len(Xva)})...", flush=True)
    model = build_50m(vocab=len(tok), max_len=a.max_len)
    print(f"      params={model.num_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.max_steps, pct_start=0.1)
    print(f"[4/6] Training (budget {a.max_steps} steps x batch {a.batch})...", flush=True)
    history, best_acc, step = [], -1, 0; done = False
    while not done:
        perm = rng.permutation(len(Xtr)); model.train()
        for i in range(0, len(Xtr), a.batch):
            if step >= a.max_steps: done = True; break
            idx = perm[i:i+a.batch]
            xb = torch.tensor(Xtr[idx]); yb_ = torch.tensor(ytr[idx]); ybin = torch.tensor((ytr[idx] != 0).astype(np.int64))
            logits, bl = model(xb)
            loss = F.cross_entropy(logits, yb_, weight=class_w, label_smoothing=0.05) + 0.5*F.cross_entropy(bl, ybin)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 20 == 0: print(f"      step {step}/{a.max_steps} loss={float(loss):.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step % a.eval_every == 0 or step == a.max_steps:
                m = evaluate_torch(model, Xva, yva, limit=1200)
                history.append({"step": step, "loss": round(float(loss), 4), "val_acc_subset": m["accuracy"], "val_binary_subset": m["binary_threat_accuracy"]})
                print(f"      [eval] {history[-1]}", flush=True)
                if m["accuracy"] >= best_acc:
                    best_acc = m["accuracy"]; torch.save(model.state_dict(), f"{a.out}/zitran_v5_50m.pt")
                model.train()
    print("[5/6] Final evaluation (val / attack heldout subset / official subset)...", flush=True)
    model.load_state_dict(torch.load(f"{a.out}/zitran_v5_50m.pt"))
    metrics = evaluate_torch(model, Xva, yva)
    Xa = np.array([tok.encode(r["text"])[:a.max_len] for r in atk_test[::3]], dtype=np.int64)
    ya = np.array([LABELS.index(r["label"]) for r in atk_test[::3]])
    atk = evaluate_torch(model, Xa, ya)
    official = load_official_test()
    Xo = np.array([tok.encode(t)[:a.max_len] for t, _, _ in official[::4]], dtype=np.int64)
    yo = np.array([l for _, l, _ in official[::4]])
    off = evaluate_torch(model, Xo, yo)
    print(f"      val={metrics['accuracy']} attack_sub={atk['accuracy']} official_sub={off['accuracy']}", flush=True)
    print("[6/6] Saving...", flush=True)
    metrics.update({"attack_heldout_subset_accuracy": atk["accuracy"], "attack_heldout_subset_n": len(ya),
                    "official_kddtest_subset_accuracy": off["accuracy"], "official_kddtest_subset_binary": off["binary_threat_accuracy"],
                    "official_subset_n": len(yo), "attack_per_class": atk["per_class"], "official_per_class": off["per_class"],
                    "history": history, "params": model.num_params(), "training_seconds": round(time.time()-t0, 1),
                    "config": {"architecture": "d512, 15 layers, 8 heads, ff2048, MLP head, dropout 0.1",
                               "max_steps": a.max_steps, "batch": a.batch, "train_samples": len(Xtr),
                               "note": "step-budgeted CPU run; checkpoint = best val-subset point"}})
    with open(f"{a.out}/config.json", "w") as f: json.dump({"vocab": len(tok), "max_len": a.max_len, "d": 512, "heads": 8, "ff": 2048, "layers": 15, "params": model.num_params()}, f, indent=2)
    with open(f"{a.out}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps({"val": metrics["accuracy"], "attack": atk["accuracy"], "official": off["accuracy"], "params": model.num_params()}, indent=2), flush=True)

if __name__ == "__main__":
    main()
