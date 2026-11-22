"""Train ZITRAN-Gen (decoder-only LM) on the everywhere-corpus.
Step-budgeted CPU training; reports held-out perplexity and generates
sample completions at the end (verbatim, unedited, in metrics.json).
"""
import os, sys, time, json, argparse, math, random
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.gen_data import build_corpus, tokenize_corpus
from zitran.gen_model import ZitranGen

def ppl(model, ids, seq, bs=32):
    model.eval(); tot = cnt = 0
    chunks = [ids[i:i+seq] for i in range(0, len(ids) - seq, seq)]
    with torch.no_grad():
        for i in range(0, len(chunks), bs):
            x = torch.tensor(chunks[i:i+bs])
            loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, model.vocab), x[:, 1:].reshape(-1), reduction="sum")
            tot += float(loss); cnt += x[:, 1:].numel()
    return math.exp(tot / cnt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=600); ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=96); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=384); ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--out", default="artifacts_gen"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.set_num_threads(2); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    print("[1/5] Building corpus from all sources (defensive framing)...", flush=True)
    docs = build_corpus()
    print(f"      docs={len(docs)}", flush=True)
    print("[2/5] Tokenizing with v4 BPE vocab...", flush=True)
    tok = BPETokenizer.load("artifacts_v4/tokenizer.json")
    ids = tokenize_corpus(tok, docs)
    tok.save(f"{a.out}/tokenizer.json")
    np.save(f"{a.out}/corpus_ids.npy", np.array(ids, dtype=np.int32))
    n_test = int(0.05 * len(ids)); test_ids, train_ids = ids[-n_test:], ids[:-n_test]
    print(f"      tokens={len(ids)/1e6:.2f}M train={len(train_ids)/1e6:.2f}M test={len(test_ids)/1e6:.2f}M ({time.time()-t0:.0f}s)", flush=True)
    train_ids = np.array(train_ids, dtype=np.int64)
    print("[3/5] Building decoder-only Transformer...", flush=True)
    model = ZitranGen(vocab=len(tok), max_len=a.seq, d=a.d, layers=a.layers)
    print(f"      params={model.num_params():,} d={a.d} layers={a.layers} seq={a.seq}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.max_steps, pct_start=0.08)
    print(f"[4/5] Training ({a.max_steps} steps x batch {a.batch} x seq {a.seq})...", flush=True)
    rng = np.random.default_rng(a.seed); history = []; best = 1e18; step = 0; model.train()
    while step < a.max_steps:
        starts = rng.integers(0, len(train_ids) - a.seq - 1, size=a.batch)
        x = torch.tensor(np.stack([train_ids[s:s+a.seq] for s in starts]))
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, model.vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 25 == 0: print(f"      step {step}/{a.max_steps} loss={float(loss):.3f} train_ppl~{math.exp(min(float(loss),10)):.1f} ({time.time()-t0:.0f}s)", flush=True)
        if step % 150 == 0 or step == a.max_steps:
            cur = ppl(model, test_ids[:20000], a.seq)
            history.append({"step": step, "loss": round(float(loss), 4), "test_ppl": round(cur, 2)})
            print(f"      [eval] {history[-1]}", flush=True)
            if cur < best: best = cur; torch.save({k: v.half() for k, v in model.state_dict().items()}, f"{a.out}/zitran_gen_fp16.pt")
            model.train()
    print("[5/5] Sample generations (verbatim) + save...", flush=True)
    model.load_state_dict({k: v.float() for k, v in torch.load(f"{a.out}/zitran_gen_fp16.pt").items()})
    sep = tok.token_to_id["<SEP>"]
    prompts = ["Question: What is ransomware?\nAnswer:",
               "Incident: mass file encryption files renamed locked ransom note bitcoin\nAnalysis:",
               "User: I feel sad today.\nAssistant:",
               "User: How are you?\nAssistant:",
               "Question: How do you detect and respond to phishing?\nAnswer:"]
    samples = []
    for p in prompts:
        tok.max_len = 100000; pids = tok.encode(p)[:40]
        out = model.generate(pids, max_new=45, temperature=0.8, top_k=40, eos_id=sep, seed=3)
        text = tok.decode(out)
        samples.append({"prompt": p, "generated": text})
        print(f"--- PROMPT: {p}\n    GEN: {text[len(tok.decode(pids)):][:300]}", flush=True)
    cfg = {"vocab": len(tok), "max_len": a.seq, "d": a.d, "heads": 6, "ff": a.d * 4, "layers": a.layers, "params": model.num_params()}
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)
    metrics = {"params": model.num_params(), "tokens_trained_on": int(a.max_steps * a.batch * a.seq),
               "corpus_tokens": len(ids), "best_test_perplexity": round(best, 2), "history": history,
               "samples_verbatim": samples, "training_seconds": round(time.time() - t0, 1),
               "note": "step-budgeted CPU run; small specialist LM, not a frontier model"}
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=2)
    print(json.dumps({"params": cfg["params"], "best_test_ppl": metrics["best_test_perplexity"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
