"""ZITRAN-Gen3: continual / self-learning fine-tune.

Starts from the Gen2 checkpoint (never from scratch) and learns from:
  world corpus (250 countries + 118 elements + encyclopedia, real datasets),
  self-teaching corpus (verified facts paraphrased by self_learn.py + logged
  trusted interactions), and a rehearsal slice of the Gen2 corpus so it does
  not forget cybersecurity/instruction skills.

Reports world-knowledge perplexity BEFORE vs AFTER so learning is measured,
not claimed. Usage: python3 -m zitran.train_gen3 --max-steps 250
"""
import os, sys, time, json, argparse, math, random
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.gen_data import tokenize_corpus, build_corpus
from zitran.gen_model import ZitranGen
from zitran.world_data import build_world_corpus
from zitran.self_learn import build_selflearn_corpus

def ppl(model, ids, seq, bs=32):
    if len(ids) <= seq + 1: return float("nan")
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
    ap.add_argument("--max-steps", type=int, default=250); ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=80); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="artifacts_gen3"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default="artifacts_gen2", help="checkpoint to start from (continual learning)")
    a = ap.parse_args()
    torch.set_num_threads(2); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    print("[1/6] Building world + self-teaching + rehearsal corpora...", flush=True)
    world = build_world_corpus()
    test_docs, world_train = world[-200:], world[:-200]
    self_docs = build_selflearn_corpus()
    rehearsal = build_corpus(); random.Random(9).shuffle(rehearsal); rehearsal = rehearsal[:3000]
    from zitran.instruction_data import load_instruction_docs
    idocs, isrc = load_instruction_docs(2000); rehearsal += idocs
    docs = world_train * 2 + self_docs * 2 + rehearsal
    random.Random(13).shuffle(docs)
    print(f"      world={len(world_train)} self_teaching={len(self_docs)} rehearsal={len(rehearsal)} total_docs={len(docs)} (instruction: {isrc})", flush=True)
    print("[2/6] Tokenizing (Gen2 BPE vocab)...", flush=True)
    tok = BPETokenizer.load("artifacts_gen2/tokenizer.json")
    ids = tokenize_corpus(tok, docs); test_ids = tokenize_corpus(tok, test_docs)
    train_ids = np.array(ids, dtype=np.int64)
    print(f"      tokens={len(ids)/1e6:.2f}M ({time.time()-t0:.0f}s)", flush=True)
    cfg = json.load(open(f"{a.base}/config.json"))
    model = ZitranGen(vocab=cfg["vocab"], max_len=cfg["max_len"], d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], layers=cfg["layers"])
    sd = torch.load(f"{a.base}/zitran_gen_fp16.pt", map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    print(f"[3/6] Loaded {a.base} checkpoint ({model.num_params():,} params). Measuring BEFORE...", flush=True)
    before = ppl(model, test_ids, a.seq)
    print(f"      world-knowledge perplexity BEFORE self-learning: {before:.2f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.max_steps, pct_start=0.08)
    print(f"[4/6] Self-learning fine-tune ({a.max_steps} steps x {a.batch} x {a.seq})...", flush=True)
    rng = np.random.default_rng(a.seed); step = 0; model.train(); losses = []
    while step < a.max_steps:
        starts = rng.integers(0, len(train_ids) - a.seq - 1, size=a.batch)
        x = torch.tensor(np.stack([train_ids[s:s+a.seq] for s in starts]))
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, model.vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1; losses.append(float(loss))
        if step % 50 == 0: print(f"      step {step} loss {np.mean(losses[-50:]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("[5/6] Measuring AFTER + generating samples...", flush=True)
    after = ppl(model, test_ids, a.seq)
    print(f"      world-knowledge perplexity AFTER self-learning: {after:.2f}", flush=True)
    tok.save(f"{a.out}/tokenizer.json")
    torch.save({k: v.half() for k, v in model.state_dict().items()}, f"{a.out}/zitran_gen_fp16.pt")
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)
    from zitran.generate import generate_text
    prompts = ["Q: What is the capital of France?\nA:", "Q: What is the capital of Japan?\nA:",
               "Q: What is photosynthesis?\nA:", "Q: What is phishing?\nA:"]
    samples = {p: generate_text(tok, model, p, max_new=40, temperature=0.4, top_k=20, seed=2) for p in prompts}
    metrics = {"base": f"{a.base} (fine-tuned, not from scratch)", "params": model.num_params(),
               "steps": a.max_steps, "docs": len(docs), "tokens": len(ids),
               "world_perplexity_before": round(before, 2), "world_perplexity_after": round(after, 2),
               "sources": ["mledoze/countries (250 countries)", "Bowserinator periodic table (118 elements)",
                           "curated encyclopedia (62 topics)", "self-teaching paraphrases of verified facts",
                           "logged trusted interactions", "rehearsal: Gen2 cyber+instruction slice"],
               "samples_after": samples, "seconds": round(time.time() - t0)}
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=1)
    print("[6/6] Saved", a.out, json.dumps({k: metrics[k] for k in ("world_perplexity_before", "world_perplexity_after", "steps", "docs")}))
    for p, s_ in samples.items(): print("PROMPT:", p.replace("\n", " "), "\n  ->", s_.replace("\n", " ")[:220])

if __name__ == "__main__":
    main()
