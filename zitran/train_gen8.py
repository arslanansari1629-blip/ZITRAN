"""ZITRAN-Gen8-LARGE: parameter scale-up — 40M params (d512, 12 layers, 8 heads, ff2048) vs Gen7 19.3M (d384/10L). Trained FROM SCRATCH on the full Gen7 corpus mix; compared fairly against Gen7 on identical held-out data.

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


FACT_PROBE = [("France","Paris"),("Japan","Tokyo"),("Brazil","Brasília"),("India","New Delhi"),
 ("Australia","Canberra"),("Canada","Ottawa"),("Germany","Berlin"),("Italy","Rome"),
 ("Spain","Madrid"),("Egypt","Cairo"),("China","Beijing"),("Russia","Moscow"),
 ("Uttar Pradesh","Lucknow"),("Maharashtra","Mumbai"),("California","Sacramento"),("Texas","Austin")]

def fact_probe(tok, model):
    from zitran.generate import generate_text
    hits, detail = 0, {}
    for q, want in FACT_PROBE:
        out = generate_text(tok, model, f"Q: What is the capital of {q}?\nA:", max_new=14, temperature=0.15, top_k=10, seed=4)
        ok = want.lower() in out.lower(); hits += ok; detail[q] = {"want": want, "ok": ok, "got": out.replace("\n", " ")[:110]}
    return hits, len(FACT_PROBE), detail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1000); ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=80); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="artifacts_gen8"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.set_num_threads(2); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    print("[1/6] Building world + self-teaching + rehearsal corpora...", flush=True)
    world = build_world_corpus()
    test_docs, world_all = world[-250:], world[:-250]
    random.Random(21).shuffle(world_all); world_train = world_all[:20000]
    self_docs = build_selflearn_corpus()
    rehearsal = build_corpus(); random.Random(9).shuffle(rehearsal); rehearsal = rehearsal[:3000]
    from zitran.instruction_data import load_instruction_docs
    idocs, isrc = load_instruction_docs(2000); rehearsal += idocs
    from zitran.sharegpt_data import load_sharegpt_docs
    sg_docs, sg_src = load_sharegpt_docs(14000)
    print(f"      sharegpt={len(sg_docs)} (source: {sg_src})", flush=True)
    from zitran.lmsys_data import load_lmsys_docs
    lm_docs, lm_src = load_lmsys_docs(12000)
    print(f"      lmsys={len(lm_docs)} (source: {lm_src})", flush=True)
    docs = world_train + self_docs + rehearsal + sg_docs + lm_docs
    random.Random(13).shuffle(docs)
    print(f"      world={len(world_train)} self_teaching={len(self_docs)} rehearsal={len(rehearsal)} total_docs={len(docs)} (instruction: {isrc})", flush=True)
    print("[2/6] Tokenizing (Gen2 BPE vocab)...", flush=True)
    tok = BPETokenizer.load("artifacts_gen2/tokenizer.json")
    ids = tokenize_corpus(tok, docs); test_ids = tokenize_corpus(tok, test_docs)
    train_ids = np.array(ids, dtype=np.int64)
    print(f"      tokens={len(ids)/1e6:.2f}M ({time.time()-t0:.0f}s)", flush=True)
    cfg = {"vocab": len(tok), "max_len": a.seq, "d": 512, "heads": 8, "ff": 2048, "layers": 12}
    model = ZitranGen(vocab=cfg["vocab"], max_len=cfg["max_len"], d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], layers=cfg["layers"])
    print(f"[3/6] Gen8-LARGE from scratch: {model.num_params():,} params (d=512 layers=12 heads=8 ff=2048). Measuring BEFORE (random init)...", flush=True)
    before = ppl(model, test_ids, a.seq)
    print(f"      world-knowledge perplexity BEFORE training (random): {before:.2f}", flush=True)
    # fair comparison: Gen7 on the SAME held-out test stream
    try:
        cfg7 = json.load(open("artifacts_gen7/config.json"))
        m7 = ZitranGen(vocab=cfg7["vocab"], max_len=cfg7["max_len"], d=cfg7["d"], heads=cfg7["heads"], ff=cfg7["ff"], layers=cfg7["layers"])
        sd7 = torch.load("artifacts_gen7/zitran_gen_fp16.pt", map_location="cpu")
        m7.load_state_dict({k: v.float() for k, v in sd7.items()})
        gen7_ppl = ppl(m7, test_ids, a.seq); del m7, sd7
        print(f"      Gen7 (19.3M) perplexity on SAME test data: {gen7_ppl:.2f}", flush=True)
    except Exception as e:
        gen7_ppl = None; print("      Gen7 comparison unavailable:", e, flush=True)
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
    ph, pn, pdetail = fact_probe(tok, model)
    print(f"      FACT PROBE after training: {ph}/{pn} capitals generated correctly", flush=True)
    tok.save(f"{a.out}/tokenizer.json")
    torch.save({k: v.half() for k, v in model.state_dict().items()}, f"{a.out}/zitran_gen_fp16.pt")
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=1)
    from zitran.generate import generate_text
    prompts = ["Q: What is the capital of France?\nA:", "Q: What is the capital of Japan?\nA:",
               "Q: What is photosynthesis?\nA:", "Q: What is phishing?\nA:"]
    samples = {p: generate_text(tok, model, p, max_new=40, temperature=0.4, top_k=20, seed=2) for p in prompts}
    metrics = {"fact_probe": {"hits": ph, "total": pn, "detail": pdetail},
               "base": "FROM SCRATCH (Gen8-Large parameter scale-up)", "gen7_ppl_same_test": (round(gen7_ppl,2) if gen7_ppl else None), "architecture": cfg, "params": model.num_params(),
               "steps": a.max_steps, "docs": len(docs), "tokens": len(ids),
               "world_perplexity_before_random": round(before, 2), "world_perplexity_after": round(after, 2),
               "sources": ["mledoze/countries (250 countries)", "Bowserinator periodic table (118 elements)",
                           "curated encyclopedia (62 topics)", "self-teaching paraphrases of verified facts",
                           "logged trusted interactions", "rehearsal: Gen2 cyber+instruction slice",
                           "OpenTriviaQA (49,716 Q&A, 22 categories)", "India (36) + US (50) state capitals", "planets + continents tables", "Jeopardy (214,236 Q&A) -> unified 263,948-fact QA bank", "ShareGPT_Vicuna_unfiltered (94,145 conversations GitHub mirror, 85,817 safety-filtered pairs)", "LMSYS-Chat-1M (gated HF; GitHub slices 7,116 conversations, moderation+blocklist filtered to 6,959 pairs)"],
               "samples_after": samples, "seconds": round(time.time() - t0)}
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=1)
    print("[6/6] Saved", a.out, json.dumps({k: metrics.get(k) for k in ("world_perplexity_before_random", "world_perplexity_after", "steps", "docs")}))
    for p, s_ in samples.items(): print("PROMPT:", p.replace("\n", " "), "\n  ->", s_.replace("\n", " ")[:220])

if __name__ == "__main__":
    main()
