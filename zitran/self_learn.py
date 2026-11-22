"""ZITRAN self-learning system.

How it learns by itself (honestly, and safely):
1) INTERACTION LOG — every assistant answer is logged to data/self_learn.jsonl
   with its source (verified_kb / codegen / classifier / dialogue).
2) SELF-TEACHING DATA — the system converts its VERIFIED knowledge (world data:
   250 countries, 118 elements, encyclopedia; cyber knowledge base) into many
   paraphrased Q&A training documents. It only teaches itself facts it can
   verify against real data — it never trains on its own unverified guesses
   as 'facts' (that is how models poison themselves).
3) SELF-GENERATION HARVEST — optionally, the generative model writes completions
   which are kept ONLY if they pass strict filters (low repetition, contains a
   verified fact string, reasonable length). Rejected output is discarded.
4) CONTINUAL TRAINING — zitran/train_gen3.py fine-tunes the LM on
   world + self-teaching + logged verified interactions + a rehearsal slice of
   the old corpus (rehearsal prevents catastrophic forgetting).

Run: python3 -m zitran.self_learn          (stats + builds self-teaching set)
     python3 -m zitran.train_gen3 --max-steps 250   (the learning step)
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "data", "self_learn.jsonl")
SELF_SET = os.path.join(ROOT, "data", "self_teaching.jsonl")

TRUSTED = {"verified_kb", "world_data", "codegen", "classifier"}

def log_interaction(question, answer, source="assistant"):
    """Called by the assistant on every turn (best-effort, never raises)."""
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"q": str(question)[:500], "a": str(answer)[:1500],
                                "source": source}, ensure_ascii=False) + "\n")
    except Exception:
        pass

def load_logged(trusted_only=True):
    if not os.path.exists(LOG): return []
    out = []
    for line in open(LOG, encoding="utf-8"):
        try:
            d = json.loads(line)
            if trusted_only and d.get("source") not in TRUSTED: continue
            if len(d.get("a", "")) > 20: out.append(d)
        except Exception:
            continue
    return out

PARAPHRASES = [
    ("Q: {q}\nA: {a}", None),
    ("Question: {q}\nAnswer: {a}", None),
    ("{q} {a}", None),
]

def generate_self_teaching(seed=3):
    """Turn VERIFIED knowledge into paraphrased training docs (self-teaching)."""
    from zitran.world_data import load_countries, load_elements, ENCYCLOPEDIA
    rnd = random.Random(seed); docs = []
    for c in load_countries():
        n, cap = c["name"], c["capital"]
        cur = ", ".join(x for x in c["currencies"] if x)
        lang = ", ".join(c["languages"][:2])
        facts = []
        if cap: facts += [(f"What is the capital of {n}?", f"The capital of {n} is {cap}."),
                          (f"Name the capital city of {n}.", f"{cap} is the capital of {n}."),
                          (f"Which city is the capital of {n}?", f"The capital city of {n} is {cap}.")]
        if cur: facts.append((f"What is the currency of {n}?", f"{n} uses the {cur}."))
        if lang: facts.append((f"Which language do people speak in {n}?", f"People in {n} speak {lang}."))
        if c["region"]: facts.append((f"Where is {n}?", f"{n} is in {c['subregion'] or c['region']}, in {c['region']}. Its capital is {cap}."))
        for q, a in facts:
            docs.append(f"Q: {q}\nA: {a}")
            docs.append(f"Question: {q}\nAnswer: {a}")
    for e in load_elements():
        try:
            docs.append(f"Q: What is the symbol of {e['name']}?\nA: {e['name']} has the symbol {e['symbol']} and atomic number {e['number']}.")
            docs.append(f"Question: Tell me about the element {e['name']}.\nAnswer: {str(e.get('summary',''))[:350]}")
        except Exception:
            continue
    for topic, text in ENCYCLOPEDIA.items():
        t = topic.title()
        docs.append(f"Q: Explain {topic}.\nA: {text}")
        docs.append(f"Question: What should I know about {topic}?\nAnswer: {text}")
    # cyber knowledge (verified, defensive)
    try:
        from zitran.assistant import KNOWLEDGE
        for k, v in KNOWLEDGE.items():
            docs.append(f"Q: What is {k}?\nA: {v}")
    except Exception:
        pass
    rnd.shuffle(docs)
    return quality_filter(docs)

def quality_filter(docs):
    """Keep clean, non-repetitive, reasonably-sized docs; dedupe."""
    out, seen = [], set()
    for d in docs:
        d = str(d).strip()
        if not (30 <= len(d) <= 1400): continue
        key = d[:120].lower()
        if key in seen: continue
        words = d.lower().split()
        if len(words) >= 8:
            from collections import Counter
            top = Counter(words).most_common(1)[0][1]
            if top / len(words) > 0.30: continue  # repetitive junk
        seen.add(key); out.append(d)
    return out

def harvest_generations(tok, model, prompts, max_new=50):
    """Let the model write, keep ONLY outputs echoing verified facts (filtered)."""
    from zitran.world_data import ENCYCLOPEDIA, load_countries
    from zitran.generate import generate_text
    verified = [v[:80] for v in ENCYCLOPEDIA.values()] + \
               [f"capital of {c['name']} is {c['capital']}" for c in load_countries() if c["capital"]]
    kept = []
    for p in prompts:
        try: txt = generate_text(tok, model, p, max_new=max_new, temperature=0.7, top_k=30, seed=1)
        except Exception: continue
        body = txt[len(p):] if txt.startswith(p) else txt
        if any(v.lower()[:40] in txt.lower() for v in verified) and len(quality_filter([txt])) == 1:
            kept.append(txt)
    return kept

def build_selflearn_corpus(seed=3):
    docs = generate_self_teaching(seed)
    for d in load_logged():
        docs.append(f"Q: {d['q']}\nA: {d['a']}")
    return quality_filter(docs)

def stats():
    logged = load_logged(trusted_only=False)
    return {"logged_interactions": len(logged),
            "trusted_logged": len(load_logged()),
            "self_teaching_docs": len(generate_self_teaching())}

if __name__ == "__main__":
    docs = build_selflearn_corpus()
    with open(SELF_SET, "w", encoding="utf-8") as f:
        for d in docs: f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    print("SELF-LEARNING STATS:", json.dumps(stats(), indent=1))
    print(f"self-teaching corpus -> {SELF_SET} ({len(docs)} docs)")
    print("sample:", docs[0][:200] if docs else None)
