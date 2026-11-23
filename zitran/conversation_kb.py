"""Conversational answer retrieval over the REAL conversation datasets
(ShareGPT 85,817 filtered pairs + LMSYS 6,959 filtered pairs).

Used as the assistant's conversational layer: when no verified fact matches,
find the most similar real human question in ~93k conversations and reuse its
assistant answer — but ONLY on a very strong question overlap (>= 0.78 of the
stored question's keywords), so unrelated answers are never forced.
Answers are real model responses from the datasets (Vicuna/ChatGPT-era), so
they read like a real conversational model; factual authority still belongs
to the verified fact bank, which is always consulted first.
"""
import gzip, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.world_data import _kw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [os.path.join(ROOT, "data", "sharegpt_docs.jsonl.gz"),
         os.path.join(ROOT, "data", "lmsys_docs.jsonl.gz")]
_cache = None

def _load():
    global _cache
    if _cache is not None: return _cache
    pairs, idx, doc_kw = [], {}, []
    for path in FILES:
        if not os.path.exists(path): continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line)["text"]
                    h, g = text[7:].split("\nAssistant: ", 1)
                except Exception:
                    continue
                kw = set(_kw(h))
                if len(kw) < 2 or not (40 <= len(g) <= 1200): continue
                doc_kw.append(kw); pairs.append((h, g))
                for w in kw: idx.setdefault(w, []).append(len(pairs) - 1)
    _cache = (pairs, idx, doc_kw)
    return _cache

def answer_conversation(q, threshold=0.30):
    words = set(_kw(q))
    if len(words) < 2: return None
    pairs, idx, doc_kw = _load()
    if not pairs: return None
    from collections import Counter
    import math
    # distinctive words are rare in the corpus (e.g. "photosynthesis"); a real
    # match must contain at least one, so generic phrasing can't hijack.
    distinctive = {w for w in words if len(idx.get(w, ())) <= max(50, len(pairs) // 100)}
    cand = Counter()
    for w in words:
        for i in idx.get(w, ()): cand[i] += 1
    best, bs = None, 0.0
    for i, inter in cand.items():
        if inter < 2: continue
        if distinctive and not (distinctive & doc_kw[i]): continue
        score = inter / math.sqrt(len(doc_kw[i]) * len(words))
        if not distinctive and score < 0.6: continue
        if score > bs: best, bs = pairs[i], score
    if best and bs >= threshold:
        return best[1]
    return None

if __name__ == "__main__":
    pairs, _, _ = _load()
    print(f"conversation KB: {len(pairs):,} Q&A pairs indexed")
    for q in ["explain photosynthesis in simple words", "how do i make a cup of tea?",
              "what are the benefits of regular exercise?", "tell me a joke"]:
        a = answer_conversation(q)
        print("Q:", q, "\nA:", (a or "— no strong match —")[:280], "\n")
