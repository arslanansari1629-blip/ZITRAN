"""Teach ZITRAN from ShareGPT_Vicuna_unfiltered (90k real conversations).

Source order (same pattern as instruction_data.py):
1) HuggingFace: load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")
   — attempted first; huggingface.co is unreachable from this sandbox.
2) GitHub mirror of the very file that dataset is built from:
   Marugannwg/ShareGPT92k_user_linguistic_mining ->
   raw_data/ShareGPT_V3_unfiltered_cleaned_split.7z (105 MB) -> extracted to
   data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json (673 MB).

SAFETY: the dataset is *unfiltered* real-world chat logs. ZITRAN is a
defensive assistant, so conversations are kept ONLY if they pass a strict
filter: English-ish, reasonable length, at least one human->gpt exchange,
and NO harmful-intent content (weapons, malware creation, exploitation
instructions, sexual content involving minors, etc. — see BLOCK list).
What survives is general conversation/instruction data.

Build the processed training file once:
    python3 -m zitran.sharegpt_data          (streams the 673 MB JSON)
Then training reads data/sharegpt_docs.jsonl.gz.
"""
import gzip, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "sharegpt", "ShareGPT_V3_unfiltered_cleaned_split.json")
PROCESSED = os.path.join(ROOT, "data", "sharegpt_docs.jsonl.gz")

BLOCK = ["make a bomb", "build a bomb", "create malware", "write malware", "make malware",
         "write a virus", "create a virus", "ransomware", "keylogger", "stealer",
         "how to hack into", "hack into", "exploit code", "shellcode", "child porn",
         "csam", "underage", "molest", "suicide method", "kill myself", "synthesize meth",
         "make meth", "cocaine recipe", "credit card dump", "carding", "phishing kit",
         "create a phishing", "ddos attack script", "botnet", "sql injection payload",
         "bypass drm crack", "serial key generator", "keygen"]

def _clean(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t

def _safe(human, gpt):
    low = (human + " " + gpt).lower()
    return not any(b in low for b in BLOCK)

def _englishish(t):
    if not t: return False
    asc = sum(1 for c in t if ord(c) < 128)
    return asc / max(1, len(t)) > 0.92

def iter_filtered_conversations(path=RAW):
    """Stream the huge JSON; yield training docs (human->gpt pairs)."""
    import ijson
    kept = total = 0
    with open(path, "rb") as f:
        for conv in ijson.items(f, "item"):
            total += 1
            turns = conv.get("conversations") or []
            pairs = []
            for i in range(len(turns) - 1):
                if turns[i].get("from") == "human" and turns[i + 1].get("from") == "gpt":
                    pairs.append((_clean(turns[i].get("value", "")), _clean(turns[i + 1].get("value", ""))))
            for h, g in pairs[:2]:  # at most 2 exchanges per conversation
                if not (8 <= len(h) <= 700 and 30 <= len(g) <= 1600): continue
                if not (_englishish(h) and _englishish(g)): continue
                if not _safe(h, g): continue
                kept += 1
                yield f"Human: {h}\nAssistant: {g}"
    print(f"[sharegpt] conversations scanned={total:,} kept_pairs={kept:,} (safety+quality filtered)")

def build_processed(limit=None):
    docs = []
    for d in iter_filtered_conversations():
        docs.append(d)
        if limit and len(docs) >= limit: break
    os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
    with gzip.open(PROCESSED, "wt", encoding="utf-8") as f:
        for d in docs: f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    print(f"[sharegpt] wrote {len(docs):,} docs -> {PROCESSED} ({os.path.getsize(PROCESSED)/1e6:.1f} MB)")
    return docs

def try_huggingface(n=2000):
    try:
        from datasets import load_dataset
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        docs = []
        for ex in ds.select(range(min(n, len(ds)))):
            turns = ex.get("conversations") or []
            for i in range(len(turns) - 1):
                if turns[i].get("from") == "human" and turns[i + 1].get("from") == "gpt":
                    h, g = _clean(turns[i].get("value", "")), _clean(turns[i + 1].get("value", ""))
                    if 8 <= len(h) <= 700 and 30 <= len(g) <= 1600 and _safe(h, g):
                        docs.append(f"Human: {h}\nAssistant: {g}")
                    break
        return docs, "huggingface: anon8231489123/ShareGPT_Vicuna_unfiltered"
    except Exception as e:
        print(f"[sharegpt] HF unavailable ({type(e).__name__}); using GitHub-mirror file")
        return None, None

def load_sharegpt_docs(n=12000, seed=4):
    """Return (docs, source). Prefers HF, falls back to the processed mirror."""
    docs, src = try_huggingface(n)
    if docs: return docs[:n], src
    if not os.path.exists(PROCESSED) and os.path.exists(RAW):
        build_processed()
    out = []
    if os.path.exists(PROCESSED):
        import random
        with gzip.open(PROCESSED, "rt", encoding="utf-8") as f:
            all_docs = [json.loads(line)["text"] for line in f]
        random.Random(seed).shuffle(all_docs)
        out = all_docs[:n]
        return out, f"GitHub mirror of ShareGPT_V3_unfiltered_cleaned_split ({len(all_docs):,} filtered pairs)"
    return [], "none"

if __name__ == "__main__":
    if os.path.exists(RAW):
        build_processed()
    else:
        docs, src = load_sharegpt_docs(10)
        print(src, len(docs))
