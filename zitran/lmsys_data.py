"""Teach ZITRAN from LMSYS-Chat-1M (real-world LLM conversations).

Source order:
1) HuggingFace: load_dataset("lmsys/lmsys-chat-1m") — attempted first;
   it is a gated dataset and huggingface.co is unreachable from this sandbox.
2) GitHub mirrors of real slices of that exact dataset:
   - chenwenxiaolive/lmsys-chat-1000: lmsys_10000.jsonl + lmsys_1000.jsonl
   - tanmoyAtb/lmsys-1m-chat-analysis: sample_english_5000_seed42.parquet
     (5,000 English conversations, includes OpenAI moderation annotations)
   Copied to data/raw/lmsys/.

SAFETY (real user logs are unfiltered): keep English conversations only;
drop every conversation flagged by LMSYS's own OpenAI moderation; then apply
the same harmful-intent blocklist + quality filter as ShareGPT
(zitran.sharegpt_data). Output: data/lmsys_docs.jsonl.gz ("Human:/Assistant:").

Build: python3 -m zitran.lmsys_data
"""
import gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.sharegpt_data import _clean, _safe, _englishish

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWDIR = os.path.join(ROOT, "data", "raw", "lmsys")
PROCESSED = os.path.join(ROOT, "data", "lmsys_docs.jsonl.gz")

def _pairs_from_conv(conv):
    for i in range(len(conv) - 1):
        if conv[i].get("role") == "user" and conv[i + 1].get("role") == "assistant":
            return conv  # handled by caller slicing pairs sequentially
    return conv

def iter_conversations():
    total = 0
    for name in ("lmsys_10000.jsonl", "lmsys_1000.jsonl"):
        p = os.path.join(RAWDIR, name)
        if not os.path.exists(p): continue
        for line in open(p, encoding="utf-8"):
            try:
                d = json.loads(line); total += 1
                yield d.get("conversation") or [], d.get("language", "English"), False
            except Exception:
                continue
    pq_path = os.path.join(RAWDIR, "sample_english_5000_seed42.parquet")
    if os.path.exists(pq_path):
        try:
            import pyarrow.parquet as pq
            for row in pq.read_table(pq_path).to_pylist():
                total += 1
                flagged = False
                for mod in (row.get("openai_moderation") or []):
                    if mod and mod.get("flagged"): flagged = True
                yield row.get("conversation") or [], row.get("language", "English"), flagged
        except Exception as e:
            print("[lmsys] parquet read failed:", e)
    print(f"[lmsys] conversations scanned={total:,}")

def build_processed():
    docs, kept_convs = [], 0
    for conv, lang, flagged in iter_conversations():
        if flagged or (lang and lang != "English"): continue
        pairs = []
        for i in range(len(conv) - 1):
            if conv[i].get("role") == "user" and conv[i + 1].get("role") == "assistant":
                pairs.append((_clean(conv[i].get("content", "")), _clean(conv[i + 1].get("content", ""))))
        ok = False
        for h, g in pairs[:2]:
            if not (8 <= len(h) <= 700 and 30 <= len(g) <= 1600): continue
            if not (_englishish(h) and _englishish(g)): continue
            if not _safe(h, g): continue
            docs.append(f"Human: {h}\nAssistant: {g}"); ok = True
        kept_convs += ok
    with gzip.open(PROCESSED, "wt", encoding="utf-8") as f:
        for d in docs: f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    print(f"[lmsys] kept {len(docs):,} pairs from {kept_convs:,} conversations -> {PROCESSED} ({os.path.getsize(PROCESSED)/1e6:.1f} MB)")
    return docs

def load_lmsys_docs(n=12000, seed=6):
    if not os.path.exists(PROCESSED):
        if os.path.isdir(RAWDIR): build_processed()
        else: return [], "none"
    import random
    with gzip.open(PROCESSED, "rt", encoding="utf-8") as f:
        all_docs = [json.loads(line)["text"] for line in f]
    random.Random(seed).shuffle(all_docs)
    return all_docs[:n], f"LMSYS-Chat-1M GitHub slices ({len(all_docs):,} filtered pairs)"

if __name__ == "__main__":
    build_processed()
