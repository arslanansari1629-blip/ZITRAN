"""General instruction data for ZITRAN-Gen.

First choice is exactly what was requested:
    from datasets import load_dataset
    ds = load_dataset("teknium/OpenHermes-2.5")
This sandbox cannot reach huggingface.co (only github.com / pypi.org are
allowed), so that call fails here; on an unrestricted machine it will work.
Fallback used for training in this repo: Stanford Alpaca (52k instructions,
tatsu-lab/stanford_alpaca on GitHub), loaded through the same `datasets`
library from the local JSON file.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(os.path.dirname(__file__), "..")

def try_openhermes():
    """Attempt the requested OpenHermes-2.5 load; return dataset or None."""
    try:
        from datasets import load_dataset
        return load_dataset("teknium/OpenHermes-2.5")
    except Exception as e:
        print(f"[instruction] OpenHermes via HF unavailable ({type(e).__name__}); using GitHub-hosted Alpaca fallback")
        return None

def load_instruction_docs(n=14000, seed=42):
    """Return User/Assistant documents from OpenHermes if reachable,
    else from the local Alpaca JSON (same `datasets` API)."""
    ds = try_openhermes()
    docs = []
    if ds is not None:
        split = ds["train"].shuffle(seed=seed)
        for ex in split.select(range(min(n, len(split)))):
            conv = ex.get("conversations") or []
            for a, b in zip(conv, conv[1:]):
                if a.get("from") in ("human", "user") and b.get("from") in ("gpt", "assistant"):
                    docs.append(f"User: {a['value'].strip()}\nAssistant: {b['value'].strip()}")
        return docs, "OpenHermes-2.5"
    from datasets import load_dataset
    path = os.path.join(ROOT, "data", "raw", "instruction", "alpaca_data.json")
    ds = load_dataset("json", data_files=path, split="train").shuffle(seed=seed)
    for ex in ds.select(range(min(n, len(ds)))):
        q = ex["instruction"].strip()
        if ex.get("input"): q += "\n" + ex["input"].strip()
        docs.append(f"User: {q}\nAssistant: {ex['output'].strip()}")
    return docs, "Alpaca (GitHub fallback for OpenHermes)"
