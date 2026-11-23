"""Generate text with ZITRAN-Gen. Usage:
python3 -m zitran.generate "Question: What is phishing?\nAnswer:" --max-new 80
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.gen_model import ZitranGen

def load_gen(artifacts=None):
    if artifacts is None:
        base = os.path.join(os.path.dirname(__file__), "..")
        artifacts = next((c for c in ("artifacts_gen6", "artifacts_gen5", "artifacts_gen4", "artifacts_gen3", "artifacts_gen2", "artifacts_gen")
                          if os.path.exists(os.path.join(base, c, "config.json"))), "artifacts_gen")
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = json.load(open(os.path.join(root, artifacts, "config.json")))
    tok = BPETokenizer.load(os.path.join(root, artifacts, "tokenizer.json"))
    model = ZitranGen(vocab=cfg["vocab"], max_len=cfg["max_len"], d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], layers=cfg["layers"])
    sd = torch.load(os.path.join(root, artifacts, "zitran_gen_fp16.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()}); model.eval()
    return tok, model

def _prompt_ids(tok, prompt, limit=48):
    """CLS + prompt tokens, NO <SEP> and NO padding (padding made the model
    start a brand-new document instead of continuing the prompt)."""
    ids = [tok.token_to_id["<CLS>"]]
    for w in tok._words(prompt):
        for t in tok._encode_word(w):
            ids.append(tok.token_to_id.get(t, tok.token_to_id["<UNK>"]))
    return ids[:limit]

def generate_text(tok, model, prompt, max_new=60, temperature=0.8, top_k=40, seed=None):
    pids = _prompt_ids(tok, prompt)
    out = model.generate(pids, max_new=max_new, temperature=temperature, top_k=top_k,
                         eos_id=tok.token_to_id["<SEP>"], seed=seed)
    return tok.decode(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("prompt")
    ap.add_argument("--max-new", type=int, default=60); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--artifacts", default=None)
    a = ap.parse_args()
    tok, model = load_gen(a.artifacts)
    print(generate_text(tok, model, a.prompt, a.max_new, a.temperature, seed=a.seed))
