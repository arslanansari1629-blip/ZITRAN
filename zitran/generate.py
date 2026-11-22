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
        artifacts = "artifacts_gen2" if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "artifacts_gen2", "config.json")) else "artifacts_gen"
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = json.load(open(os.path.join(root, artifacts, "config.json")))
    tok = BPETokenizer.load(os.path.join(root, artifacts, "tokenizer.json"))
    model = ZitranGen(vocab=cfg["vocab"], max_len=cfg["max_len"], d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], layers=cfg["layers"])
    sd = torch.load(os.path.join(root, artifacts, "zitran_gen_fp16.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()}); model.eval()
    return tok, model

def generate_text(tok, model, prompt, max_new=60, temperature=0.8, top_k=40, seed=None):
    tok.max_len = 100000
    pids = tok.encode(prompt)[:48]
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
