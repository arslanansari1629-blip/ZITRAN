# ZITRAN — A New Era

**ZITRAN** is an advanced **defensive cybersecurity AI model built 100% from scratch** —
no pretrained weights, no fine-tuning, no ML frameworks. Only NumPy + Python stdlib.

> ⚠️ Scope: ZITRAN is a *defensive detection* model (classify security events, recommend
> response). It does not generate exploits, malware, or attack instructions.

## What was built from scratch

| Component | Implementation |
|---|---|
| Tokenizer | Byte-Pair Encoding (BPE) trained from zero on the corpus (`zitran/tokenizer.py`) |
| Dataset | Programmatic generator: 10 classes × combinatorial templates/entities/obfuscation (`zitran/data.py`) |
| Model | Transformer encoder: multi-head self-attention, positional embeddings, pre-norm, GELU FFN (`zitran/model.py`) |
| Training | Hand-written backpropagation, Adam, cosine LR schedule, gradient clipping — no autograd |
| Heads | Multi-task: 10-class attack classification + binary threat detection |
| Verified | Numerical gradient-check test proves the manual backprop is correct (`tests/test_zitran.py`) |

## Classes
`benign, malware, phishing, intrusion_dos, exploit_injection, credential_attack, exfiltration, recon_scan, ransomware, command_control`

Each prediction also returns: confidence, threat probability, risk level
(`NONE/MEDIUM/HIGH/CRITICAL`), top-3 classes, and a recommended defensive action.

## Quickstart (free, CPU-only)

```bash
pip install numpy --break-system-packages   # only dependency
python3 tests/test_zitran.py                # gradient check + unit tests
python3 -m zitran.train --epochs 14 --per-class 2200   # train from scratch
python3 -m zitran.predict                   # demo predictions
python3 -m zitran.predict artifacts "ransom note pay bitcoin files encrypted .locked"
```

## Results (actual run in this repo)

- **Model:** 591,884 parameters, d_model=128, 2 Transformer layers, 4 heads, BPE vocab 2,500, seq len 40
- **Data:** 16,000 generated samples (1,600/class), 85/15 stratified split
- **Training:** 10 epochs, pure-NumPy backprop + Adam + cosine LR, ~14 min on 2 CPU cores, $0 (no GPU/cloud)
- **Held-out synthetic validation: 100% accuracy** (10-class and binary threat) — see `artifacts/metrics.json`.
  ⚠️ Honest caveat: this split shares templates with training, so it saturates. On 10 hand-written
  *unseen paraphrases*, ZITRAN scored **7/10 exact class and 10/10 threat-vs-benign** — real-world use
  needs training on real corpora (see Roadmap), not just synthetic data.
- **Tests:** numerical gradient check + tokenizer + learning tests all pass (`tests/test_zitran.py`)

## Files
```
zitran/tokenizer.py   BPE from scratch
zitran/data.py        synthetic dataset generator (seeded, reproducible)
zitran/model.py       Transformer + manual backprop + Adam (pure NumPy)
zitran/train.py       end-to-end training + evaluation pipeline
zitran/predict.py     inference API + CLI with risk/action output
tests/test_zitran.py  tokenizer / gradient-check / learning tests
artifacts/            zitran_model.npz, tokenizer.json, metrics.json
data/dataset.jsonl    generated labelled dataset
examples/demo.py      usage demo
```

## Roadmap ideas
- Train on real public corpora (e.g. CIC-IDS2017-style flows, phishing-email corpora) by swapping `data.py`
- Add log-sequence mode, larger vocab/d-model, quantization for edge deployment, ONNX export
