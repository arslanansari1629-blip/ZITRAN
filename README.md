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

## ✍️ ZITRAN-Gen — generative model (`artifacts_gen2/`) + emotion brain (`artifacts_emotion/`)

**Generative brain:** decoder-only Transformer (19,312,128 params, d=384, 10 layers,
causal LM with temperature/top-k sampling — `zitran/gen_model.py`, `zitran/generate.py`,
live at `POST /api/generate`). Corpus = **2.48M tokens / 26k documents from everywhere
reachable**: all cybersecurity sources (attack scenarios, detection/solutions, NSL-KDD,
CTU-13, UNSW, Enron, JPCERT, URLs) + general instruction data. Requested
`load_dataset("teknium/OpenHermes-2.5")` is attempted first in `zitran/instruction_data.py`
but huggingface.co is unreachable from this sandbox, so training used the same library
with the GitHub-hosted **Stanford Alpaca (52k)** fallback — swap in OpenHermes on an
unrestricted machine with zero code changes.

**Training result:** 320 steps × 24 × 80 tokens (614k tokens), test perplexity
**4000 (random) → 103.8**. Verbatim samples are stored in `artifacts_gen2/metrics.json` —
read them before believing any hype: it produces topical, format-aware but often
incoherent text. **It is not Claude and not a "god model"** — frontier LLMs train on
trillions of tokens across thousands of GPUs; this ran ~25 minutes on 2 CPU cores.
What it *is*: a real, from-scratch generative foundation that improves predictably
with more steps (`python3 -m zitran.train_gen2 --max-steps 5000`) and the right
architecture to scale. For production answers, the classifier ensemble (94-98%)
+ reasoning assistant remain the reliable brain; Gen is the creative draft engine.

**Emotion brain:** GoEmotions-trained classifier (Ekman 7-class) — **57.5% accuracy /
0.48 macro-F1 on the official held-out test** (`artifacts_emotion/metrics.json`).
The assistant reads the emotion of every message, tracks session mood, and adapts
(empathetic openers under stress, celebration on joy); Cornell-dialogue retrieval
(`zitran/dialogue.py`, 40k pairs in `data/dialogue_pairs.jsonl`) powers casual chat.

## 🧠 Conversational assistant upgrade — talk, reasoning & logic

ZITRAN now **talks like an assistant**, not just a classifier (`zitran/assistant.py` +
`zitran/reasoning.py`, both in the web chat and CLI, with per-session memory):

- **Dialogue**: greetings, remembers your name & the last analysis, follow-ups —
  ask **"why?"** (full evidence trail), **"what should I do?"** (response playbook),
  **"tell me more / MITRE?"**, **"which model is best?"** (live metrics comparison)
- **Reasoning trail** for every verdict: Observe → URL/evidence extraction →
  corroboration (rules vs neural model) → elimination (why the runner-up lost) →
  risk logic → **grounding in the closest real scenario from the refined
  Attack_Dataset (MITRE technique, detection method, solution)**
- **Logic & math**: safe built-in evaluator ("what is 23*7+5?" → 166; boolean logic too)
- **Knowledge base**: defensive explanations (phishing, ransomware, SQLi, XSS, MITRE, Zero Trust, MFA, SOC…)
- **Safety**: refuses attack instructions, offers the defensive playbook instead
- Honest scope: this is a specialist analyst with structured reasoning — not a
  general-purpose LLM; open-domain chat outside security will be routed to analysis or answered from the KB.

Test the conversation:
```bash
python3 tests/test_assistant.py     # memory, follow-ups, math, knowledge, refusal
# in chat, try this sequence:  "my name is Alex" → paste an incident → "why?" →
# "what should I do?" → "what is my name?" → "what is 23*7+5?"
```

## 💬 Chat interface — and how to test it

**Web chat (easiest):**
```bash
pip install numpy --break-system-packages   # only dependency for v1–v4/ensemble
python3 -m zitran.chat_server --port 8000
```
Open **http://localhost:8000** in your browser. Pick a model (Ensemble recommended,
or v1/v2/v3/v4 individually), click an example chip or type any security event —
log line, suspicious URL, email text, flow record, or plain English — and press
**Analyse**. You get: threat class, risk badge (NONE/MEDIUM/HIGH/CRITICAL),
confidence bar, threat probability, recommended action, runner-up classes and
each model's vote (ensemble mode).

**Terminal chat:**
```bash
python3 -m zitran.chat_cli          # ensemble
python3 -m zitran.chat_cli v4       # single model; type 'model v2' to switch, 'quit' to exit
```

**Test it yourself (4 ways):**
```bash
python3 tests/test_chat.py           # 1. automated chat tests (asserts ransomware/benign replies)
curl http://127.0.0.1:8000/api/health                                   # 2. server health + model list
curl -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"ransom note pay bitcoin files encrypted","model":"ensemble"}'   # 3. raw API
python3 -m zitran.test_responses     # 4. full 14-prompt response benchmark, all models
```
Good test prompts: `ransom note pay bitcoin files encrypted locked` ·
`url https://secure-login-verify.top/account/update` ·
`user alice logged in successfully via mfa` (should say benign) ·
`union select password dump sql injection login bypass`.
Note: ZITRAN is a defensive *classifier* — it analyses and advises; it does not
generate attacks, exploits or malware.

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

## ZITRAN v2 — real-data upgrade (`artifacts_v2/`, `zitran/train_v2.py`)

Trained on **real public data from 5 sources** (provenance: `data/SOURCES.md`):
NSL-KDD intrusion flows, CTU-13 botnet traffic, Enron spam/ham emails, JPCERT/CC
phishing URLs (~290k), and a ~650k malicious/benign URL corpus — with synthetic
data only topping up classes that have no public corpus (ransomware, exfiltration).
Unified corpus: 21,673 samples (`data/dataset_v2.jsonl`, each row tagged by source).

**Upgrades:** 1,088,508 params (d=144, 3 layers, ff=288), BPE vocab 4,000, seq 56,
label smoothing, class-weighted loss, LR warmup+cosine, EMA weights, early stopping.

**Results (honest, no cherry-picking):**
- Mixed real-data validation: **96.5%** accuracy (per-class F1 0.90–1.00)
- **Official NSL-KDD KDDTest+ — attack types never seen in training: 71.6% exact-class,
  84.9% threat-vs-benign** (DoS F1 0.88, recon F1 0.77; novel credential attacks are the weak spot)
- Trade-off: v2 is specialised for structured telemetry (flows/URLs/emails — real SOC data).
  For free-text incident descriptions, v1 (`artifacts/`) generalises better. Use v2 for
  telemetry triage, v1 for analyst-style text, or ensemble both.

```bash
python3 -m zitran.fetch_data   # download the 5 public datasets (~90 MB, git-ignored)
python3 -m zitran.train_v2     # retrain v2 from scratch
python3 -m zitran.predict artifacts_v2 "url https://secure-login-verify.top/account/update"
```

## ZITRAN v3 — ultra (`artifacts_v3/`) + Ensemble (`zitran/ensemble.py`)

**Upgrades:** MLP classification head (gradient-checked), paraphrase augmentation
(`zitran/augment.py`), 65,673-sample corpus (real telemetry + augmented free-text,
26k balanced for training), 1,296,652 params (d=160, 3 layers, ff=320), EMA 0.995,
early stopping, and **temperature calibration** (T=0.70; calibration error ECE
0.035 → 0.006, so confidences are trustworthy).

**Results:** validation **97.6%**; official KDDTest+ novel-attack binary threat
detection **89.6%** (exact-class 64.6% — augmented free-text trades some telemetry
specialisation for wording robustness; v2 remains best on pure telemetry at 71.6%).

**Response test** (`zitran/test_responses.py`, 14 mixed prompts — telemetry, URLs,
emails, free-text; full table in `artifacts_v3/response_test.md`):

| Model | Exact-class |
|---|---|
| v1 | 64.3% |
| v2 | 35.7% |
| v3 | 71.4% |
| **ENSEMBLE (v1+v2+v3, accuracy-weighted)** | **78.6%** |

```bash
python3 -m zitran.train_v3        # retrain v3
python3 -m zitran.test_responses  # what each model says, prompt by prompt
python3 -m zitran.ensemble "ransom note pay bitcoin files encrypted locked"
```

## Roadmap ideas
- Train on real public corpora (e.g. CIC-IDS2017-style flows, phishing-email corpora) by swapping `data.py`
- Add log-sequence mode, larger vocab/d-model, quantization for edge deployment, ONNX export

## 🌍 General knowledge + 🛠️ HTML page builder (assistant skills)

The chat assistant now also:
- **Answers general questions** (`zitran/general_kb.py`): curated facts (Ronaldo,
  Messi, Einstein, Gandhi, Elon Musk…), ~90 country capitals, plus retrieval over the
  Alpaca instruction data. Unknown people get an honest "I don't know" — no hallucination.
- **Builds web pages on demand** (`zitran/codegen.py`): say *"build a dark green
  restaurant page called SpiceHouse"* and it writes a complete single-file HTML page
  (themes, login/portfolio/dashboard/shop/restaurant/landing kinds, inline CSS/JS) to
  `web/generated/`, served live at `/generated/<name>.html` with a clickable link in chat.
  Example output committed at `web/generated/spicehouse.html`.
- Security analysis, emotion reading, dialogue, math/logic and `POST /api/generate`
  (ZITRAN-Gen) all remain available in the same chat. Tests: `tests/test_skills.py`.
