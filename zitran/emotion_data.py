"""Emotion + conversation dataset loaders.

- GoEmotions (Google, via monologg/GoEmotions-pytorch): Reddit comments
  labelled with Ekman emotions — data/raw/goemotions/{train,dev,test}.tsv
- Cornell Movie-Dialogs Corpus: movie_lines/movie_conversations ->
  (context, response) dialogue pairs for conversational retrieval.
"""
import ast, csv, json, os, random, re

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def load_goemotions(split="train", cap_per_class=None, seed=42):
    path = os.path.join(ROOT, "data", "raw", "goemotions", f"{split}.tsv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3: continue
            text = "\t".join(row[:-2]).strip()
            lab = int(row[-2].split(",")[0])  # first label if multi-label
            if text: rows.append((text, lab))
    if cap_per_class:
        rng = random.Random(seed); by = {}
        for t, l in rows: by.setdefault(l, []).append(t)
        out = []
        for l, ts in by.items():
            rng.shuffle(ts)
            out += [(t, l) for t in ts[:cap_per_class]]
        rng.shuffle(out); return out
    return rows

def build_dialogue_pairs(max_pairs=40000, seed=42, out_path="data/dialogue_pairs.jsonl"):
    """Parse Cornell corpus into clean context->response pairs."""
    lines = {}
    lp = os.path.join(ROOT, "data", "raw", "cornell", "movie_lines.txt")
    cp = os.path.join(ROOT, "data", "raw", "cornell", "movie_conversations.txt")
    with open(lp, encoding="utf-8", errors="ignore") as f:
        for row in f:
            parts = row.split(" +++$+++ ")
            if len(parts) >= 5:
                lines[parts[0].strip()] = parts[4].strip()
    pairs = []
    def clean(t):
        t = re.sub(r"\s+", " ", t).strip()
        return t
    with open(cp, encoding="utf-8", errors="ignore") as f:
        for row in f:
            parts = row.split(" +++$+++ ")
            if len(parts) < 4: continue
            try: ids = ast.literal_eval(parts[3].strip())
            except Exception: continue
            for a, b in zip(ids, ids[1:]):
                if a in lines and b in lines:
                    ctx, resp = clean(lines[a]), clean(lines[b])
                    if 2 <= len(ctx.split()) <= 30 and 1 <= len(resp.split()) <= 30 and len(resp) > 2:
                        pairs.append((ctx, resp))
    rng = random.Random(seed); rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    with open(os.path.join(ROOT, out_path), "w") as f:
        for c, r in pairs: f.write(json.dumps({"context": c, "response": r}) + "\n")
    return pairs

def load_dialogue_pairs(path="data/dialogue_pairs.jsonl"):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p)]
