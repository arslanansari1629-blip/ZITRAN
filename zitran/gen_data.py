"""Build the generative corpus for ZITRAN-Gen from every local source.
Defensive framing only: scenario descriptions, detection, impact, MITRE and
solutions from the refined Attack_Dataset (Attack Steps / Tools columns are
 deliberately EXCLUDED), knowledge-base QA, incident->class examples,
Cornell dialogue pairs, and programmatic empathetic-dialogue examples.
Documents are separated by <|endoftext|>.
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.data import generate, LABELS

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEP = "\n<|endoftext|>\n"

def build_corpus(out="data/gen_corpus.txt", seed=42):
    rng = random.Random(seed); docs = []
    # 1) Refined attack dataset -> educational QA (no attack steps/tools)
    rp = os.path.join(ROOT, "data", "attack_dataset_refined.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            r = json.loads(line)
            docs.append(f"Question: What is {r['title']}?\nAnswer: {r['text'].split('. attack',1)[-1].strip()[:700]} This is classified as {r['label']}.")
            docs.append(f"Question: How do you detect and respond to {r['attack_type_raw']}?\nAnswer: Detection: {r['detection']} Impact: {r['impact']} Recommended response: {r['solution']} MITRE ATT&CK: {r['mitre']}.")
            docs.append(f"Incident report: {r['event_text']}\nClassification: {r['label']}. Recommended action: {r['solution']}")
    # 2) Knowledge base QA
    from zitran.assistant import KNOWLEDGE
    for k, v in KNOWLEDGE.items():
        docs.append(f"Question: What is {k}?\nAnswer: {v}")
        docs.append(f"Question: How can we defend against {k}?\nAnswer: {v}")
    # 3) Synthetic incident -> classification + explanation documents
    for text, lab in generate(500, seed=7):
        docs.append(f"Incident: {text}\nAnalysis: This event is classified as {LABELS[lab]}. It should be handled according to the {LABELS[lab]} incident response playbook with risk based triage.")
    # 4) Cornell dialogue pairs (casual conversation ability)
    dp = os.path.join(ROOT, "data", "dialogue_pairs.jsonl")
    if os.path.exists(dp):
        pairs = [json.loads(l) for l in open(dp)][:12000]
        for p in pairs:
            docs.append(f"User: {p['context']}\nAssistant: {p['response']}")
    # 5) Programmatic empathetic dialogue (emotion-aware talking)
    feelings = [("sad", "I'm really sorry you're going through this. You matter, and this feeling will pass. I'm here to listen for as long as you need."),
                ("stressed", "That sounds like a lot of pressure. Let's slow down together — one breath, one small step at a time. What's weighing on you most?"),
                ("happy", "That's wonderful to hear! Your joy is contagious. Tell me everything — what happened?"),
                ("angry", "I completely understand why you'd feel angry about that. Your feelings are valid. Want to talk through what happened?"),
                ("scared", "It's okay to feel scared — you're not alone in this. I'm right here with you. Can you tell me what's worrying you?"),
                ("lonely", "I'm glad you told me. You're not alone right now — I'm here, and I enjoy talking with you. What's on your mind?")]
    names = ["friend", "Alex", "Sam", "Priya", "Jordan"]
    for feeling, resp in feelings:
        for n in names:
            docs.append(f"User: Hi, I am feeling {feeling} today.\nAssistant: {resp}")
            docs.append(f"User: I feel so {feeling}, I don't know what to do.\nAssistant: {resp}")
    # 6) Self-knowledge documents
    docs.append("Question: Who are you?\nAnswer: I am ZITRAN, a defensive cybersecurity AI built from scratch. I classify threats, explain my reasoning, detect emotions, and generate security guidance. I help defenders; I do not create attacks.")
    docs.append("Question: What can you do?\nAnswer: I analyse security events such as logs, URLs, emails and flow records, classify them into threat categories, assess risk, recommend response actions, answer cybersecurity questions, detect emotions in messages, and hold empathetic conversations.")
    rng.shuffle(docs)
    with open(os.path.join(ROOT, out), "w") as f:
        f.write(SEP.join(docs) + "\n")
    return docs

def tokenize_corpus(tok, docs, max_len=100000):
    """Token-id stream using an existing BPE tokenizer's vocab."""
    # build ids directly (no padding): CLS + word tokens + SEP per doc
    cls_id = tok.token_to_id["<CLS>"]; sep_id = tok.token_to_id["<SEP>"]; unk = tok.token_to_id["<UNK>"]
    ids = []
    for d in docs:
        ids.append(cls_id)
        for w in tok._words(d):
            for t in tok._encode_word(w):
                ids.append(tok.token_to_id.get(t, unk))
        ids.append(sep_id)
    return ids

if __name__ == "__main__":
    docs = build_corpus()
    chars = sum(len(d) for d in docs)
    print(f"docs={len(docs)} chars={chars/1e6:.1f}M ~tokens={chars/4/1e6:.2f}M")
    from collections import Counter
    print("sample:", docs[0][:240])
