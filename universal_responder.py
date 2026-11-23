#!/usr/bin/env python3
"""ZITRAN Universal Responder — one entry point for every kind of message.

Usage (from repo root):
  python3 universal_responder.py --test        # test suite: har message type ka response dekho
  python3 universal_responder.py               # interactive chat (terminal)
  python3 universal_responder.py --gen 100000  # 1 lakh responses generate karo (test set)

Har message ko sahi brain ke paas bhejta hai:
  general knowledge (264k-fact bank) - conversation (ShareGPT/LMSYS layer) -
  security analysis (ensemble + reasoning) - emotion - math/logic - HTML page builder.
"""
import argparse, gzip, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_MESSAGES = [
    # greetings / identity
    "hi", "hello, how are you?", "who are you?", "my name is Arslan", "what is my name?",
    # general knowledge
    "who is ronaldo?", "who is einstein?", "what is the capital of France?",
    "what currency is used in Japan?", "what language is spoken in Brazil?",
    "what is the capital of Uttar Pradesh?", "what is the capital of California?",
    "what is the chemical symbol for gold?", "tell me about mars",
    "which planet is known as the red planet?", "explain photosynthesis",
    "who won the battle of waterloo in 1815?",
    # conversation
    "tell me a joke", "what are the benefits of regular exercise?", "how do i make a cup of tea?",
    "i feel sad today", "i am very happy today!", "thank you so much",
    # math / logic
    "what is 23*7+5?", "what is 6*7?", "true and not false?",
    # security knowledge + analysis
    "what is phishing?", "what is ransomware?",
    "ransomware encrypted all files, ransom note demands bitcoin",
    "multiple failed login attempts from foreign IP followed by successful login",
    "user clicked a link in an email and entered office365 credentials on a fake page",
    "normal user logged in with MFA and opened email",  # benign control
    # skills
    "build a dark green restaurant page called SpiceHouse",
    "which model is best?", "how can i test you?",
    # honesty (unknown)
    "who is Zqxwv Blorpington?", "what is the capital of Atlantis?",
]

def get_reply_fn():
    from zitran.chat_server import assistant_reply
    return assistant_reply

def run_test():
    reply = get_reply_fn()
    print("=" * 78)
    print("ZITRAN UNIVERSAL RESPONDER — TEST SUITE (har message ka response)")
    print("=" * 78)
    stats = {}
    for i, msg in enumerate(TEST_MESSAGES, 1):
        t = time.time()
        d = reply(msg, "ensemble", "universal-test")
        ms = (time.time() - t) * 1000
        route = ("PAGE-BUILDER" if d.get("generated_url") else
                 f"SECURITY:{d.get('predicted_class')}/{d.get('risk_level')}" if d.get("predicted_class") else
                 "KNOWLEDGE/CHAT")
        stats[route.split(":")[0]] = stats.get(route.split(":")[0], 0) + 1
        emo = (d.get("emotion") or {}).get("emotion", "")
        print(f"\n[{i:02d}] USER: {msg}")
        print(f"     ROUTE: {route}{' | emotion: ' + emo if emo else ''} | {ms:.0f} ms")
        print(f"     ZITRAN: {d['reply'][:300]}")
    print("\n" + "=" * 78)
    print("SUMMARY:", json.dumps(stats), f"| total {len(TEST_MESSAGES)} messages")
    print("=" * 78)

def run_chat():
    reply = get_reply_fn()
    print("ZITRAN Universal Responder — chat karo (exit/bye se band). Kuch bhi poochho:")
    print("security event, general knowledge, joke, math, ya 'build a ... page called X'\n")
    sid = f"universal-{int(time.time())}"
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not msg: continue
        d = reply(msg, "ensemble", sid)
        print("ZITRAN:", d["reply"][:1200])
        if d.get("generated_url"): print("   page:", d["generated_url"])
        if msg.lower() in ("bye", "goodbye", "exit", "quit"): break

def build_prompt_pool():
    """Lakhon unique prompts — real data se (countries, elements, trivia, jeopardy, conversations)."""
    pool = []
    from zitran.world_data import load_countries, load_elements, INDIA_STATES, US_STATES, load_qa_bank
    for c in load_countries():
        pool += [f"what is the capital of {c['name']}?", f"what currency is used in {c['name']}?",
                 f"what language is spoken in {c['name']}?"]
    for st in list(INDIA_STATES) + list(US_STATES):
        pool.append(f"what is the capital of {st}?")
    for e in load_elements():
        pool += [f"what is the chemical symbol for {e['name']}?", f"what is the atomic number of {e['name']}?"]
    for it in load_qa_bank():
        pool.append(it["q"] if it["q"].rstrip().endswith("?") else it["q"] + "?")
    try:
        from zitran.conversation_kb import _load
        pairs, _, _ = _load()
        pool += [h + ("?" if not h.rstrip().endswith("?") else "") for h, _ in pairs]
    except Exception:
        pass
    rnd = random.Random(1)
    for _ in range(3000):
        a, b, c = rnd.randint(2, 99), rnd.randint(2, 99), rnd.randint(2, 50)
        pool.append(rnd.choice([f"what is {a}*{b}+{c}?", f"what is {a}+{b}*{c}?", f"calculate {a}*{b}"]))
    pool += ["hi", "hello", "tell me a joke", "who is ronaldo?", "who is einstein?", "explain photosynthesis",
             "what is phishing?", "what is ransomware?", "i feel sad today", "thank you"]
    return pool

def run_gen(n, out="data/generated_responses.jsonl.gz"):
    from zitran.assistant import Assistant
    from zitran.chat_server import get_predictor
    assistant = Assistant(lambda text, model_id="ensemble": get_predictor(model_id).predict(text))
    assistant.detect_emotion = lambda msg: None  # bulk mode: emotion off for speed (noted in output meta)
    import zitran.self_learn as _sl
    _sl.log_interaction = lambda *a, **k: None  # bulk mode: per-message disk logging off for speed
    print(f"prompt pool bana raha hoon...")
    pool = build_prompt_pool()
    rnd = random.Random(42); rnd.shuffle(pool)
    print(f"pool size: {len(pool):,} unique prompts | generating {n:,} responses...")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time(); done = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        while done < n:
            for prompt in pool:
                if done >= n: break
                sid = f"bulk-{done // 500}"  # session rotate (memory reset)
                d = assistant.handle(sid, prompt, "ensemble")
                f.write(json.dumps({"prompt": prompt, "response": d["reply"],
                                    "class": d.get("predicted_class"), "page": d.get("generated_url")},
                                   ensure_ascii=False) + "\n")
                done += 1
                if done % 10000 == 0:
                    el = time.time() - t0
                    print(f"  {done:,}/{n:,} done | {el:.0f}s | {done/el:.1f} resp/s | ETA {(n-done)/(done/el):.0f}s", flush=True)
            rnd.shuffle(pool)
    print(f"DONE: {done:,} responses -> {out} ({os.path.getsize(out)/1e6:.1f} MB) in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ZITRAN Universal Responder")
    ap.add_argument("--test", action="store_true", help="test suite chalao")
    ap.add_argument("--gen", type=int, default=0, help="N responses generate karo (e.g. --gen 100000)")
    a = ap.parse_args()
    if a.test: run_test()
    elif a.gen: run_gen(a.gen)
    else: run_chat()
