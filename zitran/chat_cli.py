"""Terminal chat with ZITRAN. Usage: python3 -m zitran.chat_cli [ensemble|v1|v2|v3|v4]
Type a security event / log / URL / email text; 'quit' to exit, 'model v4' to switch.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.chat_server import chat_reply, available_models

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ensemble"
    print(f"ZITRAN chat ({model}) — models: {[m['id'] for m in available_models()]}. 'quit' to exit.\n")
    while True:
        try: msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if msg.lower() in ("quit", "exit"): break
        if msg.lower().startswith("model "):
            model = msg.split(None, 1)[1]; print(f"[switched to {model}]"); continue
        if not msg: continue
        r = chat_reply(msg, model)
        if "error" in r: print("zitran>", r["error"]); continue
        print(f"zitran> [{r['predicted_class']}] risk={r['risk_level']} confidence={r['confidence']:.0%}")
        print(f"        action: {r['recommended_action']}")
        print(f"        top3: {[(t['class'], t['prob']) for t in r['top3']]}")
        if r.get("model_votes"): print(f"        votes: { {k: v['class'] for k, v in r['model_votes'].items()} }")

if __name__ == "__main__":
    main()
