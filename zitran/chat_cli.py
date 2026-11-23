"""Terminal chat with ZITRAN. Usage: python3 -m zitran.chat_cli [ensemble|v1|v2|v3|v4]
Type a security event / log / URL / email text; 'quit' to exit, 'model v4' to switch.
"""
import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.chat_server import assistant_reply, available_models

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ensemble"
    sid = "cli-" + uuid.uuid4().hex[:8]
    print(f"ZITRAN assistant ({model}) — models: {[m['id'] for m in available_models()]}. Talk naturally; 'model v4' switches, 'quit' exits.\n")
    while True:
        try: msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if msg.lower() in ("quit", "exit"): break
        if msg.lower().startswith("model "):
            model = msg.split(None, 1)[1]; print(f"[switched to {model}]"); continue
        if not msg: continue
        r = assistant_reply(msg, model, sid)
        print("zitran>", r["reply"].replace("**", ""))

if __name__ == "__main__":
    main()
