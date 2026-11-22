"""ZITRAN Chat server — zero extra dependencies (Python stdlib + NumPy).

Serves the web chat UI (web/index.html) and a JSON API:
  GET  /              -> chat UI
  GET  /api/health    -> {"status": "ok", "models": [...]}
  GET  /api/models    -> model list with accuracy/params from metrics.json
  POST /api/chat      -> {"message": "...", "model": "ensemble|v1|v2|v3|v4"}
                         returns prediction, risk, action, top3, model votes

Run:  python3 -m zitran.chat_server --port 8000   (binds 0.0.0.0)
"""
import argparse, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zitran.predict import ZitranDetector
from zitran.ensemble import ZitranEnsemble

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIRS = {"v1": "artifacts", "v2": "artifacts_v2", "v3": "artifacts_v3", "v4": "artifacts_v4"}
_cache = {}
_lock = threading.Lock()

def available_models():
    out = []
    for name, d in MODEL_DIRS.items():
        if os.path.isdir(os.path.join(ROOT, d)) and any(f.endswith("_model.npz") for f in os.listdir(os.path.join(ROOT, d))):
            info = {"id": name, "name": f"ZITRAN {name}", "dir": d}
            mp = os.path.join(ROOT, d, "metrics.json")
            if os.path.exists(mp):
                m = json.load(open(mp))
                info.update({"accuracy": m.get("accuracy"), "params": m.get("params"),
                             "official_kddtest": m.get("official_kddtest_accuracy")})
            out.append(info)
    if len(out) > 1:
        out.insert(0, {"id": "ensemble", "name": "ZITRAN Ensemble (v1–v4, recommended)", "dir": "ensemble"})
    return out

def get_predictor(model_id):
    with _lock:
        if model_id in _cache:
            return _cache[model_id]
        if model_id == "ensemble":
            dirs = tuple(MODEL_DIRS[m["id"]] for m in available_models() if m["id"] != "ensemble")
            pred = ZitranEnsemble(dirs)
        elif model_id in MODEL_DIRS:
            pred = ZitranDetector(os.path.join(ROOT, MODEL_DIRS[model_id]))
        else:
            raise ValueError(f"unknown model '{model_id}'")
        _cache[model_id] = pred
        return pred

def chat_reply(message, model_id="ensemble"):
    message = (message or "").strip()
    if not message:
        return {"error": "empty message"}
    if len(message) > 4000:
        message = message[:4000]
    pred = get_predictor(model_id)
    r = pred.predict(message)
    reply = (f"**{r['predicted_class']}** · risk **{r['risk_level']}** · confidence {r['confidence']:.0%}\n"
             f"{'🚨 Threat detected. ' if r['is_threat'] else '✅ Looks benign. '}{r['recommended_action']}")
    return {"reply": reply, "model_used": model_id, **r}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            p = os.path.join(ROOT, "web", "index.html")
            self._send(200, open(p, "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, {"status": "ok", "models": [m["id"] for m in available_models()]})
        elif self.path == "/api/models":
            self._send(200, {"models": available_models()})
        else:
            self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, chat_reply(body.get("message", ""), body.get("model", "ensemble")))
        except Exception as e:
            self._send(400, {"error": str(e)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"ZITRAN chat on http://{a.host}:{a.port} models={[m['id'] for m in available_models()]}", flush=True)
    print("Warming up ensemble...", flush=True)
    try: get_predictor("ensemble")
    except Exception as e: print(f"ensemble warmup: {e}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
