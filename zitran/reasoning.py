"""ZITRAN reasoning engine — explainable, step-by-step logic on top of the
classifier. 100% from scratch (rules + statistics + dataset lookup):

1. Evidence extraction: class-specific indicators in the raw input
   (keywords, URL structure features, entropy, flow fields).
2. Hypothesis reasoning: combine classifier probability with evidence
   scores; explain why the winner beat the runner-up.
3. Risk logic: deterministic escalation rules (impact signals + class).
4. Grounding: nearest matching scenario in the refined Attack_Dataset
   supplies real MITRE technique, detection method and solution.
Also: a safe math/boolean-logic evaluator for user questions.
"""
import ast, json, math, os, re
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")

EVIDENCE = {
    "ransomware": ["ransom", "encrypt", "shadow copies", "vssadmin", ".locked", "bitcoin", "decrypt", "crypto locker", "extortion"],
    "phishing": ["verify your account", "click link", "sign in", "login page", "gift card", "wire transfer", "urgent", "password expires", "bank details", "otp", "credential harvest"],
    "malware": ["trojan", "malware", "quarantine", "injected", "unsigned", "payload", "virus", "rootkit", "shellcode", "macro"],
    "intrusion_dos": ["flood", "ddos", "denial of service", "packets per second", "botnet traffic", "amplification", "slowloris", "saturat"],
    "exploit_injection": ["sql injection", "union select", "' or '1'='1", "xss", "<script>", "etc/passwd", "command injection", "jndi", "buffer overflow", "../", "template injection"],
    "credential_attack": ["failed login", "brute force", "password spraying", "credential stuffing", "passwords against", "dictionary attack", "mfa fatigue", "pass the hash", "kerberoast"],
    "exfiltration": ["exfiltrat", "outbound transfer", "uploaded", "dns tunneling", "foreign server", "gigabytes", "data leak", "usb copy", "personal cloud"],
    "recon_scan": ["port scan", "nmap", "probing", "enumeration", "recon", "scanning", "fingerprint", "subdomain", "banner grabbing"],
    "command_control": ["beacon", "command and control", "c2", "evil-c2", "random generated domains", "dga", "reverse shell", "polling", "covert"],
    "benign": ["successfully", "completed successfully", "mfa", "verified", "no threats", "up to date", "status success", "logged in successfully"],
}
IMPACT_SIGNALS = ["admin", "root", "database", "credentials", "payment", "bitcoin", "encrypted", "takeover", "steal", "leak"]

def extract_evidence(text):
    low = text.lower(); found = {}
    for label, keys in EVIDENCE.items():
        hits = [k for k in keys if k in low]
        if hits: found[label] = hits
    return found

def url_features(text):
    m = re.search(r"(https?://)?([a-z0-9\-\.]+\.[a-z]{2,})", text.lower())
    if not m: return None
    host = m.group(2); feats = []
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host): feats.append("host is a raw IP address")
    if host.count(".") >= 3: feats.append(f"deep subdomain chain ({host.count('.')} dots)")
    if host.count("-") >= 2: feats.append("multiple hyphens (lookalike-brand pattern)")
    for tld in [".top", ".xyz", ".click", ".ru", ".su", ".buzz"]:
        if host.endswith(tld): feats.append(f"high-risk TLD {tld}")
    for brand in ["login", "verify", "secure", "account", "update", "auth", "bank"]:
        if brand in host: feats.append(f"trust-word '{brand}' inside hostname")
    chars = Counter(host); ent = -sum((c/len(host)) * math.log2(c/len(host)) for c in chars.values())
    if ent > 3.3: feats.append(f"high hostname entropy ({ent:.1f}) — algorithmically generated look")
    if "@" in text: feats.append("'@' in URL (hides real destination)")
    return {"host": host, "features": feats, "entropy": round(ent, 2)}

_scenario_cache = None
def nearest_scenario(label, text, limit_label_rows=400):
    """Ground the answer in the refined Attack_Dataset (MITRE/detection/solution)."""
    global _scenario_cache
    path = os.path.join(ROOT, "data", "attack_dataset_refined.jsonl")
    if not os.path.exists(path): return None
    if _scenario_cache is None:
        _scenario_cache = [json.loads(l) for l in open(path)]
    words = set(re.findall(r"[a-z]{4,}", text.lower()))
    best, best_score = None, 0
    seen = 0
    for r in _scenario_cache:
        if r["label"] != label: continue
        seen += 1
        if seen > limit_label_rows * 20: break
        score = len(words & set(re.findall(r"[a-z]{4,}", (r["title"] + " " + r["attack_type_raw"] + " " + r["text"][:300]).lower())))
        if score > best_score: best, best_score = r, score
    if best and best_score >= 2:
        return {"title": best["title"], "mitre": best["mitre"], "detection": best["detection"],
                "solution": best["solution"], "impact": best["impact"], "match_score": best_score}
    return None

def reason(text, prediction):
    """Build the reasoning trace for a classifier prediction."""
    steps = []
    ev = extract_evidence(text)
    steps.append({"step": "Observe", "detail": f"Input type: {'URL-bearing' if 'http' in text.lower() or '.top' in text.lower() or '.xyz' in text.lower() else 'text/flow event'} ({len(text)} chars, {len(text.split())} tokens)."})
    uf = url_features(text)
    if uf and uf["features"]:
        steps.append({"step": "URL analysis", "detail": f"Host {uf['host']}: " + "; ".join(uf["features"]) + "."})
    winner = prediction["predicted_class"]
    if ev:
        ranked = sorted(ev.items(), key=lambda kv: -len(kv[1]))
        steps.append({"step": "Evidence", "detail": "Indicators found — " + "; ".join(
            f"{lab}: {', '.join(hits[:4])}" for lab, hits in ranked[:3]) + "."})
        top_ev = ranked[0][0]
        if top_ev == winner:
            steps.append({"step": "Corroboration", "detail": f"Rule evidence and the neural model independently agree on '{winner}' — confidence reinforced."})
        else:
            steps.append({"step": "Cross-check", "detail": f"Keyword evidence leans '{top_ev}' but the trained model (which weighs full context, not just keywords) chose '{winner}'. Model verdict kept; treat as a signal to review."})
    else:
        steps.append({"step": "Evidence", "detail": "No strong keyword indicators — verdict rests on the model's learned patterns (word order, feature combinations)."})
    top3 = prediction.get("top3", [])
    if len(top3) > 1:
        steps.append({"step": "Elimination", "detail": f"Runner-up '{top3[1]['class']}' scored {top3[1]['prob']:.0%} vs winner {top3[0]['prob']:.0%} — margin {abs(top3[0]['prob']-top3[1]['prob']):.0%}; " +
                      ("clear decision." if top3[0]['prob'] - top3[1]['prob'] > 0.3 else "close call — human review recommended.")})
    impacts = [s for s in IMPACT_SIGNALS if s in text.lower()]
    steps.append({"step": "Risk logic", "detail": f"Class base-risk + impact signals {impacts or 'none'} + confidence {prediction['confidence']:.0%} → final risk {prediction['risk_level']}."})
    scen = nearest_scenario(winner, text) if winner != "benign" else None
    if scen:
        steps.append({"step": "Grounding", "detail": f"Closest known scenario in the refined attack dataset: '{scen['title']}' (MITRE {scen['mitre']})."})
    return {"steps": steps, "evidence": ev, "url_features": uf, "scenario": scen}

# ---------- safe math / boolean logic ----------
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Mod, ast.Pow, ast.FloorDiv, ast.USub, ast.UAdd, ast.BoolOp, ast.And, ast.Or, ast.Not,
            ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

def evaluate_logic(expr):
    """Safely evaluate arithmetic / boolean expressions. Returns result or None."""
    expr = expr.strip().lower().replace("×", "*").replace("÷", "/")
    expr = expr.replace("true", "True").replace("false", "False")
    expr = re.sub(r"\b(and|or|not)\b", lambda m: m.group(1).capitalize() if False else m.group(1), expr)
    if not re.fullmatch(r"[0-9\.\s\+\-\*/%\(\)<>!=andornotTrueFals]+", expr) or not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                return None
        return eval(compile(tree, "<logic>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
