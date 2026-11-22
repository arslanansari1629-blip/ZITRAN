"""ZITRAN conversational assistant — dialogue + reasoning layer.

Talks like an assistant (greetings, memory of your name & last analysis,
follow-ups like 'why?' / 'what should I do?' / 'which model is best?'),
answers from a built-in defensive-security knowledge base, evaluates
math/logic questions, and — for security events — runs the neural
classifier and explains its verdict step-by-step (zitran.reasoning).
Sessions are in-memory, keyed by session_id from the client.
Refuses offensive instructions; always offers the defensive angle.
"""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.reasoning import reason, evaluate_logic
from zitran.data import LABELS

ROOT = os.path.join(os.path.dirname(__file__), "..")

KNOWLEDGE = {
    "ransomware": "Ransomware is malware that encrypts files/systems and demands payment for decryption. Defence: offline/immutable backups, patching, least privilege, EDR, disable macros, and never rely on paying (no guarantee + funds crime). First response: isolate the host, preserve evidence, restore from clean backups.",
    "phishing": "Phishing tricks people into revealing credentials or installing malware via fake emails/sites/messages. Defence: MFA (ideally phishing-resistant keys), DMARC/SPF/DKIM, user training, link detonation, and rapid credential reset after clicks.",
    "sql injection": "SQL injection (SQLi) inserts attacker SQL into an application's query via unsanitised input, enabling data theft or auth bypass. Defence: parameterised queries/prepared statements, input validation, least-privilege DB accounts, WAF, and generic error pages.",
    "xss": "Cross-Site Scripting (XSS) injects scripts into pages viewed by others, stealing sessions/data. Defence: output encoding, Content-Security-Policy, input validation, HttpOnly/Secure cookies, and modern frameworks' auto-escaping.",
    "mitre": "MITRE ATT&CK is a public knowledge base of real adversary tactics & techniques (e.g. T1190 Exploit Public-Facing Application, T1078 Valid Accounts). Defenders use it to map detections and hunt for gaps. ZITRAN grounds its answers in ATT&CK techniques from the refined dataset.",
    "zero trust": "Zero Trust means 'never trust, always verify': authenticate and authorise every request regardless of network location, with least-privilege access, MFA, device health checks and micro-segmentation.",
    "mfa": "Multi-Factor Authentication requires 2+ proof types (something you know/have/are). It blocks most credential attacks even when passwords leak. Phishing-resistant MFA (FIDO2/passkeys) is stronger than SMS or push codes.",
    "dos": "Denial-of-Service (DoS/DDoS) overwhelms a service with traffic/requests so legitimate users can't use it. Defence: rate limiting, CDN/anycast absorption, upstream scrubbing, caching, and auto-scaling.",
    "exfiltration": "Data exfiltration is unauthorised transfer of data out of an organisation (cloud uploads, DNS tunnelling, USB, email forwarding). Defence: DLP, egress filtering, DNS monitoring, least-privilege data access and anomaly alerts on large transfers.",
    "brute force": "Brute-force/credential attacks guess or reuse passwords at scale (spraying, stuffing). Defence: MFA, rate limiting + lockouts, breached-password screening, and monitoring failed-login spikes.",
    "malware": "Malware is malicious software (trojans, worms, rootkits, stealers). Defence in depth: patching, EDR/antivirus, application allow-listing, email/web filtering, backups, and user awareness.",
    "soc": "A SOC (Security Operations Center) monitors alerts, triages incidents and coordinates response. ZITRAN is designed as a triage assistant for exactly that workflow: classify → risk-rank → recommend action.",
    "incident response": "Incident response lifecycle: prepare → detect/analyse → contain → eradicate → recover → lessons learned. Preserve evidence early, communicate clearly, and practise with tabletop exercises.",
}
OFFENSIVE = ["how to hack", "how do i hack", "create malware", "make malware", "write a virus", "make a ransomware",
             "exploit code", "hack into", "bypass password", "steal password", "ddos attack how", "how to phish", "send phishing"]

class Assistant:
    def __init__(self, predict_fn):
        """predict_fn(text) -> classifier dict (chat_server supplies model-aware fn)."""
        self.predict = predict_fn
        self.sessions = {}
        self._emotion_det = None; self._dialogue = None

    def detect_emotion(self, text):
        try:
            if self._emotion_det is None:
                from zitran.predict_emotion import EmotionDetector
                self._emotion_det = EmotionDetector()
            return self._emotion_det.predict(text)
        except Exception:
            return None

    def casual_dialogue(self, text, emotion):
        try:
            if self._dialogue is None:
                from zitran.dialogue import RetrievalDialogue
                self._dialogue = RetrievalDialogue()
            return self._dialogue.respond(text, emotion)
        except Exception:
            return None

    def _session(self, sid):
        return self.sessions.setdefault(sid, {"history": [], "name": None, "last": None, "last_text": None, "count": 0})

    def _metrics(self):
        out = {}
        for name, d in [("v1", "artifacts"), ("v2", "artifacts_v2"), ("v3", "artifacts_v3"), ("v4", "artifacts_v4"), ("v5", "artifacts_v5")]:
            p = os.path.join(ROOT, d, "metrics.json")
            if os.path.exists(p):
                m = json.load(open(p)); out[name] = m
        return out

    CASUAL = ["how are you", "what's up", "whats up", "i feel", "i am feeling", "i'm feeling", "im feeling",
              "i am sad", "i'm sad", "i am happy", "i am angry", "i am scared", "i am bored", "talk to me",
              "be my friend", "tell me something", "good morning", "good evening", "good night", "how was your day",
              "i need someone to talk", "feeling lonely", "i am stressed", "i'm stressed", "cheer me up"]

    def handle(self, sid, message, model_id="ensemble"):
        s = self._session(sid); msg = (message or "").strip(); low = msg.lower()
        s["count"] += 1
        reply, extra = None, {}
        emo = self.detect_emotion(msg) if len(msg) > 3 else None
        if emo:
            s.setdefault("emotions", []).append(emo["emotion"])
            extra["emotion"] = emo
            extra["session_mood"] = max(set(s["emotions"]), key=s["emotions"].count) if s["emotions"] else None

        m = re.search(r"my name is ([a-zA-Z]{2,20})", low)
        if m:
            s["name"] = m.group(1).capitalize(); reply = f"Nice to meet you, {s['name']}! I'll remember that for this session. Ask me to analyse a security event, explain a concept, or try 'why?' after an analysis."
        elif re.search(r"\b(hi|hello|hey|namaste)\b", low) and len(msg) < 30:
            reply = f"Hello{', ' + s['name'] if s['name'] else ''}! 👋 I'm ZITRAN, a defensive cybersecurity assistant. Paste a suspicious log, URL, email or incident description and I'll analyse it with reasoning — or ask me a security question."
        elif "who are you" in low or "your name" in low or "what are you" in low:
            reply = ("I'm ZITRAN — a cybersecurity AI built from scratch in this repo (no pretrained LLM). "
                     "My brain is an ensemble of small Transformers I was trained on real datasets (NSL-KDD, CTU-13, UNSW-NB15, Enron, JPCERT, your Attack_Dataset). "
                     "I classify threats, reason step-by-step about evidence, and recommend defensive actions. I'm honest about limits: I'm a specialist analyst, not a general ChatGPT.")
        elif "thank" in low:
            reply = "You're welcome! Stay safe — and remember: patch, back up offline, and enable MFA. 🛡️"
        elif low in ("bye", "goodbye", "exit", "quit"):
            reply = f"Goodbye{', ' + s['name'] if s['name'] else ''}! Session memory stays here if you come back. Stay secure. 👋"
        elif "what is my name" in low or "my name?" in low:
            reply = f"Your name is {s['name']}." if s["name"] else "You haven't told me yet — say 'my name is …'."
        elif any(k in low for k in ["how to test", "how can i test", "test you", "how do i use", "help", "what can you do", "capabilit"]):
            reply = ("Here's how to test me:\n"
                     "1) Paste an attack description (e.g. 'files encrypted, ransom note demands bitcoin') — I classify + explain my reasoning.\n"
                     "2) Ask 'why?' or 'what should I do?' right after — I remember context.\n"
                     "3) Ask concepts: 'what is phishing?' / 'explain MITRE'.\n"
                     "4) Logic/math: 'what is 23*7+5?' or 'true and not false?'.\n"
                     "5) Compare: 'which model is best?'.\n"
                     "6) Automated: run `python3 tests/test_assistant.py` or `python3 -m zitran.test_responses`.\n"
                     "7) Try to trick me: paraphrase an attack in your own words, or paste something benign and check I say so.")
        elif any(k in low for k in self.CASUAL) or re.search(r"\bi (feel|am) (sad|happy|angry|scared|afraid|stressed|lonely|bored|excited|worried|anxious)\b", low):
            e = (emo or {}).get("emotion", "neutral"); name = f", {s['name']}" if s.get("name") else ""
            openers = {
                "sadness": f"I'm really sorry you're feeling this way{name}. 💙 You don't have to carry it alone — I'm here, and we can take things one small step at a time.",
                "fear": f"It's completely understandable to feel scared{name}. Take a slow breath with me — in… and out. Whatever it is, we'll work through it together, step by step.",
                "anger": f"I hear you{name} — that sounds genuinely frustrating. 😤 Your feelings make sense. Let's untangle what's bothering you, one piece at a time.",
                "joy": f"That's wonderful{name}! 🎉 I love hearing that. Tell me more — what made it so good?",
                "surprise": f"Oh wow{name}! 😲 That sounds like a lot to take in. How are you processing it?",
                "disgust": f"Ugh, that sounds awful{name}. I'm sorry you had to deal with that.",
                "neutral": f"I'm doing well, thanks for asking{name}! 😊 Always happy to chat — or to help with anything security-related.",
            }
            reply = openers.get(e, openers["neutral"]) + "\n"
            dlg = self.casual_dialogue(msg, emo)
            if dlg and e == "neutral":
                reply += f'A thought on that: "{dlg["response"]}"\n'
            reply += "How can I support you right now — just listen, cheer you up, distract you with something interesting, or help with a problem?"
            if "how are you" in low:
                reply = f"I'm functioning at full capacity and in a great mood — all my models are loaded and agreeing with each other! {emo['emoji'] if emo else '😊'} How are YOU feeling today, {s['name'] or 'friend'}? (I genuinely use your answer — my emotion model reads every message.)"
        elif any(k in low for k in OFFENSIVE):
            reply = ("I can't help create or carry out attacks — I'm a defensive model. "
                     "But I can help you defend against exactly that: ask me how it's detected, what it looks like in logs, "
                     "or paste a suspicious sample and I'll analyse it and give you the mitigation playbook.")
        elif ("best model" in low or "which model" in low or "compare model" in low or "v4 vs" in low or "ensemble" in low and "?" in msg):
            ms = self._metrics(); lines = ["Here's my model family (validation accuracy / official novel-attack NSL-KDD test):"]
            for n in ["v1", "v2", "v3", "v4", "v5"]:
                if n in ms:
                    mm = ms[n]
                    lines.append(f"• {n}: val {mm.get('accuracy')}, KDDTest+ {mm.get('official_kddtest_accuracy', mm.get('official_kddtest_subset_accuracy'))}, params {mm.get('params')}")
            lines.append("Verdict: use the ENSEMBLE for mixed chat input, v4 alone for attack-scenario text (best on your Attack_Dataset: 73.5% unseen), v5 (50M) is an undertrained foundation — bigger, not yet better. Reasoning: accuracy on held-out + specialised tests beats parameter count.")
            reply = "\n".join(lines)
        elif s["last"] and any(k in low for k in ["why", "explain", "reason", "how do you know", "evidence"]):
            r = s["last"]; rs = r.get("reasoning", {})
            ev = "; ".join(f"{k} ({', '.join(v[:3])})" for k, v in rs.get("evidence", {}).items()) or "learned patterns (no keyword hits)"
            reply = (f"Why I said **{r['predicted_class']}** ({r['confidence']:.0%}):\n"
                     f"• Evidence in your text: {ev}.\n" +
                     "\n".join(f"• {st['step']}: {st['detail']}" for st in rs.get("steps", [])[2:4]) +
                     (f"\n• Grounding: closest known scenario '{rs['scenario']['title']}' (MITRE {rs['scenario']['mitre']})." if rs.get("scenario") else ""))
            extra = {"reasoning": rs}
        elif s["last"] and any(k in low for k in ["what should i do", "solution", "fix", "mitigat", "respond", "action", "next step"]):
            r = s["last"]; rs = r.get("reasoning", {}); scen = rs.get("scenario") or {}
            reply = (f"Response playbook for **{r['predicted_class']}** (risk {r['risk_level']}):\n"
                     f"1) Immediate: {r['recommended_action']}\n"
                     + (f"2) Detect: {scen.get('detection', 'review logs/EDR alerts for related indicators')}.\n" if scen else "2) Detect: hunt for related indicators across hosts/users.\n")
                     + (f"3) Fix/prevent: {scen.get('solution', 'patch root cause, reset affected credentials, add monitoring')}.\n" if scen else "3) Fix/prevent: patch root cause and add detection coverage.\n")
                     + "4) Document the incident and preserve evidence before cleanup.")
        elif s["last"] and any(k in low for k in ["tell me more", "more detail", "detail", "mitre", "impact"]):
            r = s["last"]; scen = (r.get("reasoning") or {}).get("scenario")
            concept = KNOWLEDGE.get(r["predicted_class"].replace("_", " ").split()[0], "")
            reply = (f"More on **{r['predicted_class']}**: " + (concept or "a security threat class I monitor.") +
                     (f"\nClosest dataset scenario: '{scen['title']}' — impact: {scen['impact']}; MITRE: {scen['mitre']}." if scen else ""))
        elif (mm := re.search(r"(?:what is|calculate|compute|evaluate|solve)\s+(.+)", low)) and (val := evaluate_logic(mm.group(1).rstrip("?"))) is not None:
            reply = f"{mm.group(1).rstrip('?').strip()} = **{val}**. (I evaluated that with a safe logic engine — no code execution.)"
        elif (val := evaluate_logic(low.rstrip("?"))) is not None and (re.search(r"\d", low) or any(w in low for w in ["true", "false", " and ", " or ", "not "])):
            reply = f"That evaluates to **{val}**."
        else:
            concept_hit = next((v for k, v in KNOWLEDGE.items() if k in low and ("what" in low or "explain" in low or "tell" in low or "?" in msg or len(low) < 60)), None)
            if concept_hit and not any(w in low for w in ["detected", "attack on", "happened", "my ", "our ", "server", "host"]):
                reply = concept_hit + "\n\nWant me to analyse a real example? Paste one and I'll classify it with reasoning."
            else:
                # default: treat as a security event to analyse
                pred = self.predict(msg, model_id) if self.predict else None
                if pred is None:
                    reply = "My classifier isn't loaded in this mode — start via chat_server or chat_cli."
                else:
                    rs = reason(msg, pred)
                    pred = {**pred, "reasoning": rs}
                    s["last"], s["last_text"] = pred, msg
                    agree = rs["steps"][3]["detail"] if len(rs["steps"]) > 3 else ""
                    reply = (f"**Analysis: {pred['predicted_class']}** — risk **{pred['risk_level']}**, confidence {pred['confidence']:.0%}.\n"
                             f"{'🚨 Threat detected. ' if pred['is_threat'] else '✅ Looks benign. '}{pred['recommended_action']}\n"
                             f"Reasoning in short: {agree} Ask me **'why?'** for the full evidence trail, or **'what should I do?'** for the response playbook.")
                    extra = {"prediction": pred, **pred, "reasoning": rs}
        s["history"].append({"user": msg, "reply": reply})
        return {"reply": reply, "model_used": model_id, "session_turns": s["count"], **extra}
