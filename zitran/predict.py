"""ZITRAN inference — defensive threat classification (no offensive generation)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.tokenizer import BPETokenizer
from zitran.model import ZITRAN, softmax
from zitran.data import LABELS

_RISK = {"benign": "NONE", "recon_scan": "MEDIUM", "phishing": "HIGH", "credential_attack": "HIGH",
         "exploit_injection": "CRITICAL", "malware": "CRITICAL", "intrusion_dos": "HIGH",
         "exfiltration": "CRITICAL", "ransomware": "CRITICAL", "command_control": "CRITICAL"}
_ACTION = {"benign": "Allow; log normally.", "malware": "Isolate endpoint, quarantine file, run full AV/EDR scan.",
 "phishing": "Block sender/domain, warn user, reset exposed credentials.", "intrusion_dos": "Enable rate-limiting/DDoS mitigation, block offending IPs.",
 "exploit_injection": "Block request/IP at WAF, patch input validation, audit logs for compromise.",
 "credential_attack": "Force password reset, enforce MFA, block source IP, review login history.",
 "exfiltration": "Block transfer, preserve evidence, invoke incident response / DLP review.",
 "recon_scan": "Monitor source, ensure patching, block if scanning persists.",
 "ransomware": "Isolate host NOW, disable shares, restore from offline backups, call IR.",
 "command_control": "Isolate host, block C2 domain/IP at DNS+firewall, hunt for persistence."}

class ZitranDetector:
    def __init__(self, artifacts="artifacts"):
        self.tok = BPETokenizer.load(f"{artifacts}/tokenizer.json")
        path = f"{artifacts}/zitran_model.npz"
        if not os.path.exists(path):
            path = f"{artifacts}/zitran_v2_model.npz"
        self.model = ZITRAN.load(path)
    def predict(self, text):
        x = np.array([self.tok.encode(text)])
        logits, bl = self.model.forward(x)
        probs = softmax(logits)[0]; bprobs = softmax(bl)[0]
        i = int(probs.argmax()); label = LABELS[i]
        return {"input": text, "predicted_class": label, "confidence": round(float(probs[i]), 4),
                "is_threat": bool(label != "benign"), "threat_probability": round(float(bprobs[1]), 4),
                "risk_level": _RISK[label], "recommended_action": _ACTION[label],
                "top3": [{"class": LABELS[j], "prob": round(float(probs[j]), 4)} for j in probs.argsort()[::-1][:3]]}

if __name__ == "__main__":
    d = ZitranDetector(sys.argv[1] if len(sys.argv) > 2 else "artifacts")
    samples = sys.argv[2:] or [
        "user alice logged in successfully from 10.0.0.5 via mfa",
        "email urgent verify your account at secure-login-verify.top enter password",
        "mass file encryption 50000 files renamed .locked ransom note bitcoin",
        "GET /search?q=' OR '1'='1 UNION SELECT password FROM users sql injection",
        "periodic beaconing every 60s to evil-c2.xyz infected host command and control"]
    for s in samples: print(json.dumps(d.predict(s), indent=2))
