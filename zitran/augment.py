"""Paraphrase augmentation for free-text security descriptions.
Teaches invariance to wording (the v2 weakness): synonym swaps, clause
reordering, filler changes — applied only to natural-language synthetic text.
"""
import random, re

SYNONYMS = {
    "attack": ["attack", "attempt", "intrusion attempt", "assault"],
    "detected": ["detected", "observed", "spotted", "flagged", "identified"],
    "user": ["user", "employee", "account holder", "staff member"],
    "urgent": ["urgent", "immediate", "critical", "time sensitive"],
    "password": ["password", "credentials", "login details", "passphrase"],
    "email": ["email", "message", "mail", "phishing message"],
    "files": ["files", "documents", "data files", "records"],
    "host": ["host", "machine", "endpoint", "workstation"],
    "server": ["server", "machine", "system", "box"],
    "blocked": ["blocked", "stopped", "prevented", "denied"],
    "unknown": ["unknown", "unrecognised", "suspicious", "unfamiliar"],
    "large": ["large", "massive", "huge", "substantial"],
    "ransom": ["ransom", "payment demand", "extortion demand"],
    "encrypted": ["encrypted", "locked", "scrambled", "encoded"],
    "login": ["login", "sign in", "authentication", "logon"],
    "failed": ["failed", "unsuccessful", "rejected", "denied"],
}
FILLERS = ["", "please investigate", "analyst note attached", "ticket raised automatically",
           "reported by monitoring system", "requires triage", "seen in the last hour", "flagged for review"]

def paraphrase(text, rng):
    if text.startswith(("url ", "network flow", "flow ")):
        return text  # structured telemetry: don't corrupt format
    out = text
    for word, alts in SYNONYMS.items():
        if rng.random() < 0.5:
            out = re.sub(rf"\b{word}\b", rng.choice(alts), out, count=1)
    if rng.random() < 0.3 and "," in out:
        a, b = out.split(",", 1)
        out = (b.strip() + " , " + a.strip()) if rng.random() < 0.5 else out
    f = rng.choice(FILLERS)
    return (out + " " + f).strip()
