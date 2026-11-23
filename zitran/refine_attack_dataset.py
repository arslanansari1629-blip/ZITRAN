"""Refine the user-supplied Attack_Dataset.csv into train-ready data.

Cleaning: whitespace/unicode normalisation, empty-field drops, exact dedup on
(normalised title + scenario). Labelling: ordered keyword rules over
(Attack Type, Title, Tags, Category) -> ZITRAN 10-class taxonomy; rows that
match nothing (incl. pure DFIR/blue-team defensive topics) are dropped and
counted. Output rows keep MITRE technique, detection method and solution so
predictions can be enriched with them later.
"""
import csv, json, os, re, sys, unicodedata
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.data import LABELS

# Ordered: most specific first. (label, keywords matched on attack-type/title/tags/category)
RULES = [
    ("ransomware", ["ransomware", "ransom", "crypto locker", "cryptolocker", "file encryption attack", "double extortion"]),
    ("phishing", ["phishing", "spear phish", "smishing", "vishing", "whaling", "phish", "social engineering", "pretexting", "baiting", "fake login", "credential harvest", "spam mail", "malicious email"]),
    ("command_control", ["command and control", "c2", "botnet", "beaconing", "covert channel", "lateral movement", "persistence mechanism", "remote access trojan", " rat "]),
    ("exfiltration", ["exfiltrat", "data theft", "data leak", "leakage", "insider threat", "insider-enabled", "data breach", "stealing data", "physical theft", "information disclosure", "privacy risk"]),
    ("credential_attack", ["brute force", "credential stuffing", "password spray", "kerberoast", "pass the hash", "credential", "password crack", "dictionary attack", "mfa fatigue", "password reuse", "default password", "login attempt"]),
    ("intrusion_dos", ["denial of service", "ddos", "dos attack", "slowloris", "amplification", "flooding", "flood attack", "jamming", "resource exhaustion", "denial-of-service"]),
    ("recon_scan", ["reconnaissance", "recon", "port scan", "scanning", "enumeration", "fingerprint", "discovery", "sniffing", "eavesdrop", "traffic analysis", "osint", "footprinting", "vulnerability scan"]),
    ("exploit_injection", ["injection", "xss", "cross-site", "csrf", "xxe", "ssrf", "sql", "remote code", "rce", "privilege escalation", "buffer overflow", "traversal", "deserial", "exploit", "zero-day", "zero day", "bypass", "hijack", "spoof", "tamper", "man-in-the-middle", "mitm", "firmware", "supply chain", "dependency confusion", "reentrancy", "misconfiguration", "vulnerab"]),
    ("malware", ["malware", "trojan", "virus", "worm", "rootkit", "spyware", "keylog", "backdoor", "malicious librar", "malicious package", "stealer", "loader", "payload deliver"]),
]

def clean(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip().strip(",")
    return s

def map_label(row):
    hay = " ".join([row["attack_type"], row["title"], row["tags"], row["category"]]).lower()
    for label, keys in RULES:
        if any(k in hay for k in keys):
            return label
    return None

def main(src="data/Attack_Dataset.csv", out="data/attack_dataset_refined.jsonl"):
    stats = Counter(); seen = set(); rows = []
    with open(src, newline="", encoding="utf-8", errors="ignore") as f:
        for raw in csv.DictReader(f):
            stats["raw"] += 1
            r = {k.strip().lower().replace(" ", "_"): clean(v) for k, v in raw.items() if k}
            r = {"title": r.get("title", ""), "category": r.get("category", ""), "attack_type": r.get("attack_type", ""),
                 "scenario": r.get("scenario_description", ""), "tools": r.get("tools_used", ""),
                 "target": r.get("target_type", ""), "vulnerability": r.get("vulnerability", ""),
                 "mitre": r.get("mitre_technique", ""), "impact": r.get("impact", ""),
                 "detection": r.get("detection_method", ""), "solution": r.get("solution", ""),
                 "tags": r.get("tags", ""), "source": r.get("source", "")}
            if len(r["scenario"]) < 40 or not r["title"]:
                stats["dropped_short_or_empty"] += 1; continue
            key = re.sub(r"\W+", " ", (r["title"] + " " + r["scenario"][:200]).lower()).strip()
            if key in seen:
                stats["dropped_duplicate"] += 1; continue
            seen.add(key)
            label = map_label(r)
            if label is None:
                stats["dropped_unmapped_or_defensive"] += 1; continue
            stats[f"label_{label}"] += 1
            text = f"{r['title']} . category {r['category']} . attack {r['attack_type']} . {r['scenario']} target {r['target']} . vulnerability {r['vulnerability']}"
            event = f"security event {r['title']} detected by {r['detection']} impact {r['impact']} mitre {r['mitre']}"
            rows.append({"id": len(rows), "text": text, "event_text": event, "label": label,
                         "attack_type_raw": r["attack_type"], "category": r["category"], "mitre": r["mitre"],
                         "detection": r["detection"], "solution": r["solution"], "impact": r["impact"], "title": r["title"]})
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["refined_total"] = len(rows)
    with open("data/attack_dataset_stats.json", "w") as f: json.dump(dict(stats), f, indent=2)
    print(json.dumps(dict(stats), indent=2))
    print("label distribution:", {k: stats[f"label_{k}"] for k in LABELS if stats[f"label_{k}"]})

if __name__ == "__main__":
    main(*sys.argv[1:])
