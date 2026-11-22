"""Response test-suite: run v1, v2, v3 and the ENSEMBLE on identical prompts
(structured telemetry, URLs, emails AND free-text paraphrases) and report
exactly what each model says. Writes artifacts_v3/response_test.json + .md
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.predict import ZitranDetector
from zitran.ensemble import ZitranEnsemble

CASES = [
    ("employee opened attachment and pc started encrypting files asking crypto payment shadow copies gone", "ransomware"),
    ("someone tried thousands of passwords against vpn account of finance manager", "credential_attack"),
    ("fake email pretending microsoft asking to re-enter login password on strange site", "phishing"),
    ("url https://secure-login-verify.top/account/update", "phishing"),
    ("normal user completed mfa login and opened dashboard without issues", "benign"),
    ("massive traffic flood from thousands of bots website down denial of service", "intrusion_dos"),
    ("outgoing traffic sending gigabytes of database to unknown foreign server at night", "exfiltration"),
    ("unknown program keeps contacting random generated domains every minute infected machine", "command_control"),
    ("input field contains union select query trying to dump user passwords table", "exploit_injection"),
    ("attacker probing all open ports and services of our firewall from external address", "recon_scan"),
    ("network flow dur zero proto tcp svc http flag SF sbytes b6 dbytes b7 land zero count b4 srvcount b4", "benign"),
    ("email subject: urgent verify your bank account click link enter card details immediately", "phishing"),
    ("scheduled backup completed successfully files stored status ok no threats found", "benign"),
    ("trojan binary quarantined after injecting code into system process memory hidden", "malware"),
]

def run():
    systems = {}
    for name, d in [("v1", "artifacts"), ("v2", "artifacts_v2"), ("v3", "artifacts_v3")]:
        if os.path.isdir(d):
            try: systems[name] = ZitranDetector(d)
            except Exception as e: print(f"{name} unavailable: {e}")
    ens = ZitranEnsemble(tuple(f"artifacts{('_v'+k[-1]) if k!='v1' else ''}" for k in systems))
    rows = []
    for text, expected in CASES:
        row = {"input": text, "expected": expected, "responses": {}}
        for name, s in systems.items():
            r = s.predict(text)
            row["responses"][name] = {"said": r["predicted_class"], "confidence": r["confidence"],
                                      "risk": r["risk_level"], "correct": r["predicted_class"] == expected}
        r = ens.predict(text)
        row["responses"]["ENSEMBLE"] = {"said": r["predicted_class"], "confidence": r["confidence"],
                                        "risk": r["risk_level"], "correct": r["predicted_class"] == expected,
                                        "votes": r["model_votes"]}
        row["threat_flag_correct"] = all((v["said"] != "benign") == (expected != "benign") for v in row["responses"].values())
        rows.append(row)
    summary = {name: round(sum(r["responses"][name]["correct"] for r in rows if name in r["responses"]) /
                           max(1, sum(1 for r in rows if name in r["responses"])), 3)
               for name in ["v1", "v2", "v3", "ENSEMBLE"]}
    out = {"summary_accuracy": summary, "cases": rows}
    os.makedirs("artifacts_v3", exist_ok=True)
    with open("artifacts_v3/response_test.json", "w") as f: json.dump(out, f, indent=2)
    lines = ["# ZITRAN response test — what each model says", "",
             "| Model | Exact-class accuracy on 14 mixed prompts |", "|---|---|"]
    for k, v in summary.items(): lines.append(f"| {k} | {v:.1%} |")
    lines += ["", "| # | Prompt (truncated) | Expected | v1 says | v2 says | v3 says | ENSEMBLE says |",
              "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        g = lambda n: (f"{r['responses'][n]['said']} ({r['responses'][n]['confidence']:.2f})" if n in r["responses"] else "—")
        lines.append(f"| {i} | {r['input'][:52]}… | {r['expected']} | {g('v1')} | {g('v2')} | {g('v3')} | **{g('ENSEMBLE')}** |")
    with open("artifacts_v3/response_test.md", "w") as f: f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"exp={r['expected']:20s} ens={r['responses']['ENSEMBLE']['said']:20s} "
              f"v1={r['responses'].get('v1',{}).get('said','-'):20s} v2={r['responses'].get('v2',{}).get('said','-'):20s} v3={r['responses'].get('v3',{}).get('said','-')}")
    return out

if __name__ == "__main__":
    run()
