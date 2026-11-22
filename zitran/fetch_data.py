"""Download the real public datasets used by ZITRAN v2 (all from GitHub, free).
Re-runnable: skips repos already present in data/raw/. Raw data is git-ignored;
provenance is recorded in data/SOURCES.md by zitran/real_data.py.
"""
import os, subprocess, sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SOURCES = [
    ("Network-Intrusion-Detection", "https://github.com/CynthiaKoopman/Network-Intrusion-Detection.git", None,
     "NSL-KDD train/test network intrusion flows"),
    ("CTU13-CSV-Dataset", "https://github.com/imfaisalmalik/CTU13-CSV-Dataset.git", None,
     "CTU-13 botnet (C2) vs normal flows, CICFlowMeter features"),
    ("SpamEmail", "https://github.com/MSc-AI/SpamEmail.git", None,
     "Enron spam/ham email corpus (spam_ham_dataset.csv)"),
    ("phishurl-list", "https://github.com/JPCERTCC/phishurl-list.git", None,
     "JPCERT/CC real phishing URLs 2019-2026"),
    ("urldata", "https://github.com/savan77/Malicious-URL-Detection-using-Machine-Learning.git", "data",
     "Malicious vs benign URL corpus (dataNN.csv, ~650k URLs collected from public feeds)"),
    ("UNSW-NB15", "https://github.com/notsodubeyous/IoT-Network-Intrusion-Detection-System-UNSW-NB15.git", "datasets",
     "UNSW-NB15 175k real network flows with attack categories (UNSW_NB15.csv)"),
]

def fetch():
    os.makedirs(ROOT, exist_ok=True)
    for name, url, sparse, desc in SOURCES:
        dest = os.path.join(ROOT, name)
        if os.path.exists(dest):
            print(f"[skip] {name} already present — {desc}")
            continue
        print(f"[clone] {name} — {desc}")
        cmd = ["git", "clone", "-q", "--depth", "1"]
        if sparse:
            cmd += ["--filter=blob:none", "--sparse"]
        cmd += [url, dest]
        subprocess.run(cmd, check=True)
        if sparse:
            subprocess.run(["git", "-C", dest, "sparse-checkout", "set", sparse], check=True)
    print("Done. Raw datasets in data/raw/ (not committed).")

if __name__ == "__main__":
    fetch()
