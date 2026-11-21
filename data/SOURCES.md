# ZITRAN v2 — real dataset provenance (all free, public, via GitHub)

| Source | Repo | Content used | Mapping |
|---|---|---|---|
| NSL-KDD | CynthiaKoopman/Network-Intrusion-Detection | KDDTrain+ (125,973 flows) for training; **KDDTest+ (22,544) held out as official test** — contains attack types absent from training (mscan, apache2, saint, sqlattack…) | attack-name → benign / intrusion_dos / recon_scan / credential_attack / exploit_injection / malware / command_control (see `zitran/real_data.py`) |
| CTU-13 | imfaisalmalik/CTU13-CSV-Dataset | Botnet vs normal flows (CICFlowMeter features, Czech Technical University 2011 captures) | botnet → command_control, normal → benign |
| Enron spam/ham | MSc-AI/SpamEmail | 5,171 emails (3,672 ham / 1,499 spam) | ham → benign, spam → phishing |
| JPCERT/CC phishurl-list | JPCERTCC/phishurl-list | ~290k real phishing URLs (2019–2026), sampled | → phishing |
| Malicious URL corpus | savan77/Malicious-URL-Detection-using-Machine-Learning | dataNN.csv ~650k URLs from public feeds, sampled | malicious → malware, benign → benign |
| Synthetic top-up | `zitran/data.py` (this repo) | Only for classes with little/no public data (ransomware, exfiltration, etc.) | all 10 classes, capped so real data dominates |

Reproduce downloads: `python3 -m zitran.fetch_data` → `data/raw/` (git-ignored, ~90 MB).
Unified corpus: `data/dataset_v2.jsonl` (text, label, source per row).
Note: respect each upstream dataset's original terms/licences for redistribution.
