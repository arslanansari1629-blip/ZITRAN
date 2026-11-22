"""Convert real public datasets (data/raw/) + synthetic top-up into one unified
(text, label, source) corpus for ZITRAN v2. Every mapping is documented here.

Label mapping (defensive taxonomy, zitran.data.LABELS):
  NSL-KDD normal -> benign | DoS (neptune smurf pod teardrop land back apache2
    processtable mailbomb udpstorm) -> intrusion_dos | Probe (satan ipsweep
    nmap portsweep mscan saint) -> recon_scan | guess_passwd snmpguess xlock
    xsnoop snmpgetattack -> credential_attack | buffer_overflow loadmodule perl
    phf imap ftp_write multihop warezmaster warezclient named sendmail ps xterm
    sqlattack -> exploit_injection | rootkit worm -> malware | httptunnel -> command_control
  CTU-13 attack (botnet) -> command_control, normal -> benign
  Enron ham -> benign, spam -> phishing (spam/phishing email text)
  JPCERT phishing URLs -> phishing | malicious-URL corpus: malicious -> malware, benign -> benign
  Synthetic (zitran.data) only tops up classes with little/no real data.
Numeric flow features are log-bucketised (zero/low/medium/high/veryhigh) so the
token model learns magnitudes, not memorised numbers.
"""
import csv, glob, os, random, re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.data import LABELS, generate

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
KDD_FEATURES = ["duration","protocol","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
DOS = {"neptune","smurf","pod","teardrop","land","back","apache2","processtable","mailbomb","udpstorm"}
PROBE = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
CRED = {"guess_passwd","snmpguess","xlock","xsnoop","snmpgetattack"}
EXPLOIT = {"buffer_overflow","loadmodule","perl","phf","imap","ftp_write","multihop","warezmaster","warezclient","named","sendmail","ps","xterm","sqlattack"}
MALW = {"rootkit","worm"}

def kdd_label(a):
    a = a.strip().lower()
    if a == "normal": return "benign"
    if a in DOS: return "intrusion_dos"
    if a in PROBE: return "recon_scan"
    if a in CRED: return "credential_attack"
    if a in EXPLOIT: return "exploit_injection"
    if a in MALW: return "malware"
    if a == "httptunnel": return "command_control"
    return None

def bucket(v):
    """Fine-grained magnitude token: rates (0-1) in 10 bins, counts in log bins."""
    try: v = float(v)
    except Exception: return "zero"
    if v == 0: return "zero"
    if 0 < v <= 1: return f"r{int(v*10)}"
    for i, t in enumerate([1, 2, 5, 10, 50, 100, 500, 1000, 5000, 20000, 100000, 1000000]):
        if v <= t: return f"b{i}"
    return "b12"

KDD_SHORT = ["dur","proto","svc","flag","sbytes","dbytes","land","wfrag","urg","hot","failogin","loggedin","ncomp","rootshell","suatt","nroot","nfile","nshell","naccess","noutcmd","hostlogin","guestlogin","count","srvcount","serr","srvserr","rerr","srvrerr","samesrv","diffsrv","srvdiffhost","dhostcount","dhostsrv","dhostsamesrv","dhostdiffsrv","dhostsameport","dhostsrvdiff","dhostserr","dhostsrvserr","dhostrerr","dhostsrvrerr"]
def kdd_text(row):
    parts = ["network flow"]
    for name, val in zip(KDD_SHORT, row):
        if name in ("proto","svc","flag"):
            parts.append(f"{name} {val}")
        else:
            parts.append(f"{name} {bucket(val)}")
    return " ".join(parts)

def load_kdd(path, cap_per_class, rng):
    out, counts = [], {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 42: continue
            lab = kdd_label(row[-1])
            if lab is None: continue
            if counts.get(lab, 0) >= cap_per_class: continue
            counts[lab] = counts.get(lab, 0) + 1
            out.append((kdd_text(row), LABELS.index(lab), "nsl-kdd"))
    return out

def load_ctu(cap_per_class, rng):
    out = []
    specs = [("CTU13-CSV-Dataset/CTU13_Attack_Traffic.csv", "command_control"),
             ("CTU13-CSV-Dataset/CTU13_Normal_Traffic.csv", "benign")]
    for rel, lab in specs:
        path = os.path.join(RAW, rel); n = 0
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            keep = [k for k in r.fieldnames if k and k.strip()][:36]
            rows = list(r); rng.shuffle(rows)
            for row in rows:
                if n >= cap_per_class: break
                txt = "flow " + " ".join(f"f{i} {bucket(row.get(k,0))}" for i, k in enumerate(keep))
                out.append((txt, LABELS.index(lab), "ctu13")); n += 1
    return out

def clean_email(t):
    t = re.sub(r"\s+", " ", t.lower())
    return "email " + t[:600]

def load_emails(cap_per_class, rng):
    out, counts = [], {}
    path = os.path.join(RAW, "SpamEmail/spam_ham_dataset.csv")
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f)); rng.shuffle(rows)
        for row in rows:
            lab = "phishing" if row["label"] == "spam" else "benign"
            if counts.get(lab, 0) >= cap_per_class: continue
            if len(row["text"].strip()) < 20: continue
            counts[lab] = counts.get(lab, 0) + 1
            out.append((clean_email(row["text"]), LABELS.index(lab), "enron-spam"))
    return out

def load_urls(cap_phish, cap_mal, cap_benign, rng):
    out = []
    # JPCERT real phishing URLs
    files = sorted(glob.glob(os.path.join(RAW, "phishurl-list", "*", "*.csv")))
    urls = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                if row.get("URL"): urls.append(row["URL"].strip())
    rng.shuffle(urls)
    for u in urls[:cap_phish]:
        out.append((f"url {u}", LABELS.index("phishing"), "jpcert-phishurl"))
    # malicious / benign URL corpus
    path = os.path.join(RAW, "urldata/data/dataNN.csv")
    mal, ben = [], []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            (mal if row["label"] == "1" else ben).append(row["url"])
    rng.shuffle(mal); rng.shuffle(ben)
    for u in mal[:cap_mal]: out.append((f"url {u}", LABELS.index("malware"), "malicious-urls"))
    for u in ben[:cap_benign]: out.append((f"url https://{u}" if "://" not in u else f"url {u}", LABELS.index("benign"), "malicious-urls"))
    return out

def build_mixed(cap_real=2600, synthetic_per_class=900, seed=123):
    """Unified corpus: real data first, synthetic tops up rare classes."""
    rng = random.Random(seed)
    data = []
    data += load_kdd(os.path.join(RAW, "Network-Intrusion-Detection/KDDTrain+_2.csv"), cap_real, rng)
    data += load_ctu(cap_real, rng)
    data += load_emails(cap_real, rng)
    data += load_urls(cap_real, cap_real, cap_real, rng)
    # synthetic top-up: bring every class to at least synthetic_per_class extra,
    # with full synthetic volume for classes lacking real data
    synth = generate(1600, seed=99)
    from collections import Counter
    real_counts = Counter(l for _, l, _ in data)
    synth_by = {}
    for t, l in synth: synth_by.setdefault(l, []).append(t)
    for lab_idx, lab in enumerate(LABELS):
        need = max(0, 2400 - real_counts.get(lab_idx, 0))
        for t in synth_by.get(lab_idx, [])[:need]:
            data.append((t, lab_idx, "synthetic"))
    # dedup + shuffle
    seen, uniq = set(), []
    for t, l, s in data:
        k = (t, l)
        if k in seen: continue
        seen.add(k); uniq.append((t, l, s))
    rng.shuffle(uniq)
    return uniq

UNSW_MAP = {"Normal": "benign", "DoS": "intrusion_dos", "Reconnaissance": "recon_scan",
            "Exploits": "exploit_injection", "Generic": "exploit_injection", "Fuzzers": "exploit_injection",
            "Shellcode": "exploit_injection", "Backdoor": "command_control", "Worms": "malware", "Analysis": "recon_scan"}
UNSW_KEEP = ["dur","proto","service","state","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload","sinpkt","dinpkt","smean","dmean","response_body_len","ct_srv_src","ct_dst_ltm","ct_src_dport_ltm","ct_dst_src_ltm","is_ftp_login","ct_flw_http_mthd","is_sm_ips_ports"]

def load_unsw(cap_per_class=6000, seed=42):
    import random as _r
    rng = _r.Random(seed); out, counts = [], {}
    path = os.path.join(RAW, "UNSW-NB15/UNSW_NB15.csv")
    if not os.path.exists(path): path = os.path.join(RAW, "UNSW-NB15/datasets/UNSW_NB15.csv")
    if not os.path.exists(path): return out
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        rows = list(csv.DictReader(f)); rng.shuffle(rows)
        for row in rows:
            lab = UNSW_MAP.get((row.get("attack_cat") or "").strip())
            if lab is None or counts.get(lab, 0) >= cap_per_class: continue
            counts[lab] = counts.get(lab, 0) + 1
            parts = ["unsw flow"]
            for k in UNSW_KEEP:
                v = row.get(k, "")
                parts.append(f"{k} {v}" if k in ("proto", "service", "state") else f"{k} {bucket(v)}")
            out.append((" ".join(parts), LABELS.index(lab), "unsw-nb15"))
    return out

def load_official_test(cap_per_class=4000, seed=7):
    """NSL-KDD official KDDTest+ set — contains attack types NEVER in training."""
    rng = random.Random(seed)
    return load_kdd(os.path.join(RAW, "Network-Intrusion-Detection/KDDTest+_2.csv"), cap_per_class, rng)
