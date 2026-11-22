import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zitran.predict import ZitranDetector

if __name__ == "__main__":
    d = ZitranDetector("artifacts")
    for s in [
        "user bob logged in successfully via mfa backup completed status success",
        "spear phishing email impersonating ceo request wire transfer urgent click attachment",
        "vssadmin delete shadows mass file encryption files renamed .locked ransom note bitcoin",
        "dns tunneling encoded data subdomain queries to evil-c2.xyz exfiltrating files",
        "port scan 65000 ports probed nmap syn scan reconnaissance from 185.10.2.3",
    ]:
        print(json.dumps(d.predict(s), indent=2))
