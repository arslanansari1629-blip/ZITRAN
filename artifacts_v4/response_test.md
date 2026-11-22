# ZITRAN response test — what each model says

| Model | Exact-class accuracy on 14 mixed prompts |
|---|---|
| v1 | 64.3% |
| v2 | 35.7% |
| v3 | 71.4% |
| v4 | 85.7% |
| ENSEMBLE | 85.7% |

| # | Prompt (truncated) | Expected | v1 says | v2 says | v3 says | v4 says | ENSEMBLE says |
|---|---|---|---|---|---|---|---|
| 1 | employee opened attachment and pc started encrypting… | ransomware | intrusion_dos (0.30) | exfiltration (0.36) | exfiltration (0.38) | intrusion_dos (0.46) | **ransomware (0.23)** |
| 2 | someone tried thousands of passwords against vpn acc… | credential_attack | credential_attack (0.98) | credential_attack (0.45) | credential_attack (0.91) | credential_attack (0.76) | **credential_attack (0.78)** |
| 3 | fake email pretending microsoft asking to re-enter l… | phishing | phishing (0.96) | credential_attack (0.55) | phishing (0.98) | phishing (0.97) | **phishing (0.73)** |
| 4 | url https://secure-login-verify.top/account/update… | phishing | benign (1.00) | phishing (0.98) | phishing (0.99) | phishing (0.99) | **phishing (0.73)** |
| 5 | normal user completed mfa login and opened dashboard… | benign | benign (0.90) | credential_attack (0.70) | credential_attack (0.72) | intrusion_dos (0.46) | **credential_attack (0.37)** |
| 6 | massive traffic flood from thousands of bots website… | intrusion_dos | intrusion_dos (0.97) | exfiltration (0.49) | intrusion_dos (0.99) | intrusion_dos (0.98) | **intrusion_dos (0.74)** |
| 7 | outgoing traffic sending gigabytes of database to un… | exfiltration | malware (0.39) | exfiltration (0.83) | exfiltration (0.94) | exfiltration (0.88) | **exfiltration (0.66)** |
| 8 | unknown program keeps contacting random generated do… | command_control | command_control (0.92) | exploit_injection (0.32) | command_control (0.81) | command_control (0.68) | **command_control (0.61)** |
| 9 | input field contains union select query trying to du… | exploit_injection | exfiltration (0.58) | exfiltration (0.73) | exfiltration (0.92) | exploit_injection (0.43) | **exfiltration (0.62)** |
| 10 | attacker probing all open ports and services of our … | recon_scan | recon_scan (0.92) | exfiltration (0.46) | exfiltration (0.47) | recon_scan (0.88) | **recon_scan (0.46)** |
| 11 | network flow dur zero proto tcp svc http flag SF sby… | benign | credential_attack (0.52) | benign (0.98) | benign (0.81) | benign (0.95) | **benign (0.77)** |
| 12 | email subject: urgent verify your bank account click… | phishing | phishing (1.00) | phishing (0.97) | phishing (1.00) | phishing (0.99) | **phishing (0.99)** |
| 13 | scheduled backup completed successfully files stored… | benign | benign (1.00) | credential_attack (0.40) | benign (1.00) | benign (0.98) | **benign (0.75)** |
| 14 | trojan binary quarantined after injecting code into … | malware | malware (1.00) | credential_attack (0.43) | malware (0.99) | malware (0.59) | **malware (0.65)** |
