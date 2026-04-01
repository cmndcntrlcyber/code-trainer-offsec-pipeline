"""
phase1_data_collection/scrapers/offsec_keywords.py

Offensive security keyword dictionaries, domain classification,
and MITRE ATT&CK tactic mapping.
"""
import re
from typing import Dict, List, Set, Tuple

# Domain -> keyword list mapping for offensive security classification
OFFSEC_KEYWORDS: Dict[str, List[str]] = {
    "exploit": [
        "exploit", "exploit-development", "exploitdb", "cve-", "vulnerability",
        "vulnerability-research", "zero-day", "0day", "poc", "proof-of-concept",
        "buffer-overflow", "heap-overflow", "heap-spray", "stack-overflow",
        "use-after-free", "type-confusion", "integer-overflow", "format-string",
        "double-free", "out-of-bounds", "race-condition", "rop-chain", "rop-gadget",
        "rce", "remote-code-execution", "lpe", "local-privilege-escalation",
        "arbitrary-code-execution", "code-execution",
        "sql-injection", "sqli", "xss", "cross-site-scripting", "ssrf",
        "lfi", "rfi", "command-injection", "xxe", "ssti", "idor",
        "insecure-deserialization", "deserialization", "prototype-pollution",
        "path-traversal", "file-inclusion", "template-injection",
        "kernel-exploit", "browser-exploit", "windows-exploit", "linux-exploit",
        "ios-exploit", "android-exploit", "macos-exploit", "chrome-exploit",
        "v8-exploit", "webkit-exploit", "firmware-exploit",
    ],
    "c2": [
        "c2", "c2-framework", "command-and-control", "cnc",
        "rat", "remote-access-trojan", "remote-access-tool",
        "beacon", "implant", "agent", "listener", "stager", "stageless",
        "teamserver", "operator", "handler", "callback", "check-in",
        "payload-delivery", "tasking", "job-queue",
        "cobalt-strike", "metasploit", "empire", "sliver", "havoc",
        "mythic", "covenant", "brute-ratel", "nighthawk", "poshc2",
        "merlin-c2", "deimos", "villain", "pupy", "koadic", "silenttrinity",
        "faction", "nuages", "ares", "trevorC2", "shad0w",
        "dns-c2", "http-c2", "https-c2", "smb-c2", "icmp-c2",
        "websocket-c2", "dns-tunnel", "dns-exfiltration",
        "covert-channel", "data-exfiltration", "exfiltration",
        "domain-fronting", "cdn-fronting", "cloud-fronting",
        "malleable-profile", "redirector", "infrastructure",
        "teamserver-setup", "c2-profile", "traffic-shaping",
    ],
    "recon": [
        "recon", "reconnaissance", "enumeration", "scanner", "discovery",
        "information-gathering", "footprinting", "profiling",
        "osint", "open-source-intelligence", "social-engineering-recon",
        "email-harvesting", "username-enumeration", "people-search",
        "google-dork", "dorking", "search-engine-hacking",
        "port-scan", "port-scanner", "service-enumeration",
        "host-discovery", "network-mapping", "network-scan",
        "arp-scan", "ping-sweep", "traceroute",
        "banner-grabbing", "fingerprint", "fingerprinting",
        "os-detection", "version-detection",
        "subdomain-enumeration", "subdomain-discovery", "subdomain-takeover",
        "dns-enumeration", "dns-recon", "zone-transfer", "dns-brute",
        "web-recon", "directory-bruteforce", "content-discovery",
        "vhost-enumeration", "parameter-discovery", "endpoint-discovery",
        "technology-detection", "wappalyzer", "whatweb",
        "asset-discovery", "attack-surface", "external-recon",
        "cloud-enumeration", "s3-enumeration", "bucket-finder",
        "certificate-transparency", "ct-logs",
        "shodan", "censys", "zoomeye", "fofa", "binaryedge",
        "spyse", "securitytrails", "virustotal",
    ],
    "post_exploit": [
        "post-exploitation", "post-exploit", "post-compromise",
        "lateral-movement", "pivoting", "port-forwarding", "tunneling",
        "socks-proxy", "ssh-tunnel", "chisel", "ligolo",
        "psexec", "wmiexec", "smbexec", "atexec", "dcomexec",
        "winrm", "evil-winrm", "rdp-hijacking",
        "persistence", "backdoor", "rootkit", "bootkit", "webshell",
        "implant-persistence", "registry-persistence", "startup-persistence",
        "scheduled-task", "service-creation", "wmi-persistence",
        "cron-persistence", "ssh-backdoor", "pam-backdoor",
        "golden-ticket-persistence", "skeleton-key",
        "privilege-escalation", "privesc", "lpe",
        "local-privilege-escalation", "vertical-escalation",
        "uac-bypass", "potato-attack", "printspoofer",
        "suid-exploitation", "capabilities-abuse",
        "linux-privesc", "windows-privesc",
        "kernel-privesc", "service-exploitation",
        "loot", "credential-access", "data-collection",
        "screenshot-capture", "keylogger", "clipboard-hijacking",
        "browser-data", "wifi-passwords", "vault-dump",
        "process-injection", "dll-injection", "process-hollowing",
        "thread-hijacking", "apc-injection", "early-bird-injection",
        "token-manipulation", "token-impersonation", "token-theft",
        "named-pipe-impersonation", "handle-duplication",
    ],
    "evasion": [
        "evasion", "defense-evasion", "bypass", "av-bypass",
        "antivirus-bypass", "edr-bypass", "edr-evasion",
        "endpoint-detection", "detection-bypass",
        "amsi", "amsi-bypass", "amsi-patch",
        "etw", "etw-bypass", "etw-patching", "etw-blind",
        "wdac-bypass", "applocker-bypass", "clm-bypass",
        "constrained-language-mode", "script-block-logging",
        "unhooking", "syscall", "direct-syscall", "indirect-syscall",
        "hell-gate", "halo-gate", "tartarus-gate", "syswhispers",
        "ntdll-unhooking", "userland-hooking",
        "obfuscation", "code-obfuscation", "string-obfuscation",
        "string-encryption", "control-flow-obfuscation",
        "call-stack-spoofing", "return-address-spoofing",
        "packer", "crypter", "protector", "metamorphic", "polymorphic",
        "pe-packer", "runtime-packer",
        "anti-analysis", "anti-debugging", "anti-vm", "anti-sandbox",
        "sandbox-evasion", "sandbox-detection", "vm-detection",
        "timing-check", "cpuid-check",
        "timestomping", "log-evasion", "event-log-tampering",
        "indicator-removal", "artifact-cleanup", "track-covering",
        "ppid-spoofing", "argument-spoofing", "command-line-spoofing",
        "module-stomping", "phantom-dll",
        "living-off-the-land", "lolbas", "lolbins", "gtfobins",
        "loldriver", "byovd", "bring-your-own-vulnerable-driver",
        "fileless", "fileless-malware", "in-memory-execution",
        "memory-only", "reflective-execution",
        "code-signing", "signature-evasion", "entropy-reduction",
        "binary-padding", "section-manipulation",
        "opsec", "operational-security", "covert-channel",
        "traffic-blending", "jitter", "sleep-obfuscation",
    ],
    "credential": [
        "credential", "credential-theft", "credential-stuffing",
        "credential-dumping", "credential-harvesting",
        "password", "password-cracking", "password-spraying",
        "password-attack", "password-guessing", "default-password",
        "hash", "hash-cracking", "hashcat", "john-the-ripper",
        "rainbow-table", "hash-extraction",
        "brute-force", "bruteforce", "dictionary-attack", "wordlist",
        "rule-based-attack", "mask-attack", "combinator-attack",
        "mimikatz", "lsass", "lsass-dump", "sam-dump", "secretsdump",
        "dpapi", "windows-vault", "credential-manager",
        "ntds-dit", "ntds-extraction", "volume-shadow-copy",
        "kerberos", "kerberoasting", "as-rep-roasting",
        "golden-ticket", "silver-ticket", "diamond-ticket",
        "sapphire-ticket", "pass-the-ticket",
        "ntlm", "pass-the-hash", "overpass-the-hash",
        "ntlm-relay", "relay-attack", "responder",
        "dcsync", "dcshadow",
        "llmnr", "nbns", "mdns", "wpad", "llmnr-poisoning",
        "man-in-the-middle", "mitm", "arp-spoofing",
        "mfa-bypass", "2fa-bypass", "otp-bypass", "push-fatigue",
        "token-theft", "cookie-theft", "session-hijacking",
        "jwt-attack", "oauth-abuse", "saml-attack",
        "phishing-kit", "evilginx", "evilginx2", "modlishka",
        "gophish", "credential-phishing", "adversary-in-the-middle",
        "keylogger", "keylogging", "input-capture",
        "screen-capture", "clipboard-monitor",
        "spray", "dump", "extract", "harvest",
    ],
    "payload": [
        "payload", "payload-generation", "payload-framework",
        "payload-obfuscation", "payload-encryption",
        "shellcode", "shellcode-loader", "shellcode-runner",
        "shellcode-encoder", "shellcode-generator",
        "position-independent-code", "pic", "egg-hunter",
        "stager", "stageless", "staged-payload",
        "dropper", "downloader", "loader", "cradle",
        "powershell-cradle", "download-cradle",
        "bof", "beacon-object-file", "coff-loader",
        "inline-execute", "object-file",
        "donut", "pe-to-shellcode", "reflective-dll",
        "reflective-loading", "reflective-injection",
        "dll-sideloading", "dll-hijacking", "dll-proxying",
        "binary-patching", "pe-manipulation", "elf-manipulation",
        "section-injection", "code-cave",
        "msfvenom", "venom", "sharpshooter", "unicorn",
        "macro-pack", "evil-clippy", "phishery",
        "hta-payload", "iso-payload", "lnk-payload", "msi-payload",
        "vba-macro", "macro", "office-macro", "xll-payload",
        "html-smuggling", "svg-smuggling",
        "implant", "implant-framework", "agent-development",
        "c2-agent", "beacon-development",
    ],
    "active_directory": [
        "active-directory", "ad-attack", "ad-exploitation", "ad-enumeration",
        "bloodhound", "sharphound", "ldap-enumeration",
        "group-policy-abuse", "gpo-abuse",
        "constrained-delegation", "unconstrained-delegation",
        "rbcd", "resource-based-constrained-delegation",
        "acl-abuse", "dacl-abuse", "writedacl", "genericall",
        "genericwrite", "forcechangepassword",
        "certificate-abuse", "adcs", "certipy", "certify",
        "shadow-credentials", "esc1", "esc2", "esc3", "esc4",
        "esc6", "esc7", "esc8", "esc9", "esc10",
        "domain-trust", "forest-trust", "sid-history",
        "cross-domain", "inter-forest",
        "azure-ad", "entra-id", "azure-exploitation",
        "azurehound", "roadtools", "azure-enumeration",
    ],
    "cloud": [
        "cloud-security", "cloud-pentest", "cloud-exploitation",
        "cloud-enumeration", "multi-cloud",
        "aws-security", "aws-exploitation", "aws-pentest",
        "pacu", "prowler", "scout-suite", "cloudmapper",
        "s3-bucket", "iam-exploitation", "iam-escalation",
        "lambda-exploitation", "ec2-exploitation",
        "metadata-service", "imds", "ssrf-to-rce",
        "azure-security", "azure-exploitation", "azure-pentest",
        "microburst", "stormspotter", "azurite",
        "gcp-security", "gcp-exploitation", "gcp-pentest",
        "kubernetes-security", "k8s-pentest", "container-escape",
        "docker-escape", "docker-security", "pod-escape",
        "container-breakout", "namespace-escape",
        "serverless-security", "lambda-exploitation",
        "function-exploitation",
        "terraform-abuse", "cloudformation-abuse",
        "misconfiguration", "cloud-misconfiguration",
    ],
    "network": [
        "network-attack", "network-security", "packet-crafting",
        "man-in-the-middle", "mitm", "arp-spoofing", "dns-spoofing",
        "dns-poisoning", "bgp-hijacking",
        "smb-attack", "smb-relay", "rpc-attack", "dcom-attack",
        "ldap-attack", "mssql-attack", "mysql-attack",
        "snmp-attack", "ftp-attack", "telnet-attack",
        "wifi-attack", "wireless-security", "evil-twin", "wpa-crack",
        "wpa2-attack", "wps-attack", "karma-attack", "rogue-ap",
        "bluetooth-attack", "ble-attack", "zigbee", "rfid",
        "sniffing", "packet-capture", "traffic-analysis",
        "traffic-interception", "pcap-analysis",
        "vpn-attack", "ssl-stripping", "tls-attack",
        "coap", "mqtt", "modbus", "dnp3",
        "scada", "ics-security", "ot-security", "plc-attack",
    ],
    "malware": [
        "malware", "malware-development", "malware-analysis",
        "malware-research", "malware-sample",
        "reverse-engineering", "reversing", "binary-analysis",
        "static-analysis", "dynamic-analysis",
        "disassembler", "decompiler", "ida", "ghidra", "binary-ninja",
        "x64dbg", "windbg", "gdb",
        "unpacking", "deobfuscation", "config-extraction",
        "string-decryption", "api-hashing", "import-hashing",
        "ransomware", "worm", "trojan", "spyware", "infostealer",
        "botnet", "bot-framework", "banking-trojan",
        "miner", "cryptominer", "adware",
        "yara", "yara-rules", "sigma-rules", "snort-rules",
        "sandbox-analysis", "automated-analysis",
    ],
    "web_attack": [
        "web-security", "web-exploitation", "web-pentest",
        "owasp", "owasp-top-10", "web-vulnerability",
        "burp-suite", "burp-extension", "zap-extension",
        "sqlmap", "nuclei-template",
        "api-security", "api-pentest", "api-fuzzing",
        "graphql-attack", "rest-api-attack", "grpc-attack",
        "fuzzing", "fuzzer", "web-fuzzer", "protocol-fuzzer",
        "mutation-fuzzing", "grammar-fuzzing", "afl", "libfuzzer",
        "cors-misconfiguration", "csp-bypass", "cache-poisoning",
        "request-smuggling", "http-desync", "websocket-attack",
        "race-condition", "time-of-check", "business-logic",
        "waf-bypass", "firewall-bypass", "ips-evasion",
        "filter-bypass", "input-validation-bypass",
        "cms-exploit", "wordpress-exploit", "joomla-exploit",
        "drupal-exploit", "magento-exploit",
    ],
    "mobile_iot": [
        "mobile-security", "mobile-pentest",
        "android-security", "android-exploit", "android-malware",
        "ios-security", "ios-exploit", "ios-jailbreak",
        "apk-analysis", "ipa-analysis", "smali",
        "frida", "objection", "drozer",
        "iot-security", "iot-pentest", "iot-exploit",
        "firmware-analysis", "firmware-extraction", "firmware-emulation",
        "embedded-security", "hardware-hacking",
        "jtag", "uart", "spi", "i2c", "can-bus",
        "zigbee-attack", "zwave-attack", "lora-attack",
    ],
}

# MITRE ATT&CK tactic mapping by offsec domain
MITRE_TACTIC_MAP: Dict[str, List[str]] = {
    "recon": ["TA0043"],                         # Reconnaissance
    "exploit": ["TA0001", "TA0002"],             # Initial Access, Execution
    "c2": ["TA0011"],                            # Command and Control
    "post_exploit": ["TA0004", "TA0005", "TA0006", "TA0008"],  # Priv Esc, Defense Evasion, Cred Access, Lateral Movement
    "evasion": ["TA0005"],                       # Defense Evasion
    "credential": ["TA0006"],                    # Credential Access
    "payload": ["TA0002"],                       # Execution
    "active_directory": ["TA0006", "TA0008"],    # Cred Access, Lateral Movement
    "cloud": ["TA0009"],                         # Collection
    "network": ["TA0001", "TA0008"],             # Initial Access, Lateral Movement
    "malware": ["TA0002", "TA0003"],             # Execution, Persistence
    "web_attack": ["TA0001"],                    # Initial Access
    "mobile_iot": ["TA0001", "TA0002"],          # Initial Access, Execution
}

# CVE pattern
CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE)


def _build_keyword_set() -> Dict[str, Set[str]]:
    """Build lowercased keyword sets for fast lookup."""
    return {
        domain: {kw.lower() for kw in keywords}
        for domain, keywords in OFFSEC_KEYWORDS.items()
    }


_KEYWORD_SETS = _build_keyword_set()


def classify_offsec_domain(
    description: str = "",
    topics: List[str] = None,
    readme_text: str = ""
) -> str:
    """Classify a repository into an offensive security domain.

    Returns the best-matching domain name, or 'general' if no match.
    """
    topics = topics or []
    text = f"{description} {' '.join(topics)} {readme_text}".lower()
    # Also check with hyphens replaced by spaces for broader matching
    text_alt = text.replace("-", " ")

    domain_scores: Dict[str, int] = {}
    for domain, keyword_set in _KEYWORD_SETS.items():
        score = 0
        for kw in keyword_set:
            if kw in text or kw.replace("-", " ") in text_alt:
                score += 1
        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    return "general"


def extract_matched_keywords(
    description: str = "",
    topics: List[str] = None,
    readme_text: str = ""
) -> List[str]:
    """Extract all matched offensive security keywords from text."""
    topics = topics or []
    text = f"{description} {' '.join(topics)} {readme_text}".lower()
    text_alt = text.replace("-", " ")

    matched = []
    for keyword_set in _KEYWORD_SETS.values():
        for kw in keyword_set:
            if kw in text or kw.replace("-", " ") in text_alt:
                matched.append(kw)
    return sorted(set(matched))


def detect_cve_references(text: str) -> List[str]:
    """Extract CVE IDs from text."""
    return sorted(set(CVE_PATTERN.findall(text.upper())))


def get_mitre_tactics(domain: str) -> List[str]:
    """Get MITRE ATT&CK tactic IDs for a domain."""
    return MITRE_TACTIC_MAP.get(domain, [])
