"""
Generate synthetic ITSM incident dataset from HuggingFace source data.
Maps 6StringNinja/synthetic-servicenow-incidents fields to a richer schema
and fills missing columns with realistic synthetic values.
"""

import random
import datetime
import pandas as pd
from datasets import load_dataset

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ── constants ────────────────────────────────────────────────────────────────
CHANNELS = ["portal", "email", "chat", "phone", "monitoring"]
BUSINESS_SERVICES = ["VPN", "Email", "Payroll", "CRM", "HRMS", "ERP",
                     "Website", "FileShare", "Database"]
CONFIG_ITEMS = ["Laptop", "Desktop", "Server", "VM", "Firewall",
                "Router", "Switch", "Application", "Database"]
RESOLVER_GROUPS = ["Network Team", "Service Desk", "App Support",
                   "Security Ops", "DBA Team", "Cloud Ops", "DevOps Team"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
STATUS_WEIGHTS = [0.10, 0.15, 0.35, 0.40]          # favour Resolved/Closed

SLA_MAP = {"Critical": 1, "High": 4, "Medium": 24, "Low": 72}

# mapping from dataset numeric urgency / impact (1‑3) to label
LEVEL_MAP = {1: "High", 2: "Medium", 3: "Low"}     # ServiceNow convention

# priority matrix  (impact, urgency) → priority string
PRIORITY_MATRIX = {
    ("High",   "High"):   "Critical",
    ("High",   "Medium"): "High",
    ("High",   "Low"):    "Medium",
    ("Medium", "High"):   "High",
    ("Medium", "Medium"): "Medium",
    ("Medium", "Low"):    "Low",
    ("Low",    "High"):   "Medium",
    ("Low",    "Medium"): "Low",
    ("Low",    "Low"):    "Low",
}

# resolver‑group hints based on category
CATEGORY_RESOLVER = {
    "Network":  "Network Team",
    "Hardware": "Service Desk",
    "Software": "App Support",
    "Access":   "Security Ops",
    "Email":    "App Support",
    "Database": "DBA Team",
    "Cloud":    "Cloud Ops",
    "Security": "Security Ops",
    "DevOps":   "DevOps Team",
}

# root‑cause templates per category
ROOT_CAUSE_TEMPLATES = {
    "Network":  ["Misconfigured firewall rule", "Faulty network cable",
                 "DNS resolution failure", "DHCP scope exhaustion",
                 "Routing table corruption", "Switch port flapping"],
    "Hardware": ["Faulty RAM module", "Hard drive failure",
                 "Overheating due to fan failure", "Power supply unit failure",
                 "Defective docking station", "Monitor hardware defect"],
    "Software": ["Application bug in latest release",
                 "Incompatible software update", "Corrupted installation files",
                 "Missing runtime dependency", "Memory leak in service process",
                 "Unhandled exception in backend module"],
    "Access":   ["Expired user credentials", "Incorrect group policy assignment",
                 "MFA token sync failure", "Orphaned AD account",
                 "Permission inheritance misconfiguration",
                 "LDAP replication delay"],
    "Email":    ["Exchange mailbox quota exceeded",
                 "Transport rule blocking delivery",
                 "Autodiscover DNS misconfiguration",
                 "Mail queue backlog on relay server",
                 "Spam filter false positive"],
    "Database": ["Table lock contention", "Index fragmentation",
                 "Connection pool exhaustion", "Replication lag",
                 "Deadlock in stored procedure", "Corrupted transaction log"],
    "Cloud":    ["Auto‑scaling policy misconfiguration",
                 "Expired service principal credentials",
                 "Region capacity throttling", "Blob storage IOPS limit hit",
                 "Misconfigured NSG rule"],
    "Security": ["Malware detected on endpoint",
                 "Brute‑force login attempt detected",
                 "Certificate expiry on web application",
                 "Unauthorized access attempt",
                 "DLP policy violation"],
    "DevOps":   ["CI/CD pipeline timeout", "Container image pull failure",
                 "Helm chart version mismatch",
                 "Terraform state lock not released",
                 "Secret rotation failure in vault"],
}

# short / poorly‑written description templates (≥5 % of tickets)
POOR_DESCRIPTIONS = [
    "not working", "broken again", "pls fix asap", "same issue",
    "help needed", "cant login", "error on screen", "its down",
    "something wrong", "need access", "ticket for the thing",
    "plz check", "urgent!!!", "system error",
    "doesnt work since morning", "laptop issue",
]

# Category-specific title templates for better differentiation
CATEGORY_TITLE_TEMPLATES = {
    "Network": [
        "VPN connection failed",
        "Router not responding",
        "Network connectivity issue",
        "DNS resolution failure",
        "WiFi disconnecting frequently",
        "Firewall blocking access",
        "Switch port not working",
        "Network drive not accessible",
        "DHCP lease not renewing",
        "Cannot ping server",
    ],
    "Hardware": [
        "Laptop screen flickering",
        "Desktop won't power on",
        "Monitor display issue",
        "Keyboard not responding",
        "Mouse not working",
        "Printer paper jam",
        "Hard drive failure",
        "Docking station malfunction",
        "Laptop battery not charging",
        "Overheating computer",
    ],
    "Software": [
        "Application crash on startup",
        "Software update failed",
        "Database connection error",
        "Email client not syncing",
        "CRM application slow",
        "ERP system timeout",
        "File sharing app error",
        "Software license expired",
        "Application won't install",
        "Program freezing frequently",
    ],
    "Access": [
        "Cannot login to system",
        "Password reset not working",
        "Account locked out",
        "Access denied to folder",
        "User permissions issue",
        "MFA token not working",
        "Need access to shared drive",
        "Unable to access application",
        "Account disabled unexpectedly",
        "SSO authentication failing",
    ]
}

# Category-specific description templates for better ML differentiation
CATEGORY_DESCRIPTIONS = {
    "Network": [
        "VPN tunnel keeps dropping every few minutes. Users cannot access internal resources. Need immediate fix.",
        "Router in building C is not responding. Network connectivity lost for 50+ users. Critical issue.",
        "DNS server not resolving domain names. Intermittent connection to internet and intranet sites.",
        "Firewall rules blocking access to cloud applications. Users getting timeout errors.",
        "WiFi signal strength very weak on 3rd floor. Constant disconnections reported by multiple employees.",
        "Switch port in conference room not functioning. Cannot connect ethernet cable for presentations.",
        "Network drive mapping fails with error 'path not found'. Users unable to access shared folders.",
        "DHCP not assigning IP addresses. Clients showing 169.254.x.x addresses and no connectivity.",
        "Cannot ping production server from workstation. Network latency very high, packet loss 40%.",
        "Load balancer not distributing traffic. All requests going to single server causing performance issues.",
    ],
    "Hardware": [
        "Laptop screen flickering and showing horizontal lines. Display becomes unreadable after 10 minutes of use.",
        "Desktop computer won't power on at all. No lights, no beep codes, completely dead. Need replacement urgently.",
        "External monitor has black screen. Checked cables and power, still not displaying anything.",
        "Wireless keyboard not responding to keystrokes. Changed batteries, still not working properly.",
        "Mouse cursor moving erratically across screen. Cleaned mouse but issue persists.",
        "Office printer showing paper jam error but no paper visible. Already tried turning off and on.",
        "Hard drive making clicking noise. Computer very slow and files not opening. Suspect drive failure.",
        "Docking station not detecting laptop. Monitor and peripherals not connecting through dock.",
        "Laptop battery drains in 30 minutes even when fully charged. Battery health showing critical.",
        "Desktop computer overheating and shutting down randomly. Fans running very loud and hot air blowing.",
    ],
    "Software": [
        "Microsoft Outlook crashes immediately on startup. Error message 'Outlook.exe stopped working' appears.",
        "Software update for SAP ERP failed halfway. Now system won't load and showing database error.",
        "Cannot connect to SQL Server database. Application timing out with 'connection refused' error.",
        "Email client not syncing with Exchange server. Emails from yesterday still not downloaded.",
        "CRM application extremely slow. Takes 5+ minutes to load customer records. Unusable.",
        "ERP system shows timeout error when running monthly reports. Query execution exceeds 30 seconds.",
        "File sharing application gives permission denied error. Cannot upload documents to team folder.",
        "Software license expired for Adobe Creative Suite. Users unable to launch Photoshop and Illustrator.",
        "Cannot install latest version of accounting software. Setup fails with error code 0x80070643.",
        "Program freezes when opening large Excel files. CPU usage spikes to 100% and system becomes unresponsive.",
    ],
    "Access": [
        "Cannot login to employee portal. Username and password correct but getting 'invalid credentials' error.",
        "Password reset link in email not working. Click takes me to blank page instead of reset form.",
        "User account locked out after 3 failed login attempts. Need immediate unlock as user needs urgent access.",
        "Access denied when trying to open HR shared folder. Getting error 'insufficient permissions'.",
        "User permissions not updated correctly. Recent promotion to manager but still have read-only access.",
        "MFA authentication token not working. Authenticator app showing different code than system expects.",
        "Need access to finance database for quarterly reporting. Require read access to all tables.",
        "Unable to access Salesforce application. Account active but getting unauthorized access message.",
        "Account disabled without notice. User received email saying account deactivated but needs access immediately.",
        "Single sign-on authentication failing for cloud applications. Getting redirect loop error on login page.",
    ]
}


# ── helpers ──────────────────────────────────────────────────────────────────
def random_business_hours_ts(base_date: datetime.date, days_back: int = 90):
    """Return a random datetime within the last `days_back` days, 09‑19h."""
    day_offset = random.randint(0, days_back - 1)
    day = base_date - datetime.timedelta(days=day_offset)
    hour = random.randint(9, 18)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.datetime(day.year, day.month, day.day,
                             hour, minute, second)


def derive_priority(impact_label: str, urgency_label: str) -> str:
    return PRIORITY_MATRIX.get((impact_label, urgency_label), "Medium")


def realistic_resolution_time(sla: int) -> float:
    """Return hours between 0.2×SLA and 1.5×SLA, rounded to 1 decimal."""
    return round(random.uniform(0.2 * sla, 1.5 * sla), 1)


def affected_users(priority: str) -> int:
    if priority == "Critical":
        return random.randint(50, 500)
    if priority == "High":
        return random.randint(10, 100)
    if priority == "Medium":
        return random.randint(2, 20)
    return 1  # Low – single user


# ── load source dataset ─────────────────────────────────────────────────────
print("Loading dataset from Hugging Face …")
ds = load_dataset("6StringNinja/synthetic-servicenow-incidents")
source = ds["train"]
source_len = len(source)
print(f"Source rows: {source_len}")

# ── build target dataframe ───────────────────────────────────────────────────
TODAY = datetime.date(2026, 2, 14)
NUM_ROWS = 100000  # Generate 100000 unique tickets for maximum accuracy

# BALANCE CATEGORIES: 25,000 per category for perfect balance
CATEGORY_QUOTAS = {
    "Network": 25000,
    "Hardware": 25000,
    "Software": 25000,
    "Access": 25000
}
category_counts = {cat: 0 for cat in CATEGORY_QUOTAS.keys()}

# pre‑compute which rows get bad descriptions (~5 %)
bad_desc_indices = set(random.sample(range(NUM_ROWS), max(1, int(NUM_ROWS * 0.05))))

# knowledge article sequential counter
ka_counter = 1001

rows: list[dict] = []
for i in range(NUM_ROWS):
    # Cycle through source records (500 rows) to generate variations
    rec = source[i % source_len]
    
    # ── map fields ───────────────────────────────────────────────────────
    ticket_id = f"INC{1000 + i}"

    # category – ENFORCE BALANCED DISTRIBUTION (determine category first!)
    category = rec.get("category", None)
    all_categories = list(CATEGORY_QUOTAS.keys())
    
    # If source category is not valid or quota filled, pick from unfilled categories
    if category not in all_categories or category_counts[category] >= CATEGORY_QUOTAS[category]:
        # Find categories that haven't reached quota yet
        available_cats = [cat for cat in all_categories if category_counts[cat] < CATEGORY_QUOTAS[cat]]
        if not available_cats:
            break  # All quotas filled
        category = random.choice(available_cats)
    
    # Increment category count
    category_counts[category] += 1

    # Use category-specific title AND description templates for better differentiation
    title = random.choice(CATEGORY_TITLE_TEMPLATES[category])
    description = random.choice(CATEGORY_DESCRIPTIONS[category])

    # possibly replace description with a poor one (~5% of tickets)
    if i in bad_desc_indices:
        description = random.choice(POOR_DESCRIPTIONS)

    created_at = random_business_hours_ts(TODAY).strftime("%Y-%m-%d %H:%M:%S")
    channel = random.choice(CHANNELS)

    # impact / urgency
    raw_impact = rec.get("impact", 2)
    raw_urgency = rec.get("urgency", 2)
    impact_label = LEVEL_MAP.get(raw_impact, "Medium")
    urgency_label = LEVEL_MAP.get(raw_urgency, "Medium")

    priority = derive_priority(impact_label, urgency_label)
    sla = SLA_MAP[priority]
    res_time = realistic_resolution_time(sla)

    users = affected_users(priority)
    biz_service = random.choice(BUSINESS_SERVICES)
    ci = random.choice(CONFIG_ITEMS)
    location = "HeadOffice"

    # resolver group
    src_group = rec.get("assignment_group", "")
    if src_group == "Network Ops":
        resolver = "Network Team"
    elif src_group == "IT Support":
        resolver = random.choice(["Service Desk", "App Support"])
    else:
        resolver = CATEGORY_RESOLVER.get(category, random.choice(RESOLVER_GROUPS))

    status = random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
    root_cause = random.choice(ROOT_CAUSE_TEMPLATES.get(category, ["Unknown root cause"]))

    # No duplicates - all tickets are unique
    dup_gid = "NONE"
    ka_id = f"KA-{ka_counter}"
    ka_counter += 1

    rows.append({
        "ticket_id":                ticket_id,
        "title":                    title,
        "description":              description,
        "created_at":               created_at,
        "channel":                  channel,
        "ground_truth_category":    category,
        "ground_truth_priority":    priority,
        "impact_level":             impact_label,
        "urgency_level":            urgency_label,
        "affected_users_count":     users,
        "business_service":         biz_service,
        "configuration_item":       ci,
        "location":                 location,
        "ground_truth_resolver_group": resolver,
        "sla_hours":                sla,
        "resolution_time_hours":    res_time,
        "status":                   status,
        "root_cause":               root_cause,
        "duplicate_group_id":       dup_gid,
        "knowledge_article_id":     ka_id,
    })

df = pd.DataFrame(rows)

# ── sanity checks ────────────────────────────────────────────────────────────
assert df.shape[0] == NUM_ROWS, f"Expected {NUM_ROWS} rows, got {df.shape[0]}"
assert df.isnull().sum().sum() == 0, "Found null values!"
assert (df["duplicate_group_id"] == "NONE").all(), "Found duplicate groups (should be all NONE)"

# ── save ─────────────────────────────────────────────────────────────────────
OUTPUT = "synthetic_itsm_tickets.csv"
df.to_csv(OUTPUT, index=False)
print(f"\n✅  Saved {len(df)} tickets to {OUTPUT}")
print(df.head(3).to_string(index=False))
print(f"\nColumn list : {list(df.columns)}")
print(f"Bad descriptions  : {len(bad_desc_indices)}")
print(f"Priority dist     :\n{df['ground_truth_priority'].value_counts().to_string()}")
