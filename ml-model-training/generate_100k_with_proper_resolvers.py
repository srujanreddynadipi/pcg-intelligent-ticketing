"""
Generate 100,000 balanced ITSM tickets with PROPER category→resolver mapping
11 categories with logical resolver group assignments
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# PROPER CATEGORY → RESOLVER MAPPING (from user specifications)
# ═══════════════════════════════════════════════════════════════════════════
CATEGORY_RESOLVER_MAP = {
    "Network": "Network Team",
    "Hardware": "Service Desk",
    "Software": "App Support",
    "Access": "Service Desk",
    "Database": "DBA Team",
    "Security": "Security Ops",
    "Cloud": "Cloud Ops",
    "DevOps": "DevOps Team",
    "Email": "Service Desk",
    "Monitoring": "Cloud Ops",
    "Service Request": "Service Desk"
}

CATEGORIES = list(CATEGORY_RESOLVER_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY-SPECIFIC TITLE AND DESCRIPTION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

TITLE_TEMPLATES = {
    "Network": [
        "VPN connection failed",
        "Unable to connect to network",
        "Network outage in building",
        "Slow network performance",
        "WiFi not working",
        "Cannot access internal resources",
        "VPN disconnects frequently",
        "Network drive not accessible",
        "Firewall blocking application",
        "DNS resolution failure",
        "Router configuration issue",
        "Switch port down",
        "Network timeout errors",
        "Proxy server not responding",
        "VPN error code 800"
    ],
    "Hardware": [
        "Laptop not powering on",
        "Desktop computer crashes",
        "Monitor not displaying",
        "Keyboard keys not working",
        "Mouse not responding",
        "Printer not printing",
        "Hard drive making noise",
        "Computer running very slow",
        "Blue screen error on startup",
        "Laptop battery not charging",
        "USB ports not working",
        "Docking station issues",
        "Scanner not detected",
        "Headset microphone not working",
        "Computer overheating"
    ],
    "Software": [
        "Application crashing on startup",
        "Software update failed",
        "Cannot install application",
        "Application license expired",
        "Software running slowly",
        "Application freezes frequently",
        "Cannot save files in application",
        "Software error message",
        "Application compatibility issue",
        "Need software upgrade",
        "Application missing features",
        "Software configuration help needed",
        "Application not launching",
        "Software integration issue",
        "Application performance degraded"
    ],
    "Access": [
        "Request access to shared folder",
        "Cannot login to application",
        "Need permissions for system",
        "Access denied error",
        "Password reset request",
        "Account locked out",
        "Request VPN access",
        "Need admin rights",
        "Cannot access file share",
        "Request application access",
        "Permission denied error",
        "Need access to database",
        "Request email access",
        "Cannot access portal",
        "MFA not working"
    ],
    "Database": [
        "Database connection timeout",
        "Query running very slow",
        "Database server not responding",
        "Cannot connect to database",
        "Database error message",
        "SQL query failing",
        "Database backup failed",
        "Replication lag issue",
        "Database performance degraded",
        "Table lock timeout",
        "Stored procedure error",
        "Database migration issue",
        "Connection pool exhausted",
        "Transaction deadlock",
        "Database corruption detected"
    ],
    "Security": [
        "Malware detected on computer",
        "Suspicious email received",
        "Account compromised",
        "Security alert notification",
        "Certificate expired",
        "Firewall rule request",
        "Security scan needed",
        "Ransomware suspected",
        "Unauthorized access attempt",
        "Data breach notification",
        "Security policy violation",
        "Phishing email reported",
        "SSL certificate error",
        "Antivirus not updating",
        "Security audit required"
    ],
    "Cloud": [
        "Azure VM not starting",
        "AWS S3 access denied",
        "Cloud storage full",
        "Cloud application timeout",
        "Cannot access cloud resource",
        "Cloud service unavailable",
        "Azure AD sync issue",
        "AWS billing spike",
        "Cloud backup failed",
        "Kubernetes pod crashing",
        "Cloud scaling issue",
        "Azure function error",
        "Cloud network connectivity",
        "Container registry issue",
        "Cloud resource quota exceeded"
    ],
    "DevOps": [
        "CI/CD pipeline failed",
        "Deployment to production failed",
        "Jenkins build error",
        "Git repository access issue",
        "Docker container not starting",
        "Kubernetes deployment error",
        "Terraform apply failed",
        "Pipeline timeout error",
        "Build artifacts missing",
        "Helm chart installation failed",
        "Container image pull error",
        "Release pipeline stuck",
        "Deployment rollback needed",
        "Pipeline permissions issue",
        "Build agent offline"
    ],
    "Email": [
        "Email not sending",
        "Cannot receive emails",
        "Mailbox quota exceeded",
        "Email stuck in outbox",
        "Outlook not syncing",
        "Email bouncing back",
        "Cannot access mailbox",
        "Email delivery delayed",
        "Outlook crashes on startup",
        "Shared mailbox access issue",
        "Email rules not working",
        "Calendar sync issue",
        "Email attachments blocked",
        "Outlook search not working",
        "Email forwarding not working"
    ],
    "Monitoring": [
        "Alert not triggering",
        "Monitoring dashboard down",
        "False positive alerts",
        "Metric collection failed",
        "Monitoring agent offline",
        "Alert notification not received",
        "Dashboard displaying wrong data",
        "Monitoring integration issue",
        "Log collection stopped",
        "Performance metrics missing",
        "Alert threshold not working",
        "Monitoring tool not accessible",
        "Grafana dashboard error",
        "Prometheus scraping failed",
        "Monitoring agent upgrade needed"
    ],
    "Service Request": [
        "Request new laptop",
        "Need software installation",
        "Request mobile phone",
        "Need new monitor",
        "Request conference room equipment",
        "Need printer setup",
        "Request desk phone",
        "Need additional storage",
        "Request software license",
        "Need workstation upgrade",
        "Request headset",
        "Need docking station",
        "Request access card",
        "Need office supplies",
        "Request equipment replacement"
    ]
}

DESCRIPTION_TEMPLATES = {
    "Network": [
        "User cannot connect to VPN after latest update",
        "Network connectivity issues affecting entire floor",
        "Unable to access internal websites and applications",
        "Network speed is extremely slow during business hours",
        "WiFi keeps disconnecting every few minutes",
        "Cannot map network drives to user laptop",
        "VPN authentication fails with error code",
        "Network shares are not accessible",
        "Firewall is blocking required application ports",
        "DNS server not resolving internal hostnames"
    ],
    "Hardware": [
        "Laptop does not turn on even after charging overnight",
        "Desktop computer randomly shuts down during work",
        "External monitor shows no signal from laptop",
        "Several keyboard keys are not responding",
        "Mouse pointer is jumping around the screen",
        "Printer shows error message and cannot print",
        "Hard drive making clicking sounds",
        "Computer performance is degraded significantly",
        "Blue screen appears on startup with error code",
        "Battery indicator shows not charging"
    ],
    "Software": [
        "Application displays error message on launch",
        "Software update installation failed midway",
        "Cannot complete installation of required application",
        "License key validation fails for software",
        "Application response time is very slow",
        "Software freezes and requires forced termination",
        "Cannot save work in application - save button grayed out",
        "Error code appears when opening application",
        "Application not compatible with Windows 11",
        "Need to upgrade to latest software version"
    ],
    "Access": [
        "Need access to shared drive for project work",
        "Login credentials not working for application",
        "Require elevated permissions for system administration",
        "Access denied when trying to open file",
        "Forgot password and need to reset",
        "Account locked after multiple login attempts",
        "New employee needs VPN access setup",
        "Require administrator rights to install software",
        "Cannot access department file share",
        "Need access provisioned for new application"
    ],
    "Database": [
        "Database connection times out after 30 seconds",
        "Query execution time has increased significantly",
        "Cannot connect to production database server",
        "Error message appears when running database query",
        "SQL stored procedure fails with error",
        "Database backup job failed last night",
        "Replication lag is causing data inconsistency",
        "Application cannot connect to database pool",
        "Transaction is blocked by table lock",
        "Database slow performance affecting application"
    ],
    "Security": [
        "Antivirus detected malware on workstation",
        "Received suspicious email with attachment",
        "User account showing suspicious login activity",
        "Security alert notification from monitoring system",
        "SSL certificate expired on internal website",
        "Need firewall rule changes for application",
        "Security scan shows vulnerabilities",
        "Files encrypted with ransomware extension",
        "Detected unauthorized access attempt to server",
        "Potential data breach detected in logs"
    ],
    "Cloud": [
        "Azure virtual machine stuck in starting state",
        "Cannot access S3 bucket - permission denied",
        "Cloud storage showing almost full capacity",
        "Cloud-based application showing timeout errors",
        "Access denied to cloud resource group",
        "Cloud service showing degraded availability",
        "Azure AD sync not working properly",
        "AWS bill is higher than expected this month",
        "Cloud backup job failed with error",
        "Kubernetes pods in crash loop state"
    ],
    "DevOps": [
        "CI/CD pipeline failing at deployment stage",
        "Production deployment failed with errors",
        "Jenkins build fails with compilation error",
        "Cannot access Git repository - permission denied",
        "Docker container exits immediately after start",
        "Kubernetes deployment shows ImagePullBackOff",
        "Terraform apply command failed with state lock error",
        "Pipeline running longer than expected timeout",
        "Build artifacts not published to repository",
        "Helm installation fails with validation error"
    ],
    "Email": [
        "Outgoing emails are stuck in outbox",
        "Not receiving any emails since this morning",
        "Mailbox full - cannot send or receive emails",
        "Email fails to send with error message",
        "Outlook not synchronizing with Exchange server",
        "Sent emails are bouncing back with delivery failure",
        "Cannot login to webmail interface",
        "Email delivery delayed by several hours",
        "Outlook crashes when trying to open",
        "Cannot access shared mailbox"
    ],
    "Monitoring": [
        "Critical alert not generating notification",
        "Monitoring dashboard is not loading",
        "Getting false positive alerts every hour",
        "Metrics not being collected from servers",
        "Monitoring agent shows offline status",
        "Alert emails are not being received",
        "Dashboard showing incorrect metric values",
        "Integration with monitoring tool not working",
        "Application logs not being collected",
        "Performance counters missing from dashboard"
    ],
    "Service Request": [
        "Need new laptop for new employee starting next week",
        "Request installation of Adobe Creative Suite",
        "New employee requires mobile phone setup",
        "Need dual monitor setup for workstation",
        "Request projector and screen for conference room",
        "Need network printer installed on workstation",
        "Request IP desk phone for new office",
        "Need additional 500GB storage on network drive",
        "Request Microsoft Visio license for team member",
        "Need workstation RAM upgrade to 32GB"
    ]
}

# Other constants
CHANNELS = ["portal", "email", "chat", "phone", "monitoring"]
LOCATIONS = ["HeadOffice", "Branch1", "Branch2", "Remote", "DataCenter"]
URGENCY_LEVELS = ["High", "Medium", "Low"]
IMPACT_LEVELS = ["High", "Medium", "Low"]

# Priority matrix (impact, urgency) → priority
PRIORITY_MATRIX = {
    ("High", "High"): "Critical",
    ("High", "Medium"): "High",
    ("High", "Low"): "Medium",
    ("Medium", "High"): "High",
    ("Medium", "Medium"): "Medium",
    ("Medium", "Low"): "Low",
    ("Low", "High"): "Medium",
    ("Low", "Medium"): "Low",
    ("Low", "Low"): "Low"
}

def generate_ticket(ticket_id, category):
    """Generate a single ticket with proper resolver mapping"""
    
    # Select random title and description from category-specific templates
    title = random.choice(TITLE_TEMPLATES[category])
    description = random.choice(DESCRIPTION_TEMPLATES[category])
    
    # Get correct resolver based on category
    resolver_group = CATEGORY_RESOLVER_MAP[category]
    
    # Random impact and urgency
    impact = random.choice(IMPACT_LEVELS)
    urgency = random.choice(URGENCY_LEVELS)
    
    # Calculate priority
    priority = PRIORITY_MATRIX[(impact, urgency)]
    
    # Other fields
    channel = random.choice(CHANNELS)
    location = random.choice(LOCATIONS)
    
    # Affected users (more for high impact)
    if impact == "High":
        affected_users = random.randint(10, 100)
    elif impact == "Medium":
        affected_users = random.randint(2, 10)
    else:
        affected_users = 1
    
    # Generate timestamps
    created_date = datetime.now() - timedelta(days=random.randint(0, 90))
    
    return {
        "ticket_id": f"INC{ticket_id:07d}",
        "title": title,
        "description": description,
        "category": category,
        "ground_truth_resolver_group": resolver_group,  # PROPER MAPPING!
        "priority": priority,
        "impact": impact,
        "urgency": urgency,
        "channel": channel,
        "location": location,
        "affected_users": affected_users,
        "created_date": created_date.strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_balanced_dataset(total_tickets=100000):
    """Generate perfectly balanced dataset with proper resolver mapping"""
    
    print("="*80)
    print("GENERATING 100K ITSM TICKETS WITH PROPER CATEGORY→RESOLVER MAPPING")
    print("="*80)
    
    # Calculate tickets per category (perfectly balanced)
    tickets_per_category = total_tickets // len(CATEGORIES)
    remaining = total_tickets % len(CATEGORIES)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Total tickets: {total_tickets:,}")
    print(f"   Categories: {len(CATEGORIES)}")
    print(f"   Tickets per category: {tickets_per_category:,}")
    print(f"   Extra tickets: {remaining}")
    
    print(f"\n🎯 Category → Resolver Mapping:")
    for cat, res in CATEGORY_RESOLVER_MAP.items():
        print(f"   {cat:20s} → {res}")
    
    print(f"\n⚙️  Generating tickets...")
    
    tickets = []
    ticket_id = 1
    
    # Generate balanced tickets
    for category in CATEGORIES:
        count = tickets_per_category
        if remaining > 0:
            count += 1
            remaining -= 1
        
        for _ in range(count):
            ticket = generate_ticket(ticket_id, category)
            tickets.append(ticket)
            ticket_id += 1
        
        print(f"   ✓ {category:20s}: {count:,} tickets")
    
    # Create DataFrame
    df = pd.DataFrame(tickets)
    
    # Shuffle to mix categories
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    print("\n" + "="*80)
    print("STEP 1: GENERATING DATASET")
    print("="*80)
    
    df = generate_balanced_dataset(100000)
    
    print("\n" + "="*80)
    print("STEP 2: VALIDATING DATA")
    print("="*80)
    
    # Validate category distribution
    print(f"\n📂 Category Distribution:")
    cat_dist = df['category'].value_counts().sort_index()
    for cat, count in cat_dist.items():
        pct = (count / len(df)) * 100
        print(f"   {cat:20s}: {count:6,} ({pct:5.2f}%)")
    
    # Validate resolver distribution
    print(f"\n👥 Resolver Group Distribution:")
    res_dist = df['ground_truth_resolver_group'].value_counts().sort_index()
    for res, count in res_dist.items():
        pct = (count / len(df)) * 100
        print(f"   {res:20s}: {count:6,} ({pct:5.2f}%)")
    
    # Validate category→resolver mapping
    print(f"\n🔍 Validating Category→Resolver Mapping:")
    print(f"{'Category':<20} {'Expected Resolver':<20} {'Actual Resolver':<20} {'Status':<10}")
    print("─"*80)
    
    all_correct = True
    for category in CATEGORIES:
        expected_resolver = CATEGORY_RESOLVER_MAP[category]
        category_tickets = df[df['category'] == category]
        actual_resolvers = category_tickets['ground_truth_resolver_group'].unique()
        
        if len(actual_resolvers) == 1 and actual_resolvers[0] == expected_resolver:
            status = "✅ CORRECT"
        else:
            status = "❌ ERROR"
            all_correct = False
        
        print(f"{category:<20} {expected_resolver:<20} {actual_resolvers[0]:<20} {status:<10}")
    
    print("\n" + "="*80)
    print("STEP 3: SAVING DATASET")
    print("="*80)
    
    output_file = "synthetic_itsm_tickets.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Dataset saved: {output_file}")
    print(f"   Total records: {len(df):,}")
    print(f"   Total categories: {df['category'].nunique()}")
    print(f"   Total resolvers: {df['ground_truth_resolver_group'].nunique()}")
    print(f"   File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    if all_correct:
        print(f"\n✅ VALIDATION PASSED: All categories have correct resolver mapping!")
    else:
        print(f"\n❌ VALIDATION FAILED: Some categories have incorrect resolver mapping!")
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = main()
