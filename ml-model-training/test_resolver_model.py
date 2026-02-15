"""
Test the trained Resolver Routing Model with real test cases
Covers all 11 categories
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import hstack

print("="*80)
print("TESTING RESOLVER ROUTING MODEL")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════
print("\n⚡ Loading models...")

models_dir = Path("models_improved")

resolver_model = joblib.load(models_dir / "resolver_router.pkl")
tfidf_vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
category_encoder = joblib.load(models_dir / "category_encoder.pkl")
impact_encoder = joblib.load(models_dir / "impact_encoder.pkl")
urgency_encoder = joblib.load(models_dir / "urgency_encoder.pkl")

print("✅ All models loaded!")

# ═══════════════════════════════════════════════════════════════════════════
# EXPECTED CATEGORY→RESOLVER MAPPING
# ═══════════════════════════════════════════════════════════════════════════
EXPECTED_MAPPING = {
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

# ═══════════════════════════════════════════════════════════════════════════
# TEST CASES (covering all 11 categories)
# ═══════════════════════════════════════════════════════════════════════════
test_cases = [
    # Network
    {
        "id": 1,
        "category": "Network",
        "title": "VPN connection failed",
        "description": "User cannot connect to VPN after Windows update",
        "impact": "High",
        "urgency": "High",
        "affected_users": 50
    },
    {
        "id": 2,
        "category": "Network",
        "title": "Network outage in building",
        "description": "Entire floor cannot access network resources",
        "impact": "High",
        "urgency": "High",
        "affected_users": 100
    },
    
    # Hardware
    {
        "id": 3,
        "category": "Hardware",
        "title": "Laptop not powering on",
        "description": "User laptop does not turn on even after charging",
        "impact": "Medium",
        "urgency": "High",
        "affected_users": 1
    },
    {
        "id": 4,
        "category": "Hardware",
        "title": "Printer not printing",
        "description": "Office printer shows error and cannot print documents",
        "impact": "Medium",
        "urgency": "Medium",
        "affected_users": 5
    },
    
    # Software
    {
        "id": 5,
        "category": "Software",
        "title": "Application crashing on startup",
        "description": "CRM application crashes immediately after launch",
        "impact": "High",
        "urgency": "High",
        "affected_users": 20
    },
    {
        "id": 6,
        "category": "Software",
        "title": "Cannot install application",
        "description": "Installation of required software fails with error",
        "impact": "Low",
        "urgency": "Low",
        "affected_users": 1
    },
    
    # Access
    {
        "id": 7,
        "category": "Access",
        "title": "Request access to shared folder",
        "description": "New employee needs access to department shared drive",
        "impact": "Low",
        "urgency": "Medium",
        "affected_users": 1
    },
    {
        "id": 8,
        "category": "Access",
        "title": "Cannot login to application",
        "description": "User credentials not working for payroll system",
        "impact": "Medium",
        "urgency": "High",
        "affected_users": 1
    },
    
    # Database
    {
        "id": 9,
        "category": "Database",
        "title": "Database connection timeout",
        "description": "Application cannot connect to production database server",
        "impact": "High",
        "urgency": "High",
        "affected_users": 50
    },
    {
        "id": 10,
        "category": "Database",
        "title": "Query running very slow",
        "description": "SQL query execution time has increased significantly",
        "impact": "Medium",
        "urgency": "Medium",
        "affected_users": 10
    },
    
    # Security
    {
        "id": 11,
        "category": "Security",
        "title": "Malware detected on computer",
        "description": "Antivirus detected suspicious malware on user workstation",
        "impact": "High",
        "urgency": "High",
        "affected_users": 1
    },
    {
        "id": 12,
        "category": "Security",
        "title": "Suspicious email received",
        "description": "User received phishing email with malicious attachment",
        "impact": "Medium",
        "urgency": "High",
        "affected_users": 1
    },
    
    # Cloud
    {
        "id": 13,
        "category": "Cloud",
        "title": "Azure VM not starting",
        "description": "Virtual machine stuck in starting state in Azure portal",
        "impact": "High",
        "urgency": "High",
        "affected_users": 10
    },
    {
        "id": 14,
        "category": "Cloud",
        "title": "Cloud storage full",
        "description": "AWS S3 bucket reached storage quota limit",
        "impact": "Medium",
        "urgency": "Medium",
        "affected_users": 5
    },
    
    # DevOps
    {
        "id": 15,
        "category": "DevOps",
        "title": "CI/CD pipeline failed",
        "description": "Jenkins build pipeline failing at deployment stage",
        "impact": "High",
        "urgency": "High",
        "affected_users": 10
    },
    {
        "id": 16,
        "category": "DevOps",
        "title": "Docker container not starting",
        "description": "Container exits immediately with error code",
        "impact": "Medium",
        "urgency": "High",
        "affected_users": 5
    },
    
    # Email
    {
        "id": 17,
        "category": "Email",
        "title": "Email not sending",
        "description": "Outgoing emails stuck in outbox for multiple users",
        "impact": "High",
        "urgency": "High",
        "affected_users": 30
    },
    {
        "id": 18,
        "category": "Email",
        "title": "Mailbox quota exceeded",
        "description": "User mailbox full cannot send or receive emails",
        "impact": "Medium",
        "urgency": "Medium",
        "affected_users": 1
    },
    
    # Monitoring
    {
        "id": 19,
        "category": "Monitoring",
        "title": "Alert not triggering",
        "description": "Critical monitoring alert not generating notifications",
        "impact": "High",
        "urgency": "High",
        "affected_users": 10
    },
    {
        "id": 20,
        "category": "Monitoring",
        "title": "Dashboard displaying wrong data",
        "description": "Grafana dashboard showing incorrect metric values",
        "impact": "Medium",
        "urgency": "Medium",
        "affected_users": 5
    },
    
    # Service Request
    {
        "id": 21,
        "category": "Service Request",
        "title": "Request new laptop",
        "description": "New employee requires laptop setup for work",
        "impact": "Low",
        "urgency": "Low",
        "affected_users": 1
    },
    {
        "id": 22,
        "category": "Service Request",
        "title": "Need software installation",
        "description": "Request installation of Adobe Creative Suite on workstation",
        "impact": "Low",
        "urgency": "Low",
        "affected_users": 1
    }
]

print(f"\n📝 Testing {len(test_cases)} test cases across all 11 categories...")

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def predict_resolver(ticket):
    """Predict resolver group for a ticket"""
    
    # Combine title and description
    text = f"{ticket['title']} {ticket['description']}"
    text_lower = text.lower()
    
    # Vectorize text
    text_vec = tfidf_vectorizer.transform([text])
    
    # Extract keywords
    has_network_keyword = int(any(kw in text_lower for kw in ['network', 'vpn', 'wifi', 'dns', 'firewall', 'router', 'switch', 'connection', 'connectivity']))
    has_hardware_keyword = int(any(kw in text_lower for kw in ['laptop', 'desktop', 'computer', 'monitor', 'keyboard', 'mouse', 'printer', 'hardware', 'device']))
    has_database_keyword = int(any(kw in text_lower for kw in ['database', 'sql', 'query', 'db', 'table', 'replication', 'backup', 'connection pool']))
    has_cloud_keyword = int(any(kw in text_lower for kw in ['azure', 'aws', 'cloud', 'vm', 'container', 'kubernetes', 'docker', 's3', 'blob']))
    has_security_keyword = int(any(kw in text_lower for kw in ['security', 'malware', 'virus', 'phishing', 'breach', 'unauthorized', 'certificate', 'firewall']))
    has_devops_keyword = int(any(kw in text_lower for kw in ['cicd', 'pipeline', 'jenkins', 'git', 'docker', 'kubernetes', 'terraform', 'helm', 'deployment']))
    has_email_keyword = int(any(kw in text_lower for kw in ['email', 'outlook', 'mailbox', 'exchange', 'mail', 'inbox', 'outbox', 'smtp']))
    
    keyword_features = [[
        has_network_keyword, has_hardware_keyword, has_database_keyword,
        has_cloud_keyword, has_security_keyword, has_devops_keyword,
        has_email_keyword
    ]]
    
    # Encode categorical
    category_encoded = category_encoder.transform([ticket['category']])[0]
    impact_encoded = impact_encoder.transform([ticket['impact']])[0]
    urgency_encoded = urgency_encoder.transform([ticket['urgency']])[0]
    
    categorical_features = [[
        category_encoded,
        impact_encoded,
        urgency_encoded,
        ticket['affected_users']
    ]]
    
    # Combine features
    X = hstack([text_vec, categorical_features, keyword_features])
    
    # Predict
    prediction = resolver_model.predict(X)[0]
    probabilities = resolver_model.predict_proba(X)[0]
    confidence = probabilities.max() * 100
    
    return prediction, confidence

# ═══════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)

results = []
correct = 0
total = 0

category_stats = {}

for ticket in test_cases:
    predicted_resolver, confidence = predict_resolver(ticket)
    expected_resolver = EXPECTED_MAPPING[ticket['category']]
    is_correct = predicted_resolver == expected_resolver
    
    if is_correct:
        correct += 1
    total += 1
    
    # Track category stats
    if ticket['category'] not in category_stats:
        category_stats[ticket['category']] = {'correct': 0, 'total': 0}
    category_stats[ticket['category']]['total'] += 1
    if is_correct:
        category_stats[ticket['category']]['correct'] += 1
    
    status = "✅" if is_correct else "❌"
    
    print(f"\n{'─'*80}")
    print(f"Test #{ticket['id']}: {ticket['title']}")
    print(f"{'─'*80}")
    print(f"Category: {ticket['category']}")
    print(f"Expected: {expected_resolver}")
    print(f"Predicted: {predicted_resolver} ({confidence:.1f}% confidence)")
    print(f"Status: {status} {'CORRECT' if is_correct else 'INCORRECT'}")
    
    results.append({
        'id': ticket['id'],
        'category': ticket['category'],
        'expected': expected_resolver,
        'predicted': predicted_resolver,
        'confidence': confidence,
        'correct': is_correct
    })

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

accuracy = (correct / total) * 100
print(f"\n📊 Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")

print(f"\n📂 Accuracy by Category:")
for category in sorted(category_stats.keys()):
    stats = category_stats[category]
    cat_accuracy = (stats['correct'] / stats['total']) * 100
    status = "✅" if cat_accuracy == 100 else "⚠️"
    print(f"   {status} {category:20s}: {stats['correct']}/{stats['total']} ({cat_accuracy:5.1f}%)")

if accuracy == 100:
    print(f"\n🎉 PERFECT! All predictions are correct!")
elif accuracy >= 95:
    print(f"\n✅ EXCELLENT! Accuracy >= 95%")
elif accuracy >= 90:
    print(f"\n✅ GOOD! Accuracy >= 90%")
else:
    print(f"\n⚠️  NEEDS REVIEW! Accuracy < 90%")

# Detailed table
print(f"\n📋 Detailed Results:")
print(f"{'─'*80}")
print(f"{'ID':<4} {'Category':<16} {'Expected':<20} {'Predicted':<20} {'Conf':<6} {'Result':<6}")
print(f"{'─'*80}")
for r in results:
    status = "✅" if r['correct'] else "❌"
    print(f"{r['id']:<4} {r['category']:<16} {r['expected']:<20} {r['predicted']:<20} {r['confidence']:5.1f}% {status:<6}")
print(f"{'─'*80}")

print("\n" + "="*80)
print("✅ TESTING COMPLETE!")
print("="*80)
