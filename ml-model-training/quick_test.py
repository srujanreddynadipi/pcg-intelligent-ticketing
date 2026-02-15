"""
Quick Manual Test - Single Ticket from HuggingFace Models
Usage: python quick_test.py
"""

import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.sparse import hstack

# ===== EDIT YOUR TICKET HERE =====
TICKET_TITLE = " CRM application crashing"
TICKET_DESCRIPTION = "Sales team unable to load CRM module. Application throws null pointer exception"
IMPACT_LEVEL = "Low"  # Choose: Low, Medium, High
URGENCY_LEVEL = "Low"    # Choose: Low, Medium, High
AFFECTED_USERS = 1
# ==================================

print("="*80)
print("🚀 QUICK TICKET TEST FROM HUGGINGFACE")
print("="*80)

REPO_ID = "viveksai12/itsm-ticket-classifier"

print("\n📥 Downloading models from HuggingFace...")
category_model = joblib.load(hf_hub_download(REPO_ID, "models/category_classifier.pkl"))
tfidf = joblib.load(hf_hub_download(REPO_ID, "models/tfidf_vectorizer.pkl"))
priority_model = joblib.load(hf_hub_download(REPO_ID, "models/priority_predictor.pkl"))
impact_encoder = joblib.load(hf_hub_download(REPO_ID, "models/impact_encoder.pkl"))
urgency_encoder = joblib.load(hf_hub_download(REPO_ID, "models/urgency_encoder.pkl"))
resolver_model = joblib.load(hf_hub_download(REPO_ID, "models/resolver_router.pkl"))
category_encoder = joblib.load(hf_hub_download(REPO_ID, "models/category_encoder.pkl"))
print("✓ All models loaded from HuggingFace!")

print("\n" + "="*80)
print("📝 YOUR TICKET")
print("="*80)
print(f"Title: {TICKET_TITLE}")
print(f"Description: {TICKET_DESCRIPTION}")
print(f"Impact: {IMPACT_LEVEL} | Urgency: {URGENCY_LEVEL} | Affected: {AFFECTED_USERS}")

print("\n⏳ Analyzing with AI models...")

text_combined = f"{TICKET_TITLE} {TICKET_DESCRIPTION}"
text_vec = tfidf.transform([text_combined])

# Category
category_pred = category_model.predict(text_vec)[0]
category_proba = category_model.predict_proba(text_vec)[0]
category_conf = max(category_proba) * 100

# Priority
impact_enc = impact_encoder.transform([IMPACT_LEVEL]).reshape(-1, 1)
urgency_enc = urgency_encoder.transform([URGENCY_LEVEL]).reshape(-1, 1)
affected_enc = np.array([[AFFECTED_USERS]])
priority_features = hstack([text_vec, impact_enc, urgency_enc, affected_enc])
priority_pred = priority_model.predict(priority_features)[0]
priority_proba = priority_model.predict_proba(priority_features)[0]
priority_conf = max(priority_proba) * 100

# Resolver
category_enc = category_encoder.transform([category_pred]).reshape(-1, 1)
resolver_features = hstack([text_vec, category_enc, impact_enc, urgency_enc])
resolver_pred = resolver_model.predict(resolver_features)[0]
resolver_proba = resolver_model.predict_proba(resolver_features)[0]
resolver_conf = max(resolver_proba) * 100

print("\n" + "="*80)
print("🎯 AI PREDICTIONS")
print("="*80)
print(f"\n📋 Category:  {category_pred:<20s} ({category_conf:.1f}% confidence)")
print(f"🚨 Priority:  {priority_pred:<20s} ({priority_conf:.1f}% confidence)")
print(f"👥 Assign to: {resolver_pred:<20s} ({resolver_conf:.1f}% confidence)")
print("\n" + "="*80)
print("\n💡 To test another ticket, edit the values at the top of quick_test.py")
print("="*80)
