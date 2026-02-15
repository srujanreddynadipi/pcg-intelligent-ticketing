"""
Demo Test Environment for ITSM Models from HuggingFace
Downloads models and tests with predefined cases
"""

import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.sparse import hstack

print("="*80)
print("ITSM AI MODEL TESTING - DEMO MODE")
print("="*80)
print("\n📥 Downloading models from HuggingFace Hub...")

# Configuration
REPO_ID = "viveksai12/itsm-ticket-classifier"

# Download all required models
print("\n[1/6] Downloading Category Classifier...")
category_model_path = hf_hub_download(repo_id=REPO_ID, filename="models/category_classifier.pkl")
category_model = joblib.load(category_model_path)
print("✓ Category Classifier loaded")

print("\n[2/6] Downloading TF-IDF Vectorizer...")
tfidf_path = hf_hub_download(repo_id=REPO_ID, filename="models/tfidf_vectorizer.pkl")
tfidf = joblib.load(tfidf_path)
print("✓ TF-IDF Vectorizer loaded")

print("\n[3/6] Downloading Priority Predictor...")
priority_model_path = hf_hub_download(repo_id=REPO_ID, filename="models/priority_predictor.pkl")
priority_model = joblib.load(priority_model_path)
print("✓ Priority Predictor loaded")

print("\n[4/6] Downloading Label Encoders...")
impact_encoder_path = hf_hub_download(repo_id=REPO_ID, filename="models/impact_encoder.pkl")
impact_encoder = joblib.load(impact_encoder_path)
urgency_encoder_path = hf_hub_download(repo_id=REPO_ID, filename="models/urgency_encoder.pkl")
urgency_encoder = joblib.load(urgency_encoder_path)
print("✓ Impact & Urgency Encoders loaded")

print("\n[5/6] Downloading Resolver Router...")
resolver_model_path = hf_hub_download(repo_id=REPO_ID, filename="models/resolver_router.pkl")
resolver_model = joblib.load(resolver_model_path)
category_encoder_path = hf_hub_download(repo_id=REPO_ID, filename="models/category_encoder.pkl")
category_encoder = joblib.load(category_encoder_path)
print("✓ Resolver Router & Category Encoder loaded")

print("\n[6/6] All models downloaded successfully from HuggingFace! 🎉")

def predict_ticket(title, description, impact, urgency, affected_users):
    """Make predictions for a ticket"""
    text_combined = f"{title} {description}"
    text_vec = tfidf.transform([text_combined])
    
    # Category
    category_pred = category_model.predict(text_vec)[0]
    category_proba = category_model.predict_proba(text_vec)[0]
    category_confidence = float(max(category_proba))
    category_classes = category_model.classes_
    top_3_cat_idx = np.argsort(category_proba)[-3:][::-1]
    top_3_categories = [(category_classes[i], category_proba[i]) for i in top_3_cat_idx]
    
    # Priority
    impact_encoded = impact_encoder.transform([impact]).reshape(-1, 1)
    urgency_encoded = urgency_encoder.transform([urgency]).reshape(-1, 1)
    affected_encoded = np.array([[affected_users]])
    priority_features = hstack([text_vec, impact_encoded, urgency_encoded, affected_encoded])
    priority_pred = priority_model.predict(priority_features)[0]
    priority_proba = priority_model.predict_proba(priority_features)[0]
    priority_confidence = float(max(priority_proba))
    priority_classes = priority_model.classes_
    top_3_pri_idx = np.argsort(priority_proba)[-3:][::-1]
    top_3_priorities = [(priority_classes[i], priority_proba[i]) for i in top_3_pri_idx]
    
    # Resolver
    category_encoded = category_encoder.transform([category_pred]).reshape(-1, 1)
    resolver_features = hstack([text_vec, category_encoded, impact_encoded, urgency_encoded])
    resolver_pred = resolver_model.predict(resolver_features)[0]
    resolver_proba = resolver_model.predict_proba(resolver_features)[0]
    resolver_confidence = float(max(resolver_proba))
    resolver_classes = resolver_model.classes_
    top_3_res_idx = np.argsort(resolver_proba)[-3:][::-1]
    top_3_resolvers = [(resolver_classes[i], resolver_proba[i]) for i in top_3_res_idx]
    
    return {
        'category': category_pred, 'category_confidence': category_confidence, 'top_3_categories': top_3_categories,
        'priority': priority_pred, 'priority_confidence': priority_confidence, 'top_3_priorities': top_3_priorities,
        'resolver': resolver_pred, 'resolver_confidence': resolver_confidence, 'top_3_resolvers': top_3_resolvers
    }

def confidence_label(conf):
    if conf >= 0.9: return "Very High ⭐⭐⭐"
    elif conf >= 0.75: return "High ⭐⭐"
    elif conf >= 0.6: return "Moderate ⭐"
    elif conf >= 0.4: return "Low ⚠️"
    else: return "Very Low ⚠️⚠️"

def print_results(results, ticket_info):
    print("\n" + "="*80)
    print(f"PREDICTION RESULTS")
    print("="*80)
    print(f"📝 Title: {ticket_info['title']}")
    print(f"📄 Description: {ticket_info['description'][:100]}...")
    print(f"📊 Impact: {ticket_info['impact']} | Urgency: {ticket_info['urgency']} | Affected: {ticket_info['affected']}")
    
    print("\n📋 CATEGORY CLASSIFICATION")
    print("-" * 80)
    print(f"✓ Predicted: {results['category']}")
    print(f"  Confidence: {results['category_confidence']*100:.2f}% ({confidence_label(results['category_confidence'])})")
    print("  Top 3:")
    for i, (cat, prob) in enumerate(results['top_3_categories'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {i}. {cat:15s} {prob*100:5.2f}% {bar}")
    
    print("\n🚨 PRIORITY PREDICTION")
    print("-" * 80)
    print(f"✓ Predicted: {results['priority']}")
    print(f"  Confidence: {results['priority_confidence']*100:.2f}% ({confidence_label(results['priority_confidence'])})")
    print("  Top 3:")
    for i, (pri, prob) in enumerate(results['top_3_priorities'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {i}. {pri:10s} {prob*100:5.2f}% {bar}")
    
    print("\n👥 RESOLVER GROUP ROUTING")
    print("-" * 80)
    print(f"✓ Assigned to: {results['resolver']}")
    print(f"  Confidence: {results['resolver_confidence']*100:.2f}% ({confidence_label(results['resolver_confidence'])})")
    print("  Top 3:")
    for i, (res, prob) in enumerate(results['top_3_resolvers'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {i}. {res:25s} {prob*100:5.2f}% {bar}")

# Test Cases
test_cases = [
    {
        'title': 'VPN Connection Failed',
        'description': 'Unable to connect to corporate VPN from home. Getting authentication error after entering credentials. Need access urgently for remote work.',
        'impact': 'High',
        'urgency': 'High',
        'affected': 1
    },
    {
        'title': 'Laptop Screen Flickering',
        'description': 'Dell laptop screen keeps flickering and going black intermittently. Makes it impossible to work. Hardware issue suspected.',
        'impact': 'High',
        'urgency': 'Medium',
        'affected': 1
    },
    {
        'title': 'Cannot Access Shared Drive',
        'description': 'Getting permission denied error when trying to access department shared drive. Need to retrieve important documents for meeting.',
        'impact': 'Medium',
        'urgency': 'Medium',
        'affected': 1
    },
    {
        'title': 'Database Server Down',
        'description': 'Production database server is not responding. Multiple applications are affected. This is critical and impacting all users nationwide.',
        'impact': 'High',
        'urgency': 'High',
        'affected': 500
    },
    {
        'title': 'Outlook Email Not Syncing',
        'description': 'Emails are not syncing in Outlook. Last sync was 3 hours ago. Can still access via webmail but need desktop client working.',
        'impact': 'Low',
        'urgency': 'Low',
        'affected': 1
    }
]

print("\n" + "="*80)
print("🧪 TESTING WITH PREDEFINED TEST CASES")
print("="*80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n\n{'='*80}")
    print(f"TEST CASE #{i}")
    results = predict_ticket(
        test_case['title'],
        test_case['description'],
        test_case['impact'],
        test_case['urgency'],
        test_case['affected']
    )
    print_results(results, test_case)

print("\n\n" + "="*80)
print("✅ DEMO COMPLETE!")
print("="*80)
print(f"Total test cases: {len(test_cases)}")
print("\n💡 To test with your own inputs, run:")
print("   python test_from_huggingface.py")
print("\n🌐 Models loaded from: https://huggingface.co/viveksai12/itsm-ticket-classifier")
print("="*80)
