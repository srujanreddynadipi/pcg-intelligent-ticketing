"""
Interactive Test Environment for ITSM Models from HuggingFace
Downloads models from HuggingFace Hub and provides manual testing interface
"""

import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.sparse import hstack

print("="*80)
print("ITSM AI MODEL TESTING - INTERACTIVE MODE")
print("="*80)
print("\n📥 Downloading models from HuggingFace Hub...")

# Configuration
REPO_ID = "viveksai12/itsm-ticket-classifier"

# Download all required models
print("\n[1/6] Downloading Category Classifier...")
category_model_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/category_classifier.pkl"
)
category_model = joblib.load(category_model_path)
print("✓ Category Classifier loaded")

print("\n[2/6] Downloading TF-IDF Vectorizer...")
tfidf_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/tfidf_vectorizer.pkl"
)
tfidf = joblib.load(tfidf_path)
print("✓ TF-IDF Vectorizer loaded")

print("\n[3/6] Downloading Priority Predictor...")
priority_model_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/priority_predictor.pkl"
)
priority_model = joblib.load(priority_model_path)
print("✓ Priority Predictor loaded")

print("\n[4/6] Downloading Label Encoders...")
impact_encoder_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/impact_encoder.pkl"
)
impact_encoder = joblib.load(impact_encoder_path)

urgency_encoder_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/urgency_encoder.pkl"
)
urgency_encoder = joblib.load(urgency_encoder_path)
print("✓ Impact & Urgency Encoders loaded")

print("\n[5/6] Downloading Resolver Router...")
resolver_model_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/resolver_router.pkl"
)
resolver_model = joblib.load(resolver_model_path)

category_encoder_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename="models/category_encoder.pkl"
)
category_encoder = joblib.load(category_encoder_path)
print("✓ Resolver Router & Category Encoder loaded")

print("\n[6/6] All models downloaded successfully!")
print("="*80)

def predict_ticket(title, description, impact, urgency, affected_users):
    """Make predictions for a ticket"""
    
    # Combine text
    text_combined = f"{title} {description}"
    
    # Vectorize text
    text_vec = tfidf.transform([text_combined])
    
    # 1. CATEGORY PREDICTION
    category_pred = category_model.predict(text_vec)[0]
    category_proba = category_model.predict_proba(text_vec)[0]
    category_confidence = float(max(category_proba))
    
    # Get top 3 categories with probabilities
    category_classes = category_model.classes_
    top_3_cat_idx = np.argsort(category_proba)[-3:][::-1]
    top_3_categories = [(category_classes[i], category_proba[i]) for i in top_3_cat_idx]
    
    # 2. PRIORITY PREDICTION
    # Encode additional features
    impact_encoded = impact_encoder.transform([impact]).reshape(-1, 1)
    urgency_encoded = urgency_encoder.transform([urgency]).reshape(-1, 1)
    affected_encoded = np.array([[affected_users]])
    
    # Combine features for priority
    priority_features = hstack([text_vec, impact_encoded, urgency_encoded, affected_encoded])
    
    priority_pred = priority_model.predict(priority_features)[0]
    priority_proba = priority_model.predict_proba(priority_features)[0]
    priority_confidence = float(max(priority_proba))
    
    # Get top 3 priorities
    priority_classes = priority_model.classes_
    top_3_pri_idx = np.argsort(priority_proba)[-3:][::-1]
    top_3_priorities = [(priority_classes[i], priority_proba[i]) for i in top_3_pri_idx]
    
    # 3. RESOLVER GROUP PREDICTION
    # Encode category for routing
    category_encoded = category_encoder.transform([category_pred]).reshape(-1, 1)
    
    # Combine features for resolver
    resolver_features = hstack([text_vec, category_encoded, impact_encoded, urgency_encoded])
    
    resolver_pred = resolver_model.predict(resolver_features)[0]
    resolver_proba = resolver_model.predict_proba(resolver_features)[0]
    resolver_confidence = float(max(resolver_proba))
    
    # Get top 3 resolvers
    resolver_classes = resolver_model.classes_
    top_3_res_idx = np.argsort(resolver_proba)[-3:][::-1]
    top_3_resolvers = [(resolver_classes[i], resolver_proba[i]) for i in top_3_res_idx]
    
    return {
        'category': category_pred,
        'category_confidence': category_confidence,
        'top_3_categories': top_3_categories,
        'priority': priority_pred,
        'priority_confidence': priority_confidence,
        'top_3_priorities': top_3_priorities,
        'resolver': resolver_pred,
        'resolver_confidence': resolver_confidence,
        'top_3_resolvers': top_3_resolvers
    }

def confidence_label(confidence):
    """Convert confidence score to label"""
    if confidence >= 0.9:
        return "Very High ⭐⭐⭐"
    elif confidence >= 0.75:
        return "High ⭐⭐"
    elif confidence >= 0.6:
        return "Moderate ⭐"
    elif confidence >= 0.4:
        return "Low ⚠️"
    else:
        return "Very Low ⚠️⚠️"

def print_results(results, ticket_num):
    """Pretty print prediction results"""
    print("\n" + "="*80)
    print(f"PREDICTION RESULTS - TICKET #{ticket_num}")
    print("="*80)
    
    # Category
    print("\n📋 CATEGORY CLASSIFICATION")
    print("-" * 80)
    print(f"✓ Predicted Category: {results['category']}")
    print(f"  Confidence: {results['category_confidence']*100:.2f}% ({confidence_label(results['category_confidence'])})")
    print("\n  Top 3 Categories:")
    for i, (cat, prob) in enumerate(results['top_3_categories'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {i}. {cat:15s} {prob*100:5.2f}% {bar}")
    
    # Priority
    print("\n🚨 PRIORITY PREDICTION")
    print("-" * 80)
    print(f"✓ Predicted Priority: {results['priority']}")
    print(f"  Confidence: {results['priority_confidence']*100:.2f}% ({confidence_label(results['priority_confidence'])})")
    print("\n  Top 3 Priorities:")
    for i, (pri, prob) in enumerate(results['top_3_priorities'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {i}. {pri:10s} {prob*100:5.2f}% {bar}")
    
    # Resolver
    print("\n👥 RESOLVER GROUP ROUTING")
    print("-" * 80)
    print(f"✓ Assigned to: {results['resolver']}")
    print(f"  Confidence: {results['resolver_confidence']*100:.2f}% ({confidence_label(results['resolver_confidence'])})")
    print("\n  Top 3 Resolver Groups:")
    for i, (res, prob) in enumerate(results['top_3_resolvers'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {i}. {res:25s} {prob*100:5.2f}% {bar}")
    
    print("\n" + "="*80)

# Interactive Testing Loop
print("\n🎮 INTERACTIVE TESTING MODE")
print("="*80)
print("Enter ticket details to get AI predictions!")
print("Type 'quit' or 'exit' at any prompt to stop.\n")

ticket_counter = 1

while True:
    print("\n" + "─"*80)
    print(f"📝 TICKET #{ticket_counter}")
    print("─"*80)
    
    # Get title
    title = input("\n1️⃣  Enter Ticket Title: ").strip()
    if title.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Exiting test environment. Goodbye!")
        break
    
    if not title:
        print("⚠️  Title cannot be empty. Please try again.")
        continue
    
    # Get description
    description = input("2️⃣  Enter Description: ").strip()
    if description.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Exiting test environment. Goodbye!")
        break
    
    if not description:
        print("⚠️  Description cannot be empty. Please try again.")
        continue
    
    # Get impact level
    print("\n3️⃣  Select Impact Level:")
    print("   1) Low")
    print("   2) Medium")
    print("   3) High")
    impact_choice = input("   Enter choice (1-3): ").strip()
    
    if impact_choice.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Exiting test environment. Goodbye!")
        break
    
    impact_map = {'1': 'Low', '2': 'Medium', '3': 'High'}
    impact = impact_map.get(impact_choice, 'Medium')
    print(f"   Selected: {impact}")
    
    # Get urgency level
    print("\n4️⃣  Select Urgency Level:")
    print("   1) Low")
    print("   2) Medium")
    print("   3) High")
    urgency_choice = input("   Enter choice (1-3): ").strip()
    
    if urgency_choice.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Exiting test environment. Goodbye!")
        break
    
    urgency_map = {'1': 'Low', '2': 'Medium', '3': 'High'}
    urgency = urgency_map.get(urgency_choice, 'Medium')
    print(f"   Selected: {urgency}")
    
    # Get affected users
    affected_input = input("\n5️⃣  Number of affected users (default: 1): ").strip()
    
    if affected_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Exiting test environment. Goodbye!")
        break
    
    try:
        affected_users = int(affected_input) if affected_input else 1
    except ValueError:
        affected_users = 1
    
    print(f"   Affected users: {affected_users}")
    
    # Make prediction
    print("\n⏳ Analyzing ticket with AI models...")
    try:
        results = predict_ticket(title, description, impact, urgency, affected_users)
        print_results(results, ticket_counter)
        ticket_counter += 1
    except Exception as e:
        print(f"\n❌ Error making prediction: {e}")
        print("Please try again with different inputs.")
    
    # Ask if user wants to test another
    print("\n" + "─"*80)
    another = input("\n🔄 Test another ticket? (y/n): ").strip().lower()
    if another not in ['y', 'yes', '']:
        print("\n👋 Exiting test environment. Goodbye!")
        break

print("\n" + "="*80)
print("Thank you for testing the ITSM AI Models!")
print(f"Total tickets tested: {ticket_counter - 1}")
print("="*80)
