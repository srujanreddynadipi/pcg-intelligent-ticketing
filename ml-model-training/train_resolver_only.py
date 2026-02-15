"""
Train ONLY the Resolver Routing Model
Uses the new 100K dataset with proper category→resolver mapping
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack
import json

print("="*80)
print("TRAINING RESOLVER ROUTING MODEL")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n📂 STEP 1: Loading dataset...")
df = pd.read_csv('synthetic_itsm_tickets.csv')
print(f"   Total tickets: {len(df):,}")
print(f"   Categories: {df['category'].nunique()}")
print(f"   Resolver groups: {df['ground_truth_resolver_group'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: PREPARE FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔧 STEP 2: Preparing features...")

# Combine title and description
df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

# Extract keyword features
df['has_network_keyword'] = df['text'].str.lower().str.contains(
    'network|vpn|wifi|dns|firewall|router|switch|connection|connectivity', 
    case=False, na=False
).astype(int)

df['has_hardware_keyword'] = df['text'].str.lower().str.contains(
    'laptop|desktop|computer|monitor|keyboard|mouse|printer|hardware|device', 
    case=False, na=False
).astype(int)

df['has_database_keyword'] = df['text'].str.lower().str.contains(
    'database|sql|query|db|table|replication|backup|connection pool', 
    case=False, na=False
).astype(int)

df['has_cloud_keyword'] = df['text'].str.lower().str.contains(
    'azure|aws|cloud|vm|container|kubernetes|docker|s3|blob', 
    case=False, na=False
).astype(int)

df['has_security_keyword'] = df['text'].str.lower().str.contains(
    'security|malware|virus|phishing|breach|unauthorized|certificate|firewall', 
    case=False, na=False
).astype(int)

df['has_devops_keyword'] = df['text'].str.lower().str.contains(
    'cicd|pipeline|jenkins|git|docker|kubernetes|terraform|helm|deployment', 
    case=False, na=False
).astype(int)

df['has_email_keyword'] = df['text'].str.lower().str.contains(
    'email|outlook|mailbox|exchange|mail|inbox|outbox|smtp', 
    case=False, na=False
).astype(int)

print(f"   ✓ Text feature created")
print(f"   ✓ Keyword features created (7 categories)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: VECTORIZE TEXT
# ═══════════════════════════════════════════════════════════════════════════
print("\n📝 STEP 3: Vectorizing text...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

text_vectors = tfidf_vectorizer.fit_transform(df['text'])
print(f"   ✓ TF-IDF vectors: {text_vectors.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: ENCODE CATEGORICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n🏷️  STEP 4: Encoding categorical features...")

# Encode category (most important feature for resolver routing!)
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['category'])

# Encode impact and urgency
impact_encoder = LabelEncoder()
df['impact_encoded'] = impact_encoder.fit_transform(df['impact'])

urgency_encoder = LabelEncoder()
df['urgency_encoded'] = urgency_encoder.fit_transform(df['urgency'])

print(f"   ✓ Category encoded: {len(category_encoder.classes_)} classes")
print(f"   ✓ Impact encoded: {len(impact_encoder.classes_)} classes")
print(f"   ✓ Urgency encoded: {len(urgency_encoder.classes_)} classes")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: COMBINE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔗 STEP 5: Combining features...")

# Keywords
keyword_features = df[[
    'has_network_keyword', 'has_hardware_keyword', 'has_database_keyword',
    'has_cloud_keyword', 'has_security_keyword', 'has_devops_keyword',
    'has_email_keyword'
]].values

# Categorical features
categorical_features = df[[
    'category_encoded', 'impact_encoded', 'urgency_encoded', 'affected_users'
]].values

# Combine all features
X = hstack([
    text_vectors,                      # TF-IDF features (5000 dims)
    categorical_features,              # Category, impact, urgency, users (4 dims)
    keyword_features                   # Keyword flags (7 dims)
])

y = df['ground_truth_resolver_group']

print(f"   ✓ Combined features shape: {X.shape}")
print(f"   ✓ Target classes: {y.nunique()}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: SPLIT DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n✂️  STEP 6: Splitting train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"   ✓ Training set: {X_train.shape[0]:,} samples")
print(f"   ✓ Test set: {X_test.shape[0]:,} samples")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: TRAIN RESOLVER ROUTING MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n🤖 STEP 7: Training Resolver Routing Model...")
print("   Model: Random Forest")
print("   Parameters: 200 trees, max_depth=30, balanced weights")

resolver_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

resolver_model.fit(X_train, y_train)
print("\n   ✅ Model training complete!")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: EVALUATE MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n📊 STEP 8: Evaluating model performance...")

y_pred_train = resolver_model.predict(X_train)
y_pred_test = resolver_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n   Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_test, zero_division=0))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
classes = sorted(y.unique())
print(f"\n{'':20s}", end='')
for cls in classes:
    print(f"{cls[:15]:>15s}", end='')
print()
print("─"*100)
for i, cls in enumerate(classes):
    print(f"{cls[:20]:20s}", end='')
    for j in range(len(classes)):
        print(f"{cm[i][j]:15d}", end='')
    print()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n📈 STEP 9: Feature importance analysis...")

# Get feature names
tfidf_features = tfidf_vectorizer.get_feature_names_out()
categorical_feature_names = ['category', 'impact', 'urgency', 'affected_users']
keyword_feature_names = [
    'network_keyword', 'hardware_keyword', 'database_keyword',
    'cloud_keyword', 'security_keyword', 'devops_keyword', 'email_keyword'
]
all_feature_names = list(tfidf_features) + categorical_feature_names + keyword_feature_names

# Get top 15 important features
feature_importance = resolver_model.feature_importances_
top_indices = np.argsort(feature_importance)[-15:][::-1]

print("\n   Top 15 Most Important Features:")
for idx in top_indices:
    if idx < len(all_feature_names):
        print(f"      {all_feature_names[idx][:40]:40s}: {feature_importance[idx]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════
print("\n💾 STEP 10: Saving models...")

# Create models directory
models_dir = Path("models_improved")
models_dir.mkdir(exist_ok=True)

# Save resolver model and dependencies
joblib.dump(resolver_model, models_dir / "resolver_router.pkl")
joblib.dump(tfidf_vectorizer, models_dir / "tfidf_vectorizer.pkl")
joblib.dump(category_encoder, models_dir / "category_encoder.pkl")
joblib.dump(impact_encoder, models_dir / "impact_encoder.pkl")
joblib.dump(urgency_encoder, models_dir / "urgency_encoder.pkl")

print(f"   ✅ Saved: resolver_router.pkl ({(models_dir / 'resolver_router.pkl').stat().st_size / 1024 / 1024:.2f} MB)")
print(f"   ✅ Saved: tfidf_vectorizer.pkl")
print(f"   ✅ Saved: category_encoder.pkl")
print(f"   ✅ Saved: impact_encoder.pkl")
print(f"   ✅ Saved: urgency_encoder.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: SAVE TRAINING RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n📄 STEP 11: Saving training results...")

results_dir = Path("results_improved")
results_dir.mkdir(exist_ok=True)

results = {
    "model": "Resolver Routing",
    "algorithm": "Random Forest",
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_size": len(df),
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "categories": len(df['category'].unique()),
    "resolver_groups": len(y.unique()),
    "train_accuracy": float(train_accuracy),
    "test_accuracy": float(test_accuracy),
    "features": {
        "tfidf_dimensions": text_vectors.shape[1],
        "categorical_features": 4,
        "keyword_features": 7,
        "total_features": X.shape[1]
    },
    "classes": sorted(y.unique().tolist()),
    "classification_report": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
}

with open(results_dir / "resolver_training_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✅ Saved: resolver_training_results.json")

# Create summary report
report_file = results_dir / "RESOLVER_TRAINING_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RESOLVER ROUTING MODEL - TRAINING REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: synthetic_itsm_tickets.csv\n")
    f.write(f"Total Tickets: {len(df):,}\n\n")
    
    f.write("="*80 + "\n")
    f.write("MODEL CONFIGURATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Algorithm: Random Forest\n")
    f.write(f"Trees: 200\n")
    f.write(f"Max Depth: 30\n")
    f.write(f"Class Weight: Balanced\n\n")
    
    f.write("="*80 + "\n")
    f.write("FEATURES\n")
    f.write("="*80 + "\n\n")
    f.write(f"TF-IDF Features: {text_vectors.shape[1]:,}\n")
    f.write(f"Categorical: category, impact, urgency, affected_users\n")
    f.write(f"Keywords: network, hardware, database, cloud, security, devops, email\n")
    f.write(f"Total Features: {X.shape[1]:,}\n\n")
    
    f.write("="*80 + "\n")
    f.write("RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n\n")
    
    f.write("Resolver Groups:\n")
    for cls in sorted(y.unique()):
        count = (y_test == cls).sum()
        correct = ((y_test == cls) & (y_pred_test == cls)).sum()
        acc = correct / count * 100 if count > 0 else 0
        f.write(f"  {cls:20s}: {correct:4d}/{count:4d} ({acc:5.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_test, y_pred_test, zero_division=0))
    
    f.write("\n" + "="*80 + "\n")
    f.write("CATEGORY → RESOLVER MAPPING VALIDATION\n")
    f.write("="*80 + "\n\n")
    
    for cat in sorted(df['category'].unique()):
        resolver = df[df['category'] == cat]['ground_truth_resolver_group'].iloc[0]
        f.write(f"{cat:20s} → {resolver}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("✅ TRAINING COMPLETE!\n")
    f.write("="*80 + "\n")

print(f"   ✅ Saved: RESOLVER_TRAINING_REPORT.txt")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("✅ RESOLVER ROUTING MODEL TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Final Results:")
print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Resolver Groups: {len(y.unique())}")
print(f"   Total Features: {X.shape[1]:,}")

print(f"\n💾 Saved Files:")
print(f"   models_improved/resolver_router.pkl")
print(f"   models_improved/tfidf_vectorizer.pkl")
print(f"   models_improved/category_encoder.pkl")
print(f"   models_improved/impact_encoder.pkl")
print(f"   models_improved/urgency_encoder.pkl")
print(f"   results_improved/resolver_training_results.json")
print(f"   results_improved/RESOLVER_TRAINING_REPORT.txt")

if test_accuracy >= 0.95:
    print(f"\n🎉 EXCELLENT! Test accuracy >= 95%")
elif test_accuracy >= 0.90:
    print(f"\n✅ GOOD! Test accuracy >= 90%")
elif test_accuracy >= 0.80:
    print(f"\n⚠️  ACCEPTABLE! Test accuracy >= 80%")
else:
    print(f"\n❌ NEEDS IMPROVEMENT! Test accuracy < 80%")

print("\n" + "="*80)
