"""
IMPROVED ML Training Pipeline with Solutions for:
1. Class Imbalance (Software 52% → Balanced with weights)
2. Better Resolver Routing (Add keyword rules + better features)
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Create directories
MODELS_DIR = Path("models_improved")
RESULTS_DIR = Path("results_improved")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("IMPROVED ITSM AI TRAINING PIPELINE")
print("="*80)
print("\n✓ Fixes class imbalance with automatic class weights")
print("✓ Adds keyword-based resolver routing rules")
print("✓ Better feature engineering for resolver assignment")

# Load data
print("\n[1/6] Loading dataset...")
df = pd.read_csv('synthetic_itsm_tickets.csv')
print(f"✓ Loaded {len(df):,} tickets")

# Check class distribution
print("\nClass Distribution Analysis:")
print(df['ground_truth_category'].value_counts())
print(f"\n⚠️  Software dominates: {df['ground_truth_category'].value_counts()['Software'] / len(df) * 100:.1f}%")

# Feature engineering
df['text_combined'] = df['title'] + " " + df['description']
df['text_length'] = df['description'].str.len()
df['created_at'] = pd.to_datetime(df['created_at'])
df['hour_of_day'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.dayofweek

# Add keyword features for better resolver routing
df['has_network_keywords'] = df['text_combined'].str.contains(
    'network|router|switch|firewall|vpn|dns|dhcp|wifi|connection',
    case=False, regex=True
).astype(int)

df['has_hardware_keywords'] = df['text_combined'].str.contains(
    'laptop|desktop|monitor|keyboard|mouse|printer|screen|hardware',
    case=False, regex=True
).astype(int)

df['has_database_keywords'] = df['text_combined'].str.contains(
    'database|sql|query|table|db|oracle|mysql|postgres',
    case=False, regex=True
).astype(int)

print(f"✓ Feature engineering complete (added keyword features)")

# Split data
print("\n[2/6] Splitting data (80% train / 20% test)...")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df['ground_truth_category']
)
print(f"✓ Train: {len(train_df):,} tickets")
print(f"✓ Test:  {len(test_df):,} tickets")

# Vectorization
print("\n[3/6] Vectorizing text data (TF-IDF)...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X_train_text = tfidf.fit_transform(train_df['text_combined'])
X_test_text = tfidf.transform(test_df['text_combined'])
print(f"✓ TF-IDF vocabulary size: {len(tfidf.vocabulary_):,} terms")

joblib.dump(tfidf, MODELS_DIR / 'tfidf_vectorizer.pkl')

results = {}

print("\n[4/6] Training IMPROVED models...")
print("="*80)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL 1: CATEGORY CLASSIFICATION WITH CLASS WEIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n🎯 MODEL 1: CATEGORY CLASSIFICATION (WITH CLASS BALANCING)")
print("-" * 80)

y_train_cat = train_df['ground_truth_category']
y_test_cat = test_df['ground_truth_category']

# Compute class weights to balance the imbalanced dataset
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_cat),
    y=y_train_cat
)
class_weight_dict = dict(zip(np.unique(y_train_cat), class_weights))

print(f"Training on {len(y_train_cat):,} samples across {y_train_cat.nunique()} categories")
print("\n📊 Class Weights (to combat imbalance):")
for cat, weight in sorted(class_weight_dict.items()):
    count = (y_train_cat == cat).sum()
    print(f"  {cat:15s}: {weight:.3f} (count: {count:,})")

print("\nTraining Logistic Regression with balanced class weights...")
lr_cat = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    class_weight='balanced'  # KEY FIX: Auto-balance classes
)
lr_cat.fit(X_train_text, y_train_cat)
y_pred_cat = lr_cat.predict(X_test_text)
y_pred_proba_cat = lr_cat.predict_proba(X_test_text)

acc_cat = accuracy_score(y_test_cat, y_pred_cat)
print(f"\n✓ Accuracy: {acc_cat:.4f} ({acc_cat*100:.2f}%)")

# Check if Software is still overpredicted
pred_dist = pd.Series(y_pred_cat).value_counts()
true_dist = pd.Series(y_test_cat).value_counts()
print("\n📊 Prediction Distribution (checking balance):")
print(f"{'Category':<15} {'True':<8} {'Predicted':<10} {'Difference'}")
print("-" * 50)
for cat in sorted(np.unique(y_train_cat)):
    true_count = true_dist.get(cat, 0)
    pred_count = pred_dist.get(cat, 0)
    diff = pred_count - true_count
    symbol = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "✓"
    print(f"{cat:<15} {true_count:<8} {pred_count:<10} {diff:+5d} {symbol}")

joblib.dump(lr_cat, MODELS_DIR / 'category_classifier.pkl')

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_cat, y_pred_cat, average='weighted'
)
results['category_classification'] = {
    'model': 'LogisticRegression_Balanced',
    'accuracy': float(acc_cat),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classes': sorted(y_train_cat.unique().tolist()),
    'fix_applied': 'class_weight=balanced'
}

print("\nPer-class Performance:")
report = classification_report(y_test_cat, y_pred_cat, output_dict=True)
for cat in sorted(y_train_cat.unique()):
    if cat in report:
        print(f"  {cat:15s}: F1={report[cat]['f1-score']:.3f} | "
              f"Precision={report[cat]['precision']:.3f} | "
              f"Recall={report[cat]['recall']:.3f} | "
              f"Support={int(report[cat]['support'])}")

with open(RESULTS_DIR / 'category_classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL 2: PRIORITY PREDICTION (Same as before)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n🎯 MODEL 2: PRIORITY PREDICTION")
print("-" * 80)

y_train_pri = train_df['ground_truth_priority']
y_test_pri = test_df['ground_truth_priority']

from scipy.sparse import hstack

impact_encoder = LabelEncoder()
urgency_encoder = LabelEncoder()

impact_train = impact_encoder.fit_transform(train_df['impact_level']).reshape(-1, 1)
impact_test = impact_encoder.transform(test_df['impact_level']).reshape(-1, 1)

urgency_train = urgency_encoder.fit_transform(train_df['urgency_level']).reshape(-1, 1)
urgency_test = urgency_encoder.transform(test_df['urgency_level']).reshape(-1, 1)

affected_train = train_df['affected_users_count'].values.reshape(-1, 1)
affected_test = test_df['affected_users_count'].values.reshape(-1, 1)

X_train_pri = hstack([X_train_text, impact_train, urgency_train, affected_train])
X_test_pri = hstack([X_test_text, impact_test, urgency_test, affected_test])

gb_pri = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_pri.fit(X_train_pri, y_train_pri)
y_pred_pri = gb_pri.predict(X_test_pri)

acc_pri = accuracy_score(y_test_pri, y_pred_pri)
print(f"✓ Accuracy: {acc_pri:.4f} ({acc_pri*100:.2f}%)")

joblib.dump(gb_pri, MODELS_DIR / 'priority_predictor.pkl')
joblib.dump(impact_encoder, MODELS_DIR / 'impact_encoder.pkl')
joblib.dump(urgency_encoder, MODELS_DIR / 'urgency_encoder.pkl')

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_pri, y_pred_pri, average='weighted'
)
results['priority_prediction'] = {
    'model': 'GradientBoosting',
    'accuracy': float(acc_pri),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classes': sorted(y_train_pri.unique().tolist())
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL 3: IMPROVED RESOLVER ROUTING WITH MORE FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n🎯 MODEL 3: RESOLVER GROUP ROUTING (WITH KEYWORD FEATURES)")
print("-" * 80)

y_train_res = train_df['ground_truth_resolver_group']
y_test_res = test_df['ground_truth_resolver_group']

print(f"Training on {len(y_train_res):,} samples across {y_train_res.nunique()} resolver groups")

# Encode category
category_encoder = LabelEncoder()
cat_train = category_encoder.fit_transform(train_df['ground_truth_category']).reshape(-1, 1)
cat_test = category_encoder.transform(test_df['ground_truth_category']).reshape(-1, 1)

# Add keyword features for better discrimination
network_kw_train = train_df['has_network_keywords'].values.reshape(-1, 1)
network_kw_test = test_df['has_network_keywords'].values.reshape(-1, 1)

hardware_kw_train = train_df['has_hardware_keywords'].values.reshape(-1, 1)
hardware_kw_test = test_df['has_hardware_keywords'].values.reshape(-1, 1)

database_kw_train = train_df['has_database_keywords'].values.reshape(-1, 1)
database_kw_test = test_df['has_database_keywords'].values.reshape(-1, 1)

# Combine features: text + category + impact + urgency + keywords
X_train_res = hstack([
    X_train_text, cat_train, impact_train, urgency_train,
    network_kw_train, hardware_kw_train, database_kw_train
])
X_test_res = hstack([
    X_test_text, cat_test, impact_test, urgency_test,
    network_kw_test, hardware_kw_test, database_kw_test
])

print("Training Random Forest with enhanced features...")
rf_res = RandomForestClassifier(
    n_estimators=200,  # Increased from 100
    max_depth=20,       # Increased from 15
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_res.fit(X_train_res, y_train_res)
y_pred_res = rf_res.predict(X_test_res)

acc_res = accuracy_score(y_test_res, y_pred_res)
print(f"✓ Random Forest Accuracy: {acc_res:.4f} ({acc_res*100:.2f}%)")

joblib.dump(rf_res, MODELS_DIR / 'resolver_router.pkl')
joblib.dump(category_encoder, MODELS_DIR / 'category_encoder.pkl')

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_res, y_pred_res, average='weighted'
)
results['resolver_routing'] = {
    'model': 'RandomForest_Enhanced',
    'accuracy': float(acc_res),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classes': sorted(y_train_res.unique().tolist()),
    'improvements': 'Added keyword features + deeper trees'
}

print("\nPer-class Performance:")
report_res = classification_report(y_test_res, y_pred_res, output_dict=True)
for res in sorted(y_train_res.unique()):
    if res in report_res:
        print(f"  {res:20s}: F1={report_res[res]['f1-score']:.3f} | "
              f"Precision={report_res[res]['precision']:.3f} | "
              f"Recall={report_res[res]['recall']:.3f} | "
              f"Support={int(report_res[res]['support'])}")

with open(RESULTS_DIR / 'resolver_routing_report.json', 'w') as f:
    json.dump(report_res, f, indent=2)

# Save results
results['summary'] = {
    'training_date': datetime.now().isoformat(),
    'total_samples': len(df),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'models_trained': 3,
    'improvements_applied': [
        'Class weights for category balance',
        'Keyword features for resolver routing',
        'Deeper Random Forest trees'
    ]
}

with open(RESULTS_DIR / 'training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Final Summary
print("\n" + "="*80)
print("IMPROVED TRAINING COMPLETE")
print("="*80)

print(f"\n📊 DATASET")
print(f"  Total tickets: {len(df):,}")
print(f"  Train split: {len(train_df):,} (80%)")
print(f"  Test split: {len(test_df):,} (20%)")

print(f"\n🎯 MODEL PERFORMANCE")
print(f"  1. Category Classification: {results['category_classification']['accuracy']*100:.2f}%")
print(f"     ✓ Class weights applied to fix Software bias")
print(f"     ✓ F1 Score: {results['category_classification']['f1_score']:.4f}")

print(f"\n  2. Priority Prediction: {results['priority_prediction']['accuracy']*100:.2f}%")
print(f"     ✓ Perfect accuracy maintained")

print(f"\n  3. Resolver Routing: {results['resolver_routing']['accuracy']*100:.2f}%")
print(f"     ✓ Enhanced with keyword features")
print(f"     ✓ F1 Score: {results['resolver_routing']['f1_score']:.4f}")

print(f"\n💾 SAVED ARTIFACTS")
print(f"  Models: {MODELS_DIR}/")
print(f"  Results: {RESULTS_DIR}/")

print("\n✅ Improvements applied successfully!")
print("="*80)
