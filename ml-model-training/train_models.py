"""
Complete ML Training Pipeline for ITSM Ticketing System
Trains multiple models for classification, priority, routing, and duplicate detection
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

# Create directories for models and results
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("ITSM AI TRAINING PIPELINE - STARTING")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n[1/6] Loading dataset...")
df = pd.read_csv('synthetic_itsm_tickets.csv')
print(f"✓ Loaded {len(df):,} tickets")
print(f"  Columns: {df.shape[1]}")
print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")

# Create combined text feature for better predictions
df['text_combined'] = df['title'] + " " + df['description']
df['text_length'] = df['description'].str.len()

# Convert created_at to datetime for time-based features
df['created_at'] = pd.to_datetime(df['created_at'])
df['hour_of_day'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.dayofweek

print(f"✓ Feature engineering complete")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: SPLIT DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2/6] Splitting data (80% train / 20% test)...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ground_truth_category'])
print(f"✓ Train: {len(train_df):,} tickets")
print(f"✓ Test:  {len(test_df):,} tickets")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: TEXT VECTORIZATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3/6] Vectorizing text data (TF-IDF)...")
# Use TF-IDF for text features
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
print(f"✓ Train matrix shape: {X_train_text.shape}")
print(f"✓ Test matrix shape: {X_test_text.shape}")

# Save vectorizer
joblib.dump(tfidf, MODELS_DIR / 'tfidf_vectorizer.pkl')
print(f"✓ Saved TF-IDF vectorizer")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════

results = {}

print("\n[4/6] Training models...")
print("="*80)

# ───────────────────────────────────────────────────────────────────────────
# MODEL 1: CATEGORY CLASSIFICATION
# ───────────────────────────────────────────────────────────────────────────

print("\n🎯 MODEL 1: CATEGORY CLASSIFICATION")
print("-" * 80)

y_train_cat = train_df['ground_truth_category']
y_test_cat = test_df['ground_truth_category']

print(f"Training on {len(y_train_cat):,} samples across {y_train_cat.nunique()} categories")
print("Testing Logistic Regression...")

# Train Logistic Regression
lr_cat = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_cat.fit(X_train_text, y_train_cat)
y_pred_cat_lr = lr_cat.predict(X_test_text)
y_pred_proba_cat = lr_cat.predict_proba(X_test_text)

acc_lr = accuracy_score(y_test_cat, y_pred_cat_lr)
print(f"  Logistic Regression Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")

# Train Random Forest
print("Testing Random Forest...")
rf_cat = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_cat.fit(X_train_text, y_train_cat)
y_pred_cat_rf = rf_cat.predict(X_test_text)
acc_rf = accuracy_score(y_test_cat, y_pred_cat_rf)
print(f"  Random Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

# Choose best model
if acc_rf > acc_lr:
    best_cat_model = rf_cat
    best_cat_acc = acc_rf
    best_cat_name = "RandomForest"
    y_pred_cat = y_pred_cat_rf
else:
    best_cat_model = lr_cat
    best_cat_acc = acc_lr
    best_cat_name = "LogisticRegression"
    y_pred_cat = y_pred_cat_lr
    y_pred_proba_cat = lr_cat.predict_proba(X_test_text)

print(f"\n✓ Best model: {best_cat_name} with {best_cat_acc*100:.2f}% accuracy")
joblib.dump(best_cat_model, MODELS_DIR / 'category_classifier.pkl')

# Detailed metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test_cat, y_pred_cat, average='weighted')
results['category_classification'] = {
    'model': best_cat_name,
    'accuracy': float(best_cat_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classes': sorted(y_train_cat.unique().tolist())
}

print("\nPer-class Performance:")
report = classification_report(y_test_cat, y_pred_cat, output_dict=True)
for cat in sorted(y_train_cat.unique()):
    if cat in report:
        print(f"  {cat:15s}: F1={report[cat]['f1-score']:.3f} | Precision={report[cat]['precision']:.3f} | Recall={report[cat]['recall']:.3f} | Support={int(report[cat]['support'])}")

# Save full report
with open(RESULTS_DIR / 'category_classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ───────────────────────────────────────────────────────────────────────────
# MODEL 2: PRIORITY PREDICTION
# ───────────────────────────────────────────────────────────────────────────

print("\n🎯 MODEL 2: PRIORITY PREDICTION")
print("-" * 80)

y_train_pri = train_df['ground_truth_priority']
y_test_pri = test_df['ground_truth_priority']

print(f"Training on {len(y_train_pri):,} samples across {y_train_pri.nunique()} priority levels")

# Create additional features for priority prediction
from scipy.sparse import hstack

# Encode categorical features
impact_encoder = LabelEncoder()
urgency_encoder = LabelEncoder()

impact_train = impact_encoder.fit_transform(train_df['impact_level']).reshape(-1, 1)
impact_test = impact_encoder.transform(test_df['impact_level']).reshape(-1, 1)

urgency_train = urgency_encoder.fit_transform(train_df['urgency_level']).reshape(-1, 1)
urgency_test = urgency_encoder.transform(test_df['urgency_level']).reshape(-1, 1)

affected_train = train_df['affected_users_count'].values.reshape(-1, 1)
affected_test = test_df['affected_users_count'].values.reshape(-1, 1)

# Combine text + structured features
X_train_pri = hstack([X_train_text, impact_train, urgency_train, affected_train])
X_test_pri = hstack([X_test_text, impact_test, urgency_test, affected_test])

print("Training Gradient Boosting...")
gb_pri = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_pri.fit(X_train_pri, y_train_pri)
y_pred_pri = gb_pri.predict(X_test_pri)
y_pred_proba_pri = gb_pri.predict_proba(X_test_pri)

acc_pri = accuracy_score(y_test_pri, y_pred_pri)
print(f"  Gradient Boosting Accuracy: {acc_pri:.4f} ({acc_pri*100:.2f}%)")

# Save model and encoders
joblib.dump(gb_pri, MODELS_DIR / 'priority_predictor.pkl')
joblib.dump(impact_encoder, MODELS_DIR / 'impact_encoder.pkl')
joblib.dump(urgency_encoder, MODELS_DIR / 'urgency_encoder.pkl')
print(f"\n✓ Saved priority model and encoders")

precision, recall, f1, support = precision_recall_fscore_support(y_test_pri, y_pred_pri, average='weighted')
results['priority_prediction'] = {
    'model': 'GradientBoosting',
    'accuracy': float(acc_pri),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classes': sorted(y_train_pri.unique().tolist())
}

print("\nPer-class Performance:")
report_pri = classification_report(y_test_pri, y_pred_pri, output_dict=True)
priority_order = ['Low', 'Medium', 'High', 'Critical']
for pri in priority_order:
    if pri in report_pri:
        print(f"  {pri:10s}: F1={report_pri[pri]['f1-score']:.3f} | Precision={report_pri[pri]['precision']:.3f} | Recall={report_pri[pri]['recall']:.3f} | Support={int(report_pri[pri]['support'])}")

with open(RESULTS_DIR / 'priority_prediction_report.json', 'w') as f:
    json.dump(report_pri, f, indent=2)

# ───────────────────────────────────────────────────────────────────────────
# MODEL 3: RESOLVER GROUP ROUTING
# ───────────────────────────────────────────────────────────────────────────

print("\n🎯 MODEL 3: RESOLVER GROUP ROUTING")
print("-" * 80)

y_train_res = train_df['ground_truth_resolver_group']
y_test_res = test_df['ground_truth_resolver_group']

print(f"Training on {len(y_train_res):,} samples across {y_train_res.nunique()} resolver groups")

# Encode category for routing
category_encoder = LabelEncoder()
cat_train = category_encoder.fit_transform(train_df['ground_truth_category']).reshape(-1, 1)
cat_test = category_encoder.transform(test_df['ground_truth_category']).reshape(-1, 1)

# Combine features
X_train_res = hstack([X_train_text, cat_train, impact_train, urgency_train])
X_test_res = hstack([X_test_text, cat_test, impact_test, urgency_test])

print("Training Random Forest for routing...")
rf_res = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_res.fit(X_train_res, y_train_res)
y_pred_res = rf_res.predict(X_test_res)
y_pred_proba_res = rf_res.predict_proba(X_test_res)

acc_res = accuracy_score(y_test_res, y_pred_res)
print(f"  Random Forest Accuracy: {acc_res:.4f} ({acc_res*100:.2f}%)")

joblib.dump(rf_res, MODELS_DIR / 'resolver_router.pkl')
joblib.dump(category_encoder, MODELS_DIR / 'category_encoder.pkl')
print(f"\n✓ Saved routing model and encoder")

precision, recall, f1, support = precision_recall_fscore_support(y_test_res, y_pred_res, average='weighted')
results['resolver_routing'] = {
    'model': 'RandomForest',
    'accuracy': float(acc_res),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classes': sorted(y_train_res.unique().tolist())
}

print("\nPer-class Performance:")
report_res = classification_report(y_test_res, y_pred_res, output_dict=True)
for res in sorted(y_train_res.unique()):
    if res in report_res:
        print(f"  {res:20s}: F1={report_res[res]['f1-score']:.3f} | Precision={report_res[res]['precision']:.3f} | Recall={report_res[res]['recall']:.3f} | Support={int(report_res[res]['support'])}")

with open(RESULTS_DIR / 'resolver_routing_report.json', 'w') as f:
    json.dump(report_res, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: DUPLICATE DETECTION (SENTENCE EMBEDDINGS)
# ═══════════════════════════════════════════════════════════════════════════

print("\n🎯 MODEL 4: DUPLICATE DETECTION")
print("-" * 80)

print("Loading sentence-transformers model (this may take a moment)...")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Use a lightweight but effective model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(f"✓ Loaded embedding model: all-MiniLM-L6-v2")

# For demo purposes, create some duplicate pairs
print("\nGenerating test duplicates for evaluation...")
# Take 100 random tickets and create near-duplicates
sample_indices = np.random.choice(len(test_df), 100, replace=False)
sample_tickets = test_df.iloc[sample_indices].copy()

# Create embeddings for sample
sample_embeddings = embedder.encode(sample_tickets['text_combined'].tolist(), show_progress_bar=True)
print(f"✓ Generated {len(sample_embeddings)} embeddings")

# Calculate similarity matrix
similarity_matrix = cosine_similarity(sample_embeddings)
print(f"✓ Computed similarity matrix: {similarity_matrix.shape}")

# Find potential duplicates (similarity > 0.75)
threshold = 0.75
duplicate_pairs = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            duplicate_pairs.append({
                'ticket_1': sample_tickets.iloc[i]['ticket_id'],
                'ticket_2': sample_tickets.iloc[j]['ticket_id'],
                'similarity': float(similarity_matrix[i][j]),
                'title_1': sample_tickets.iloc[i]['title'][:50],
                'title_2': sample_tickets.iloc[j]['title'][:50]
            })

print(f"\n✓ Found {len(duplicate_pairs)} potential duplicate pairs (similarity > {threshold})")
if len(duplicate_pairs) > 0:
    print("\nTop 5 duplicate pairs:")
    for i, pair in enumerate(sorted(duplicate_pairs, key=lambda x: x['similarity'], reverse=True)[:5]):
        print(f"  {i+1}. {pair['ticket_1']} ↔ {pair['ticket_2']}: {pair['similarity']:.3f}")
        print(f"     '{pair['title_1']}...' ↔ '{pair['title_2']}...'")

# Save embedding model reference
results['duplicate_detection'] = {
    'model': 'all-MiniLM-L6-v2',
    'embedding_dim': int(sample_embeddings.shape[1]),
    'similarity_threshold': threshold,
    'test_duplicates_found': len(duplicate_pairs)
}

# Save sample embeddings for testing
joblib.dump(sample_embeddings[:10], MODELS_DIR / 'sample_embeddings.pkl')
joblib.dump(sample_tickets.iloc[:10], MODELS_DIR / 'sample_tickets.pkl')
print(f"✓ Saved sample embeddings for API testing")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE COMPREHENSIVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[5/6] Saving comprehensive results...")

# Add overall summary
results['summary'] = {
    'training_date': datetime.now().isoformat(),
    'total_samples': len(df),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'models_trained': 4,
    'overall_performance': {
        'category_accuracy': results['category_classification']['accuracy'],
        'priority_accuracy': results['priority_prediction']['accuracy'],
        'routing_accuracy': results['resolver_routing']['accuracy']
    }
}

# Save results
with open(RESULTS_DIR / 'training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved training results to {RESULTS_DIR / 'training_results.json'}")

# Create a test predictions file with confidence scores
print("\n[6/6] Creating test predictions with confidence scores...")

test_predictions = []
for idx in range(min(100, len(test_df))):  # Save 100 test predictions
    row = test_df.iloc[idx]
    
    # Get predictions
    text_vec = tfidf.transform([row['text_combined']])
    
    cat_pred = best_cat_model.predict(text_vec)[0]
    cat_proba = best_cat_model.predict_proba(text_vec)[0]
    cat_confidence = float(max(cat_proba))
    
    test_predictions.append({
        'ticket_id': row['ticket_id'],
        'title': row['title'][:80],
        'true_category': row['ground_truth_category'],
        'predicted_category': cat_pred,
        'category_confidence': cat_confidence,
        'true_priority': row['ground_truth_priority'],
        'true_resolver': row['ground_truth_resolver_group']
    })

pd.DataFrame(test_predictions).to_csv(RESULTS_DIR / 'test_predictions_sample.csv', index=False)
print(f"✓ Saved 100 test predictions with confidence scores")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*80)

print(f"\n📊 DATASET")
print(f"  Total tickets: {len(df):,}")
print(f"  Train split: {len(train_df):,} (80%)")
print(f"  Test split: {len(test_df):,} (20%)")

print(f"\n🎯 MODEL PERFORMANCE")
print(f"  1. Category Classification: {results['category_classification']['accuracy']*100:.2f}% accuracy")
print(f"     Model: {results['category_classification']['model']}")
print(f"     F1 Score: {results['category_classification']['f1_score']:.4f}")

print(f"\n  2. Priority Prediction: {results['priority_prediction']['accuracy']*100:.2f}% accuracy")
print(f"     Model: {results['priority_prediction']['model']}")
print(f"     F1 Score: {results['priority_prediction']['f1_score']:.4f}")

print(f"\n  3. Resolver Routing: {results['resolver_routing']['accuracy']*100:.2f}% accuracy")
print(f"     Model: {results['resolver_routing']['model']}")
print(f"     F1 Score: {results['resolver_routing']['f1_score']:.4f}")

print(f"\n  4. Duplicate Detection: Embedding-based")
print(f"     Model: {results['duplicate_detection']['model']}")
print(f"     Test duplicates found: {results['duplicate_detection']['test_duplicates_found']}")

print(f"\n💾 SAVED ARTIFACTS")
print(f"  Models: {len(list(MODELS_DIR.glob('*.pkl')))} files in {MODELS_DIR}/")
print(f"  Results: {len(list(RESULTS_DIR.glob('*')))} files in {RESULTS_DIR}/")

print("\n✅ All models trained and saved successfully!")
print("="*80)
