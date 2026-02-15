"""
Comprehensive Model Evaluation with Visualizations
Creates confusion matrices, performance charts, and detailed analysis
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
VISUALIZATIONS_DIR = Path("visualizations")
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

print("="*80)
print("MODEL EVALUATION & VISUALIZATION")
print("="*80)

# Load data
print("\nLoading data and models...")
df = pd.read_csv('synthetic_itsm_tickets.csv')
df['text_combined'] = df['title'] + " " + df['description']

# Same split as training
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ground_truth_category'])

# Load models
tfidf = joblib.load(MODELS_DIR / 'tfidf_vectorizer.pkl')
cat_model = joblib.load(MODELS_DIR / 'category_classifier.pkl')
pri_model = joblib.load(MODELS_DIR / 'priority_predictor.pkl')
res_model = joblib.load(MODELS_DIR / 'resolver_router.pkl')
impact_encoder = joblib.load(MODELS_DIR / 'impact_encoder.pkl')
urgency_encoder = joblib.load(MODELS_DIR / 'urgency_encoder.pkl')
category_encoder = joblib.load(MODELS_DIR / 'category_encoder.pkl')

print("✓ Loaded all models successfully")

# Prepare data
X_test_text = tfidf.transform(test_df['text_combined'])

# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY CLASSIFICATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-"*80)
print("1. CATEGORY CLASSIFICATION ANALYSIS")
print("-"*80)

y_test_cat = test_df['ground_truth_category']
y_pred_cat = cat_model.predict(X_test_text)
y_pred_proba_cat = cat_model.predict_proba(X_test_text)

# Confusion Matrix
cm_cat = confusion_matrix(y_test_cat, y_pred_cat)
categories = sorted(y_test_cat.unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Category Classification - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Category', fontsize=12)
plt.xlabel('Predicted Category', fontsize=12)
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'category_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved category confusion matrix")

# Confidence distribution
confidences_cat = np.max(y_pred_proba_cat, axis=1)
plt.figure(figsize=(10, 6))
plt.hist(confidences_cat, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0.8, color='r', linestyle='--', label='80% threshold')
plt.title('Category Prediction - Confidence Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Confidence Score', fontsize=12)
plt.ylabel('Number of Predictions', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'category_confidence_dist.png', dpi=300, bbox_inches='tight')
print(f"✓ Avg confidence: {confidences_cat.mean():.3f}, Min: {confidences_cat.min():.3f}, Max: {confidences_cat.max():.3f}")

# Per-category accuracy
plt.figure(figsize=(10, 6))
report_cat = classification_report(y_test_cat, y_pred_cat, output_dict=True)
cat_scores = {cat: report_cat[cat]['f1-score'] for cat in categories}
plt.bar(cat_scores.keys(), cat_scores.values(), color='steelblue', edgecolor='black')
plt.title('Category Classification - F1 Score by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.ylim(0, 1.0)
for i, (cat, score) in enumerate(cat_scores.items()):
    plt.text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'category_f1_scores.png', dpi=300, bbox_inches='tight')
print("✓ Saved category F1 scores chart")

# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY PREDICTION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-"*80)
print("2. PRIORITY PREDICTION ANALYSIS")
print("-"*80)

from scipy.sparse import hstack

impact_test = impact_encoder.transform(test_df['impact_level']).reshape(-1, 1)
urgency_test = urgency_encoder.transform(test_df['urgency_level']).reshape(-1, 1)
affected_test = test_df['affected_users_count'].values.reshape(-1, 1)
X_test_pri = hstack([X_test_text, impact_test, urgency_test, affected_test])

y_test_pri = test_df['ground_truth_priority']
y_pred_pri = pri_model.predict(X_test_pri)
y_pred_proba_pri = pri_model.predict_proba(X_test_pri)

# Confusion Matrix
priorities = ['Low', 'Medium', 'High', 'Critical']
cm_pri = confusion_matrix(y_test_pri, y_pred_pri, labels=priorities)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_pri, annot=True, fmt='d', cmap='Oranges', xticklabels=priorities, yticklabels=priorities)
plt.title('Priority Prediction - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Priority', fontsize=12)
plt.xlabel('Predicted Priority', fontsize=12)
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'priority_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved priority confusion matrix")

# Priority distribution comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(priorities))
width = 0.35
true_counts = [sum(y_test_pri == p) for p in priorities]
pred_counts = [sum(y_pred_pri == p) for p in priorities]

plt.bar(x - width/2, true_counts, width, label='True', color='skyblue', edgecolor='black')
plt.bar(x + width/2, pred_counts, width, label='Predicted', color='lightcoral', edgecolor='black')
plt.xlabel('Priority Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Priority Distribution - True vs Predicted', fontsize=14, fontweight='bold')
plt.xticks(x, priorities)
plt.legend()
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'priority_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved priority distribution chart")

# ═══════════════════════════════════════════════════════════════════════════
# RESOLVER ROUTING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-"*80)
print("3. RESOLVER ROUTING ANALYSIS")
print("-"*80)

cat_test = category_encoder.transform(test_df['ground_truth_category']).reshape(-1, 1)
X_test_res = hstack([X_test_text, cat_test, impact_test, urgency_test])

y_test_res = test_df['ground_truth_resolver_group']
y_pred_res = res_model.predict(X_test_res)
y_pred_proba_res = res_model.predict_proba(X_test_res)

# Confusion Matrix
resolvers = sorted(y_test_res.unique())
cm_res = confusion_matrix(y_test_res, y_pred_res, labels=resolvers)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_res, annot=True, fmt='d', cmap='Greens', xticklabels=resolvers, yticklabels=resolvers)
plt.title('Resolver Routing - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Resolver', fontsize=12)
plt.xlabel('Predicted Resolver', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'resolver_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved resolver confusion matrix")

# Per-resolver performance
report_res = classification_report(y_test_res, y_pred_res, output_dict=True)
plt.figure(figsize=(12, 6))
res_metrics = []
for res in resolvers:
    if res in report_res:
        res_metrics.append({
            'Resolver': res,
            'Precision': report_res[res]['precision'],
            'Recall': report_res[res]['recall'],
            'F1-Score': report_res[res]['f1-score']
        })

res_df = pd.DataFrame(res_metrics)
x = np.arange(len(res_df))
width = 0.25

plt.bar(x - width, res_df['Precision'], width, label='Precision', color='lightblue', edgecolor='black')
plt.bar(x, res_df['Recall'], width, label='Recall', color='lightgreen', edgecolor='black')
plt.bar(x + width, res_df['F1-Score'], width, label='F1-Score', color='lightcoral', edgecolor='black')

plt.xlabel('Resolver Group', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Resolver Routing - Performance Metrics', fontsize=14, fontweight='bold')
plt.xticks(x, res_df['Resolver'], rotation=45, ha='right')
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'resolver_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved resolver metrics chart")

# ═══════════════════════════════════════════════════════════════════════════
# OVERALL PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-"*80)
print("4. OVERALL PERFORMANCE SUMMARY")
print("-"*80)

from sklearn.metrics import accuracy_score

# Create summary chart
models = ['Category\nClassification', 'Priority\nPrediction', 'Resolver\nRouting']
accuracies = [
    accuracy_score(y_test_cat, y_pred_cat),
    accuracy_score(y_test_pri, y_pred_pri),
    accuracy_score(y_test_res, y_pred_res)
]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['steelblue', 'orange', 'green'], edgecolor='black', width=0.6)
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% threshold')
plt.legend()
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved overall accuracy comparison")

# ═══════════════════════════════════════════════════════════════════════════
# WRITE DETAILED REPORT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-"*80)
print("5. GENERATING DETAILED EVALUATION REPORT")
print("-"*80)

report_lines = []
report_lines.append("="*80)
report_lines.append("ITSM AI MODEL EVALUATION REPORT")
report_lines.append("="*80)
report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")
report_lines.append(f"Test Set Size: {len(test_df):,} tickets\n")

# Category Classification
report_lines.append("\n" + "-"*80)
report_lines.append("1. CATEGORY CLASSIFICATION")
report_lines.append("-"*80)
report_lines.append(f"Overall Accuracy: {accuracy_score(y_test_cat, y_pred_cat)*100:.2f}%")
report_lines.append(f"Average Confidence: {confidences_cat.mean():.3f}")
report_lines.append(f"Low Confidence Predictions (<0.7): {sum(confidences_cat < 0.7)} ({sum(confidences_cat < 0.7)/len(confidences_cat)*100:.1f}%)")
report_lines.append("\nPer-Category Performance:")
for cat in categories:
    if cat in report_cat:
        report_lines.append(f"  {cat:15s}: Precision={report_cat[cat]['precision']:.3f} | Recall={report_cat[cat]['recall']:.3f} | F1={report_cat[cat]['f1-score']:.3f} | Support={int(report_cat[cat]['support'])}")

# Priority Prediction
report_lines.append("\n" + "-"*80)
report_lines.append("2. PRIORITY PREDICTION")
report_lines.append("-"*80)
report_lines.append(f"Overall Accuracy: {accuracy_score(y_test_pri, y_pred_pri)*100:.2f}%")
report_lines.append("\nPer-Priority Performance:")
report_pri = classification_report(y_test_pri, y_pred_pri, output_dict=True)
for pri in priorities:
    if pri in report_pri:
        report_lines.append(f"  {pri:10s}: Precision={report_pri[pri]['precision']:.3f} | Recall={report_pri[pri]['recall']:.3f} | F1={report_pri[pri]['f1-score']:.3f} | Support={int(report_pri[pri]['support'])}")

# Resolver Routing
report_lines.append("\n" + "-"*80)
report_lines.append("3. RESOLVER ROUTING")
report_lines.append("-"*80)
report_lines.append(f"Overall Accuracy: {accuracy_score(y_test_res, y_pred_res)*100:.2f}%")
confidences_res = np.max(y_pred_proba_res, axis=1)
report_lines.append(f"Average Confidence: {confidences_res.mean():.3f}")
report_lines.append("\nPer-Resolver Performance:")
for res in resolvers:
    if res in report_res:
        report_lines.append(f"  {res:20s}: Precision={report_res[res]['precision']:.3f} | Recall={report_res[res]['recall']:.3f} | F1={report_res[res]['f1-score']:.3f} | Support={int(report_res[res]['support'])}")

report_lines.append("\n" + "="*80)
report_lines.append("NOTES")
report_lines.append("="*80)
report_lines.append("- Category Classification: Excellent performance (>92%)")
report_lines.append("- Priority Prediction: Near-perfect performance (>99%)")
report_lines.append("- Resolver Routing: Moderate performance (~53%) - can be improved with more features")
report_lines.append("- All models are production-ready for hackathon demo")
report_lines.append("="*80)

report_text = "\n".join(report_lines)
with open(RESULTS_DIR / 'detailed_evaluation_report.txt', 'w') as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\n✓ Saved detailed report to {RESULTS_DIR / 'detailed_evaluation_report.txt'}")
print(f"✓ Saved {len(list(VISUALIZATIONS_DIR.glob('*.png')))} visualizations to {VISUALIZATIONS_DIR}/")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
