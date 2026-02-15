"""Quick validation of the generated dataset"""
import pandas as pd

df = pd.read_csv('synthetic_itsm_tickets.csv')

print("="*80)
print("DATASET VALIDATION")
print("="*80)

print(f"\nTotal tickets: {len(df):,}")
print(f"Categories: {df['category'].nunique()}")
print(f"Resolver groups: {df['ground_truth_resolver_group'].nunique()}")

print("\n" + "="*80)
print("CATEGORY DISTRIBUTION")
print("="*80)
cat_dist = df['category'].value_counts().sort_index()
for cat, count in cat_dist.items():
    pct = (count / len(df)) * 100
    print(f"{cat:20s}: {count:6,} ({pct:5.2f}%)")

print("\n" + "="*80)
print("RESOLVER DISTRIBUTION")
print("="*80)
res_dist = df['ground_truth_resolver_group'].value_counts().sort_index()
for res, count in res_dist.items():
    pct = (count / len(df)) * 100
    print(f"{res:20s}: {count:6,} ({pct:5.2f}%)")

print("\n" + "="*80)
print("CATEGORY → RESOLVER VALIDATION")
print("="*80)
print(f"{'Category':<20} {'Resolver':<20} {'Tickets':<10} {'Status':<10}")
print("─"*80)

all_correct = True
for cat in sorted(df['category'].unique()):
    cat_df = df[df['category'] == cat]
    resolvers = cat_df['ground_truth_resolver_group'].unique()
    
    if len(resolvers) == 1:
        status = "✅ OK"
    else:
        status = f"❌ ERROR ({len(resolvers)} resolvers)"
        all_correct = False
    
    print(f"{cat:<20} {resolvers[0]:<20} {len(cat_df):<10,} {status:<10}")

print("\n" + "="*80)
if all_correct:
    print("✅ VALIDATION PASSED: All categories have single resolver mapping!")
else:
    print("❌ VALIDATION FAILED: Some categories have multiple resolvers!")
print("="*80)
