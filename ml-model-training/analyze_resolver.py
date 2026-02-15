import pandas as pd

df = pd.read_csv('synthetic_itsm_tickets.csv')

print("="*80)
print("RESOLVER ROUTING ANALYSIS")
print("="*80)

print("\n1. RESOLVER GROUP DISTRIBUTION:")
print(df['ground_truth_resolver_group'].value_counts())

print("\n2. RESOLVER BY CATEGORY:")
print(pd.crosstab(df['ground_truth_category'], df['ground_truth_resolver_group']))

print("\n3. CATEGORY TO RESOLVER MAPPING:")
for category in sorted(df['ground_truth_category'].unique()):
    resolvers = df[df['ground_truth_category'] == category]['ground_truth_resolver_group'].value_counts()
    print(f"\n{category}:")
    for resolver, count in resolvers.items():
        pct = (count / len(df[df['ground_truth_category'] == category])) * 100
        print(f"  {resolver:20s}: {count:6d} ({pct:5.1f}%)")

print("\n" + "="*80)
