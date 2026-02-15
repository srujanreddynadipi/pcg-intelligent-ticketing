import pandas as pd

df = pd.read_csv('synthetic_itsm_tickets.csv')

print('SAMPLE TICKETS (2 per category):\n')
for cat in ['Network', 'Hardware', 'Software', 'Access']:
    print(f'\n=== {cat} ===')
    samples = df[df['ground_truth_category'] == cat].head(2)
    for idx, row in samples.iterrows():
        print(f'Title: {row["title"]}')
        print(f'Description: {row["description"][:100]}...')
        print()
