import pandas as pd
# Load dataset
df = pd.read_csv('/content/Groceries_dataset.csv')
print(df.head())

from collections import Counter
def generate_itemsets(data):
    itemsets = Counter()
    for index, transaction in data.iterrows():
        unique_items = set(transaction.dropna())
        itemsets.update(unique_items)
    return itemsets
# Generate itemsets
itmsets = generate_itemsets(df)
print(itemsets) 

from mlxtend.frequent_patterns import apriori, association_rules

# Convert dataset to a one-hot encoded DataFrame
def one_hot_encode_transaction(transaction):
    transaction = transaction.dropna()
    unique_items = set(transaction)
    return pd.Series({item: 1 for item in unique_items})

one_hot = df.apply(one_hot_encode_transaction, axis=1).fillna(0)
print(one_hot.head())

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot, min_support=0.1, use_colnames=True)
print(frequent_itemsets)    
\end{verbatim}
\begin{verbatim}
from mlxtend.frequent_patterns import apriori, association_rules

# Convert dataset to a one-hot encoded DataFrame
def one_hot_encode_transaction(transaction):
    transaction = transaction.dropna()
    unique_items = set(transaction)
    return pd.Series({item: 1 for item in unique_items})

one_hot = df.apply(one_hot_encode_transaction, axis=1).fillna(0)
print(one_hot.head())

# Apply Apriori algorithm to find frequent itemsets
min_support_threshold = 0.01  # Adjust this value as needed
frequent_itemsets = apriori(one_hot, min_support=min_support_threshold, use_colnames=True)
print(f"Number of frequent itemsets found: {len(frequent_itemsets)}")
print(frequent_itemsets)

# Check if frequent_itemsets is empty
if frequent_itemsets.empty:
    print("No frequent itemsets found. Consider lowering the min_support threshold.")
else:
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])

# Calculate lift for association rules
rules['lift'] = rules['support'] / (rules['antecedent support'] * rules['consequent support'])

# Filter rules based on your criteria, e.g., high confidence and lift
filtered_rules = rules[(rules['confidence'] > 0.7) & (rules['lift'] > 1.2)]

print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
