import pandas as pd

# Load the MovieLens dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(url, sep='\t', names=column_names)

# Explore the dataset
print(ratings.head())
print(ratings.describe())
print(ratings.info())

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create a user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Make predictions
def predict_user_based(user_id, item_id):
    similar_users = user_similarity[user_id - 1]  # Adjust for 0-indexing
    user_ratings = user_item_matrix.loc[:, item_id]
    predicted_rating = np.dot(similar_users, user_ratings) / np.sum(similar_users)
    return predicted_rating

# Example prediction
user_id = 1
item_id = 1
print(f"Predicted rating for user {user_id} on item {item_id}: {predict_user_based(user_id, item_id)}")

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Recreate the user-item matrix for training data
train_user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Recompute user similarity for the training data
train_user_similarity = cosine_similarity(train_user_item_matrix)

# Make predictions for the test set
test_data['predicted_rating'] = test_data.apply(lambda row: predict_user_based(row['user_id'], row['item_id']), axis=1)

# Define threshold for relevant items
threshold = 3.5

# Convert ratings to binary relevance
test_data['relevant'] = test_data['rating'] >= threshold
test_data['predicted_relevant'] = test_data['predicted_rating'] >= threshold

# Calculate evaluation metrics
precision = precision_score(test_data['relevant'], test_data['predicted_relevant'])
recall = recall_score(test_data['relevant'], test_data['predicted_relevant'])
f1 = f1_score(test_data['relevant'], test_data['predicted_relevant'])

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

