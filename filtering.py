#2
import pandas as pd
# Load the MovieLens dataset
!pip install surprise
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate, KFold
import numpy as np
import itertools
# Load the MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')
# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)
# Print some ratings from the training set
print("Some ratings from the training set:")
for uid, iid, rating in itertools.islice(trainset.all_ratings(), 5):
    print(f"User {uid} rated item {iid} with {rating}")
# Create a DataFrame from the raw ratings
raw_ratings = data.raw_ratings
df = pd.DataFrame(raw_ratings, columns=['user_id', 'item_id', 'rating', 'timestamp'])
# Explore the DataFrame
print("\nDataFrame head:")
print(df.head())
print("\nDataFrame info:")
print(df.info())
print("\nSummary statistics for ratings:")
print(df['rating'].describe())

# Evaluate different similarity metrics using KNNBasic
sim_options = ['cosine', 'pearson', 'msd', 'pearson_baseline']

for sim_name in sim_options:
    algo = KNNBasic(sim_options={'name': sim_name, 'user_based': True})
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print(f"\nAverage RMSE for {sim_name} similarity: {np.mean(results['test_rmse']):.4f}")
    print(f"Average MAE for {sim_name} similarity: {np.mean(results['test_mae']):.4f}")
    print("-" * 30)
# Evaluate the best similarity metric (assume it's 'pearson_baseline' for this example)
best_sim_name = 'pearson_baseline'
algo = KNNBasic(sim_options={'name': best_sim_name, 'user_based': True})
kf = KFold(n_splits=5)
precisions = []
recalls = []
f1_scores = []

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    threshold = 3.5
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for prediction in predictions:
        actual_rating = prediction.r_ui
        predicted_rating = prediction.est
        if predicted_rating >= threshold and actual_rating >= threshold:
            true_positives += 1
        elif predicted_rating >= threshold and actual_rating < threshold:
            false_positives += 1
        elif predicted_rating < threshold and actual_rating >= threshold:
            false_negatives += 1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
print(f"\nAverage Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
# Item-based collaborative filtering
algo_item = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
results_item = cross_validate(algo_item, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print(f"\nAverage RMSE for item-based collaborative filtering: {np.mean(results_item['test_rmse']):.4f}")
print(f"Average MAE for item-based collaborative filtering: {np.mean(results_item['test_mae']):.4f}")
# Evaluate item-based collaborative filtering using precision, recall, and F1-score
kf = KFold(n_splits=5)
precisions = []
recalls = []
f1_scores = []

for trainset, testset in kf.split(data):
    algo_item.fit(trainset)
    predictions = algo_item.test(testset)
    threshold = 3.5
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for prediction in predictions:
        actual_rating = prediction.r_ui
        predicted_rating = prediction.est
        if predicted_rating >= threshold and actual_rating >= 
        threshold:
            true_positives += 1
        elif predicted_rating >= threshold and actual_rating < 
        threshold:
            false_positives += 1
        elif predicted_rating < threshold and actual_rating >= 
        threshold:
            false_negatives += 1
    precision = true_positives / (true_positives + false_positives) 
    
    if true_positives + false_positives > 0 
    else 0
    recall = true_positives / (true_positives + false_negatives)
    if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if 
    precision + recall > 0 else 0
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
print(f"\nAverage Precision for Item-based CF: {np.mean(precisions):.4f}")
print(f"Average Recall for Item-based CF: {np.mean(recalls):.4f}")
print(f"Average F1-score for Item-based CF: {np.mean(f1_scores):.4f}")
# Hybrid recommender combining user-based and item-based predictions
algo_user = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
algo_item = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
# Fit the models to the data
algo_user.fit(data.build_full_trainset())
algo_item.fit(data.build_full_trainset())

# Get predictions from both models for a given user and item
user_id = '196'  # Example user ID
item_id = '302'  # Example item ID
prediction_user = algo_user.predict(user_id, item_id).est
prediction_item = algo_item.predict(user_id, item_id).est
# Combine predictions using a weighted average
weight_user = 0.6
weight_item = 0.4
hybrid_prediction = weight_user * prediction_user + weight_item * prediction_item
print(f"\nHybrid prediction for user {user_id} and item {item_id}: {hybrid_prediction:.4f}")
# Evaluate the hybrid recommender using cross-validation
kf = KFold(n_splits=5)
rmse_hybrid = []
mae_hybrid = []
for trainset, testset in kf.split(data):
    algo_user.fit(trainset)
    algo_item.fit(trainset)
    predictions_hybrid = []
    for uid, iid, true_r in testset:
        pred_user = algo_user.predict(uid, iid).est
        pred_item = algo_item.predict(uid, iid).est
        pred_hybrid = weight_user * pred_user + weight_item * pred_item
        predictions_hybrid.append((uid, iid, true_r, pred_hybrid, None))
    rmse_hybrid.append(accuracy.rmse(predictions_hybrid, verbose=False))
    mae_hybrid.append(accuracy.mae(predictions_hybrid, verbose=False))
print(f"\nAverage RMSE for Hybrid Recommender: {np.mean(rmse_hybrid):.4f}")
print(f"Average MAE for Hybrid Recommender: {np.mean(mae_hybrid):.4f}")
# Summary of RMSE and MAE
print("\nAverage RMSE:")
print(f"  User-based CF: {np.mean(results['test_rmse']):.4f}")
print(f"  Item-based CF: {np.mean(results_item['test_rmse']):.4f}")
print(f"  Hybrid CF: {np.mean(rmse_hybrid):.4f}")
print("\nAverage MAE:")
print(f"  User-based CF: {np.mean(results['test_mae']):.4f}")
print(f"  Item-based CF: {np.mean(results_item['test_mae']):.4f}")
print(f"  Hybrid CF: {np.mean(mae_hybrid):.4f}")
