#1
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate synthetic dataset
np.random.seed(42)
n_samples = 300
n_features = 5

data = np.random.rand(n_samples, n_features)
customer_data = pd.DataFrame(data, columns=['Age', 'Income','SpendingScore',
'Savings', 'Debt'])

# Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customer_data_scaled)
customer_data['Cluster'] = clusters

print(customer_data.head())
#2
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Calculate SSE for a range of cluster numbers
sse = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data_scaled)
    sse.append(kmeans.inertia_)
    
    if k > 1:
        score = silhouette_score(customer_data_scaled, kmeans.labels_)
        silhouette_scores.append(score)

# Plot the elbow method results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')

# Plot the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_range[1:], silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')

plt.show()
#3

cluster_means = customer_data.groupby('Cluster').mean()
print(cluster_means)

#4
from sklearn.decomposition import PCA

# Reduce the dimensionality of the data using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(customer_data_scaled)
customer_data['PCA1'] = principal_components[:, 0]
customer_data['PCA2'] = principal_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 6))
for cluster in range(kmeans.n_clusters):
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Customer Clusters')
plt.legend()
plt.show()
