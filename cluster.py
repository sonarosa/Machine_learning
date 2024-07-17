  K-MEANS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA for visualization (reduce to 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Find the optimal number of clusters using the elbow method
wcss = []  # within-cluster sums of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
    n_init=10, random_state=0)
    kmeans.fit(X_std)
    wcss.append(kmeans.inertia_)
# From the elbow graph, we choose 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, 
n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X_std)

# Plotting function for visualizing clustering results in 2D (PCA reduced dimensions)
def plot_clusters(X, y, centroids, title):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=100, c='purple', label='Iris-setosa')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=100, c='orange', label='Iris-versicolor')
    plt.scatter(X[y == 2, 0], X[y == 2, 1], s=100, c='green', label='Iris-virginica') 
    # Plotting the centroids of the clusters
plt.scatter (centroids [:, 0], centroids [:, 1], s = 300,
c ='red', marker = '*', label = 'Centroids') 
plt.title(title) plt.xlabel (PCA 1')
plt.ylabel (PCA 2')
plt.legend()
plt.show()
# Visualizing the clusters using PCA
plot_clusters(X_pca, y_kmeans, kmeans.cluster_centers_[:, :2],
"K-means Clustering on Iris Dataset")

    K-MEDOIDS
import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce the dimensionality to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Compute KMedoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(X_pca)
labels = kmedoids.labels_

# Plot the results
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k
    xy = X_pca[class_member_mask]
    plt.plot(
        xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k",
        markersize=6,
    )
# Plot the medoids
plt.plot(
    kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], "o",
    markerfacecolor="cyan", markeredgecolor="k",
    markersize=10, label='Medoids'
)

plt.title("KMedoids clustering on Iris dataset")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.legend()
plt.show()

FUZZY C-MEANS

import numpy as np
import skfuzzy as fuzz
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce the dimensionality to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Define the number of clusters
n_clusters = 3

# Apply fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_pca.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# Predict cluster membership for each data point
cluster_membership = np.argmax(u, axis=0)

# Print the cluster centers
print('Cluster Centers:\n', cntr)

# Print the cluster membership for each data point
print('Cluster Membership:\n', cluster_membership)

# Plot the results
plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b']
for i in range(n_clusters):
    plt.scatter(X_pca[cluster_membership == i, 0], 
    X_pca[cluster_membership == i, 1], c=colors[i], 
    label=f'Cluster {i+1}')

# Plot the cluster centers
for pt in cntr:
    plt.plot(pt[0], pt[1], 'ks', markersize=10, label='Centroid')
    plt.scatter(pt[0], pt[1], marker='o', c='cyan', 
    edgecolor='k', s=100, label='Cluster center')

plt.title('Fuzzy C-Means Clustering on Iris Dataset ')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()
