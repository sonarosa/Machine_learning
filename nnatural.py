import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Define the classifiers
svm = SVC(probability=True, random_state=42)

# Function to perform cross-validation
def evaluate_classifier(X, y, classifier):
    scores = cross_val_score(classifier, X, y, cv=10)
    return scores.mean()

# Reduce to 2 features using different methods
pca_2 = PCA(n_components=2).fit_transform(X)
lda_2 = LDA(n_components=2).fit_transform(X, y)
tsne_2 = TSNE(n_components=2, random_state=42).fit_transform(X)
svd_2 = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)

# Reduce to 3 features using different methods
pca_3 = PCA(n_components=3).fit_transform(X)
# Note: LDA cannot be used to reduce to 3 features in this case because of its limitation
tsne_3 = TSNE(n_components=3, random_state=42).fit_transform(X)
svd_3 = TruncatedSVD(n_components=3, random_state=42).fit_transform(X)

# Evaluate the classifier on the reduced datasets
scores_2d = {
    'PCA': evaluate_classifier(pca_2, y, svm),
    'LDA': evaluate_classifier(lda_2, y, svm),
    't-SNE': evaluate_classifier(tsne_2, y, svm),
    'SVD': evaluate_classifier(svd_2, y, svm)
}

scores_3d = {
    'PCA': evaluate_classifier(pca_3, y, svm),
    't-SNE': evaluate_classifier(tsne_3, y, svm),
    'SVD': evaluate_classifier(svd_3, y, svm)
}

# Print the results
print("Cross-validation scores with 2 features:")
for method, score in scores_2d.items():
    print(f"{method}: {score:.3f}")

print("\nCross-validation scores with 3 features:")
for method, score in scores_3d.items():
    print(f"{method}: {score:.3f}")
