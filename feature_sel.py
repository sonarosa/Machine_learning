import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create DataFrame and visualize
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
sns.pairplot(df, hue='target')
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection with SelectKBest
selector = SelectKBest(chi2, k=2)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)
selected_features_kbest = np.array(feature_names)[selector.get_support()]

# Feature importance with RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features_rf = np.array(feature_names)[indices[:2]]

# Feature selection with RFE using SVM
svm = SVC(kernel='linear')
rfe = RFE(svm, n_features_to_select=2)
rfe.fit(X_train_scaled, y_train)
selected_features_rfe = np.array(feature_names)[rfe.support_]

# Evaluate model
def evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

lr = LogisticRegression(random_state=42)
accuracy_original = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, lr)
accuracy_kbest = evaluate_model(X_train_kbest, X_test_kbest, y_train, y_test, lr)
accuracy_rf = evaluate_model(X_train_scaled[:, indices[:2]], X_test_scaled[:, indices[:2]], y_train, y_test, lr)
accuracy_rfe = evaluate_model(X_train_scaled[:, rfe.support_], X_test_scaled[:, rfe.support_], y_train, y_test, lr)

# Print results
print(f"Selected features (Univariate): {selected_features_kbest}")
print(f"Selected features (Random Forest): {selected_features_rf}")
print(f"Selected features (RFE): {selected_features_rfe}")
print(f"Accuracy with original features: {accuracy_original:.4f}")
print(f"Accuracy with selected features (Univariate): {accuracy_kbest:.4f}")
print(f"Accuracy with selected features (Random Forest): {accuracy_rf:.4f}")
print(f"Accuracy with selected features (RFE): {accuracy_rfe:.4f}")
