import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the classifiers
svm = SVC(probability=True, random_state=42)
nb = GaussianNB()
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()

# Evaluate the performance of each classifier using cross-validation
svm_scores = cross_val_score(svm, X, y, cv=10)
nb_scores = cross_val_score(nb, X, y, cv=10)
rf_scores = cross_val_score(rf, X, y, cv=10)
knn_scores = cross_val_score(knn, X, y, cv=10)

print('SVM score: %0.3f' % svm_scores.mean())
print('Naive Bayes score: %0.3f' % nb_scores.mean())
print('Random Forest score: %0.3f' % rf_scores.mean())
print('KNN score: %0.3f' % knn_scores.mean())

# Bagging
bagging_rf = BaggingClassifier(base_estimator=rf, n_estimators=10, random_state=42)
bagging_rf.fit(X_train, y_train)
bagging_rf_pred = bagging_rf.predict(X_test)
bagging_rf_score = accuracy_score(y_test, bagging_rf_pred)
print('Bagging (Random Forest) Score: %0.3f' % bagging_rf_score)

# Boosting
boosting_rf = AdaBoostClassifier(base_estimator=rf, n_estimators=50, random_state=42)
boosting_rf.fit(X_train, y_train)
boosting_rf_pred = boosting_rf.predict(X_test)
boosting_rf_score = accuracy_score(y_test, boosting_rf_pred)
print('Boosting (Random Forest) Score: %0.3f' % boosting_rf_score)

# Stacking
estimators = [
    ('svm', svm),
    ('nb', nb),
    ('rf', rf),
    ('knn', knn)
]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)
stacking_score = accuracy_score(y_test, stacking_pred)
print('Stacking Score: %0.3f' % stacking_score)
# Compare model performance
results = {
    'Model': ['SVM', 'Naive Bayes', 'Random Forest', 'KNN', 
    'Bagging RF', 'Boosting RF', 'Stacking'],
    'Accuracy': [
        svm_scores.mean(),
        nb_scores.mean(),
        rf_scores.mean(),
        knn_scores.mean(),
        bagging_rf_score,
        boosting_rf_score,
        stacking_score
    ]
}
# Print the results
for model, score in zip(results['Model'], results['Accuracy']):
    print(f"{model} Accuracy: {score:.3f}")
