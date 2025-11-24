
"""Experiment 2: Show overfitting using a model that overfits on small data.
Input: make_classification synthetic dataset.
Output: training and test accuracy showing overfitting.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=20, n_informative=2, n_redundant=0, random_state=0)
# small training set to encourage overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=1)

model = MLPClassifier(hidden_layer_sizes=(200,200), max_iter=1000, random_state=0)  # big model
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Training accuracy: {train_acc*100:.2f}%")
print(f"Testing accuracy:  {test_acc*100:.2f}%")
# Expect training >> testing (overfitting)
