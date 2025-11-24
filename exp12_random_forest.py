
"""Experiment 12: Random Forest on Iris dataset
Input: sklearn iris
Output: accuracy, feature importances printed and plotted
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))
imp = clf.feature_importances_
print("Feature importances:", imp)
plt.bar(np.arange(len(imp)), imp)
plt.xticks(range(len(imp)), data.feature_names, rotation=45)
plt.tight_layout()
plt.savefig("exp12_rf_importances.png")
print("Saved exp12_rf_importances.png")
