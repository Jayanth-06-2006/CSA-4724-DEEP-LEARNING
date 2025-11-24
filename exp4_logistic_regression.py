
"""Experiment 4: Logistic regression on synthetic dataset (binary).
Input: synthetic or replace with a real dataset (e.g., Pima Indians).
Output: predicted probabilities, accuracy, confusion matrix plot.
"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

X, y = make_classification(n_samples=300, n_features=5, n_informative=3, n_redundant=0, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train,y_train)
probs = clf.predict_proba(X_test)[:,1]
preds = clf.predict(X_test)

print("Example predicted probabilities:", np.round(probs[:5],3))
print("Accuracy:", accuracy_score(y_test,preds))
print("ROC AUC:", roc_auc_score(y_test, probs))

cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("exp4_logistic_cm.png")
print("Saved exp4_logistic_cm.png")
