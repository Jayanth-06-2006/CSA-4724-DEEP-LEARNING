
"""Experiment 11: Logistic Regression on breast cancer dataset.
Input: sklearn breast_cancer dataset
Output: accuracy, confusion matrix, ROC AUC saved/printed.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test,pred))
print("ROC AUC:", roc_auc_score(y_test, prob))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.title("Logistic Regression Confusion Matrix (Breast Cancer)")
plt.savefig("exp11_logreg_cm.png")
print("Saved exp11_logreg_cm.png")
