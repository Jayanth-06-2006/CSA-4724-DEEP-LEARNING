
"""Experiment 13: Decision tree on breast cancer dataset; print accuracy and save tree plot
Input: sklearn breast_cancer
Output: accuracy and saved tree visualization (requires graphviz for fancy plot, otherwise text)
"""
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True, max_depth=3)
plt.savefig("exp13_tree.png")
print("Saved tree to exp13_tree.png")
