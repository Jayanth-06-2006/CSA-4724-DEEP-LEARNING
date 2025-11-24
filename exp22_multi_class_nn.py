
"""Experiment 22: Neural network on multi-class (Iris)
Input: Iris dataset
Output: confusion matrix and accuracy printed
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000).fit(X_train,y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))
cm = confusion_matrix(y_test,pred)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot()
plt.savefig("exp22_multi_cm.png")
print("Saved exp22_multi_cm.png")
