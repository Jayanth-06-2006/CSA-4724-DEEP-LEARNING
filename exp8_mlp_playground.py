
"""Experiment 8: MLP with different inputs, learning rates and activations (sklearn MLP analog).
Input: make_moons synthetic dataset.
Output: decision boundary image and accuracy printed.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

try:
    from mlxtend.plotting import plot_decision_regions
    mlxtend_available = True
except Exception:
    mlxtend_available = False

X,y = make_moons(n_samples=300, noise=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2)

clf = MLPClassifier(hidden_layer_sizes=(50,50), activation='relu', learning_rate_init=0.1, max_iter=1000)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Test accuracy:", acc)

if mlxtend_available:
    plt.figure(figsize=(6,5))
    plot_decision_regions(X_test, y_test, clf=clf, legend=2)
    plt.title("MLP Decision Boundary")
    plt.savefig("exp8_mlp_boundary.png")
    print("Saved decision boundary to exp8_mlp_boundary.png")
else:
    print("mlxtend not installed; decision boundary plot not generated.")
