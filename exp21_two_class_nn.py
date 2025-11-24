
"""Experiment 21: Neural network on two-class synthetic data (make_moons)
Input: make_moons dataset
Output: decision boundary, accuracy
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
try:
    from mlxtend.plotting import plot_decision_regions
    mlxtend_available = True
except Exception:
    mlxtend_available = False

X,y = make_moons(n_samples=300, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000).fit(X_train,y_train)
print("Accuracy:", clf.score(X_test,y_test))
if mlxtend_available:
    plt.figure(); plot_decision_regions(X_test, y_test, clf=clf, legend=2)
    plt.title("Two-class NN boundary"); plt.savefig("exp21_two_class_boundary.png")
    print("Saved exp21_two_class_boundary.png")
else:
    print("mlxtend not installed; boundary plot not created.")
