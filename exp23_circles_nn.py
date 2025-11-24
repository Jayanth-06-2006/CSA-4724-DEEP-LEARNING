
"""Experiment 23: NN on circular data (make_circles)
Input: make_circles
Output: decision boundary image and accuracy
"""
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
try:
    from mlxtend.plotting import plot_decision_regions
    mlxtend_available = True
except Exception:
    mlxtend_available = False

X,y = make_circles(n_samples=400, noise=0.05, factor=0.5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000).fit(X_train,y_train)
print("Accuracy:", clf.score(X_test,y_test))
if mlxtend_available:
    plt.figure(); plot_decision_regions(X_test, y_test, clf=clf, legend=2)
    plt.savefig("exp23_circles_boundary.png")
    print("Saved exp23_circles_boundary.png")
else:
    print("mlxtend not installed; boundary plot not created.")
