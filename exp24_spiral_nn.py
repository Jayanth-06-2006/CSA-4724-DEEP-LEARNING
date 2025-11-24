
"""Experiment 24: NN on spiral dataset (3-class); we generate spiral data
Input: synthetic spiral (3 classes)
Output: decision boundary and accuracy
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
try:
    from mlxtend.plotting import plot_decision_regions
    mlxtend_available = True
except Exception:
    mlxtend_available = False

def generate_spiral(n_points, noise=0.5, classes=3):
    X = []
    y = []
    n = n_points // classes
    for j in range(classes):
        r = np.linspace(0.0,1,n)
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*noise
        xs = r * np.sin(t*2.5)
        ys = r * np.cos(t*2.5)
        X.extend(np.c_[xs, ys])
        y.extend([j]*n)
    return np.array(X), np.array(y)

X,y = generate_spiral(600, noise=0.2, classes=3)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000).fit(X_train,y_train)
print("Accuracy:", clf.score(X_test,y_test))
if mlxtend_available:
    plt.figure(); plot_decision_regions(X_test, y_test, clf=clf, legend=2)
    plt.savefig("exp24_spiral_boundary.png")
    print("Saved exp24_spiral_boundary.png")
else:
    print("mlxtend not installed; boundary plot not created.")
