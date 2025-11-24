
"""Experiment 25: Compare activation functions and structures.
Input: sklearn digits
Output: printed accuracies per (activation, hidden layers) config
"""
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
activations = ['relu','tanh','logistic']
structures = [(32,), (64,), (64,64)]
for act in activations:
    for st in structures:
        clf = MLPClassifier(hidden_layer_sizes=st, activation=act, max_iter=500).fit(X_train,y_train)
        acc = clf.score(X_test, y_test)
        print(f"Activation={act}, structure={st} => accuracy={acc:.4f}")
