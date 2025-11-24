
"""Experiment 9: Evaluate NN architectures (hidden neurons and learning rates)
Input: sklearn digits dataset
Output: table of accuracies per architecture printed
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import itertools

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

hidden_options = [(5,), (10,), (20,), (20,20)]
lr_options = [0.001, 0.01]
results = []

for hid, lr in itertools.product(hidden_options, lr_options):
    clf = MLPClassifier(hidden_layer_sizes=hid, learning_rate_init=lr, max_iter=500, random_state=0)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results.append((hid, lr, acc))
    print(f"Hidden={hid}, lr={lr} => accuracy={acc:.4f}")
