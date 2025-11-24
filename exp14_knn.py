
"""Experiment 14: KNN classifier on Iris; compare different K values.
Input: sklearn iris
Output: accuracy per K and a saved plot of accuracy vs K
"""
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

ks = range(1,16)
accs = []
for k in ks:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accs.append(acc)
    print(f"k={k}, accuracy={acc:.4f}")

plt.plot(ks, accs, marker='o')
plt.xlabel("k"); plt.ylabel("Accuracy")
plt.savefig("exp14_knn_acc.png")
print("Saved exp14_knn_acc.png")
