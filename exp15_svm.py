
"""Experiment 15: SVM (RBF) classifier on Iris.
Input: sklearn iris
Output: accuracy and number of support vectors
"""
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

clf = SVC(kernel='rbf', probability=True).fit(X_train,y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))
print("Number of support vectors:", clf.support_vectors_.shape[0])
