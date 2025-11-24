
"""Experiment 10: Linear regression 'classifier' approach on Iris (we'll predict one class vs rest).
Input: Iris dataset
Output: MSE and simple accuracy of thresholded regression predictions
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # class 0 vs rest

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
reg = LinearRegression().fit(X_train, y_train)
preds = reg.predict(X_test)
# threshold at 0.5
preds_bin = (preds >= 0.5).astype(int)
print("MSE:", mean_squared_error(y_test,preds))
print("Accuracy (thresholded):", accuracy_score(y_test,preds_bin))
