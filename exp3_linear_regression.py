
"""Experiment 3: Linear regression on X=[1..5], Y=[2,4,6,8,10]
Input: X,Y synthetic
Output: fitted line, MSE and R2; plot saved.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,6,8,10])

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)
print("Slope (coef):", model.coef_[0], "Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y,y_pred))
print("R2:", r2_score(y,y_pred))

plt.scatter(X,y,label="Data")
plt.plot(X,y_pred, color='red', label='Fit')
plt.legend()
plt.savefig("exp3_linear_fit.png")
print("Saved plot to exp3_linear_fit.png")
