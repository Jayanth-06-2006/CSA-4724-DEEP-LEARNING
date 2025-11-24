
"""Experiment 16: Hyperparameter tuning for a small neural network using GridSearchCV.
Input: sklearn digits dataset
Output: best parameters found and cross-validated score
"""
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (32,32)],
    'learning_rate_init': [0.001, 0.01]
}
grid = GridSearchCV(MLPClassifier(max_iter=500, random_state=0), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)
