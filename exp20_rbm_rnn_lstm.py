
"""Experiment 20: Compare BernoulliRBM, simple RNN and LSTM on a toy time-series (sine waves)
Input: synthetic monthly temperature-like series
Output: reconstruction error for RBM, MSE for RNN and LSTM
"""
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from sklearn.model_selection import train_test_split

t = np.linspace(0,10,500)
series = np.sin(t) + 0.1*np.random.randn(len(t))
def create_windows(s, window=10):
    X=[]
    y=[]
    for i in range(len(s)-window):
        X.append(s[i:i+window])
        y.append(s[i+window])
    return np.array(X), np.array(y)
X,y = create_windows(series, window=20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

rbm = BernoulliRBM(n_components=50, learning_rate=0.01, n_iter=10, random_state=0)
X_bin = (X_train > X_train.mean()).astype(int).reshape(X_train.shape[0], -1)
rbm.fit(X_bin)
recon = rbm.gibbs(X_bin)
rbm_err = np.mean((X_bin - recon)**2)
print("RBM reconstruction MSE (binarized):", rbm_err)

Xr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
Xr_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))
rnn = Sequential([SimpleRNN(16, input_shape=(Xr.shape[1],1)), Dense(1)])
rnn.compile(optimizer='adam', loss='mse')
rnn.fit(Xr, y_train, epochs=10, verbose=0)
pred_rnn = rnn.predict(Xr_test)
print("RNN MSE:", mean_squared_error(y_test, pred_rnn))

lstm = Sequential([LSTM(16, input_shape=(Xr.shape[1],1)), Dense(1)])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(Xr, y_train, epochs=10, verbose=0)
pred_lstm = lstm.predict(Xr_test)
print("LSTM MSE:", mean_squared_error(y_test, pred_lstm))
