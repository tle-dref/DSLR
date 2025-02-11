from matplotlib.colors import Normalize
import numpy as np

class LogisticRegression():
    def __init__(self, l_rate = 0.01, nb_iters = 10000):
        self.l_rate = l_rate
        self.nb_iters = nb_iters
        self.weights = None
        self.bias = None
        self.tol = 1e-2

    def normalize(self, X):
        return X / X.max(axis=0)

    def unnormalize(self, X):
        return X * X.max(axis=0)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        nb_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        X = self.normalize(X)
        for _ in range(self.nb_iters):
            # formula for logistic regression
            linear_pred = np.dot(X, self.weights) + self.bias
            prediction = self.sigmoid(linear_pred)
            # gradient for weights and bias
            dw = (1 / nb_samples) * np.dot(X.T, (prediction - y))
            db = (1 / nb_samples) * np.sum(prediction - y)
            # stop loss
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                # print(f"Convergence atteinte à l'itération {_}")
                break
            # update weights and bias
            self.weights -= self.l_rate * dw
            self.bias -= self.l_rate * db

    def predict(self, X, weights=None, bias=None):
        X = self.normalize(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
