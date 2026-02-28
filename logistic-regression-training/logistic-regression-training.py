import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def loss(y,p):
    N = len(y)
    loss = -(1/N)*np.sum(y*np.log(p)+(1-y)*np.log(1-p))
    return loss

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N= len(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    for i in range(steps):
        p = _sigmoid(X @ w + b)
        gradientW = X.T@(p-y)/N
        gradientB = np.mean(p-y)
        w = w - lr*gradientW
        b = b - lr*gradientB

    return (w,b)