from forward import *

def predict(X, parameters, activation):
    m = X.shape[1]
    p = np.zeros((1, m))
    probas, _ = forward(X, parameters, activation)
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    return p


def compute_accuracy(predictions, Y):
    return 100 - np.mean(np.abs(predictions - Y)) * 100