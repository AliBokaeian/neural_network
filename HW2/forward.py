from functions import *

def activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "Identity":
        A = Z
    elif activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "ReLU":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "leaky_relu":
        A = leaky_relu(Z)

    cache = (A_prev, W, b, Z)
    return A, cache


def forward(X, parameters, activation_fw):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_fw)
        caches.append(cache)

    AL, cache = activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation_fw)
    caches.append(cache)

    return AL, caches