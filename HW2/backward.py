from functions import *

def activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]
    if activation == "Identity_derivative":
        dZ = Z
    elif activation == "sigmoid_derivative":
        dZ = dA * sigmoid_derivative(Z)
    elif activation == "ReLU_derivative":
        # dZ = np.array(dA, copy=True)
        # dZ[Z <= 0] = 0
        dZ = relu_derivative(Z)

    elif activation == "tanh_derivative":
        dZ = tanh_derivative(Z)
    elif activation == "leaky_relu_derivative":
        dZ = leaky_relu_derivative(Z)


    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward(AL, Y, caches, activation_bw):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = Binary_Cross_Entropy_derivative(AL, Y)
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation_bw)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 2)], current_cache, activation_bw)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads