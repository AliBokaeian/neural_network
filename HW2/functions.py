import numpy as np
from sklearn.metrics import mean_squared_error as mse


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def tanh(Z):
    return np.tanh(Z)


def leaky_relu(Z):
    return np.maximum(0.01 * Z, Z)


def sigmoid_derivative(Z):
    A = sigmoid(Z)
    return A * (1 - A)


def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)


def tanh_derivative(Z):
    A = tanh(Z)
    return 1 - np.power(A, 2)


def leaky_relu_derivative(Z):
    return np.where(Z > 0, 1, 0.01)


#########################               ##############################
#########################               ##############################
######################### LOSS FUNCTION ##############################
#########################               ##############################
#########################               ##############################

def MSE(AL, y_true):
    # print(f'{y_true.shape = }\t{AL.shape = }')
    return mse(y_true, AL)


def MSE_derivative(AL, y_true):
    # print(f'{y_true.shape = }\t{AL.shape = }')
    return -2 * (y_true - AL)


def Binary_Cross_Entropy(AL, y_true):
    # print(f'{y_true.shape = }\t{AL.shape = }')
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(AL) + (1 - y_true) * np.log(1 - AL)) / m
    loss = np.squeeze(loss)
    return loss


def Binary_Cross_Entropy_derivative(AL, y_true):
    # print(f'{y_true.shape = }\t{AL.shape = }')
    return - (np.divide(y_true, AL) - np.divide(1 - y_true, 1 - AL))
