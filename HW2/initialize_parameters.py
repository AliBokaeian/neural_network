import numpy as np


def initialize_parameters_random(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((1, 1))
    return parameters


def initialize_parameters_zeros(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((1, 1))
    return parameters


def initialize_parameters_xavier(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1. / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def initialize_parameters_HE(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def initialize_velocities(parameters):
    L = len(parameters) // 2
    velocities = {}
    for l in range(L):
        velocities["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        velocities["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return velocities


def initialize_cache(parameters):
    L = len(parameters) // 2
    cache = {}
    for l in range(L):
        cache["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        cache["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return cache
