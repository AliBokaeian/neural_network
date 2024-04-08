import numpy as np


def mini_batch_gradient_descent(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters


def stochastic_gradient_descent(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters


def momentum(parameters, gradients, learning_rate, velocities, beta=0.5):
    L = len(parameters) // 2
    for l in range(L):
        velocities["dW" + str(l + 1)] = beta * velocities["dW" + str(l + 1)] + (1 - beta) * gradients["dW" + str(l + 1)]
        velocities["db" + str(l + 1)] = beta * velocities["db" + str(l + 1)] + (1 - beta) * gradients["db" + str(l + 1)]
        parameters["W" + str(l + 1)] -= learning_rate * velocities["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * velocities["db" + str(l + 1)]

    return parameters, velocities


def rmsprop(parameters, gradients, learning_rate, cache, beta=0.9, epsilon=1e-8):
    L = len(parameters) // 2
    for l in range(L):
        cache["dW" + str(l + 1)] = beta * cache["dW" + str(l + 1)] + (1 - beta) * np.square(
            gradients["dW" + str(l + 1)])
        cache["db" + str(l + 1)] = beta * cache["db" + str(l + 1)] + (1 - beta) * np.square(
            gradients["db" + str(l + 1)])
        parameters["W" + str(l + 1)] -= learning_rate * (
                gradients["dW" + str(l + 1)] / (np.sqrt(cache["dW" + str(l + 1)]) + epsilon))
        parameters["b" + str(l + 1)] -= learning_rate * (
                gradients["db" + str(l + 1)] / (np.sqrt(cache["db" + str(l + 1)]) + epsilon))

    return parameters, cache


def adam(parameters, gradients, learning_rate, velocities, cache, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    for l in range(L):
        velocities["dW" + str(l + 1)] = beta1 * velocities["dW" + str(l + 1)] + (1 - beta1) * gradients[
            "dW" + str(l + 1)]
        velocities["db" + str(l + 1)] = beta1 * velocities["db" + str(l + 1)] + (1 - beta1) * gradients[
            "db" + str(l + 1)]
        corrected_velocities_dW = velocities["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        corrected_velocities_db = velocities["db" + str(l + 1)] / (1 - np.power(beta1, t))

        cache["dW" + str(l + 1)] = beta2 * cache["dW" + str(l + 1)] + (1 - beta2) * np.square(
            gradients["dW" + str(l + 1)])
        cache["db" + str(l + 1)] = beta2 * cache["db" + str(l + 1)] + (1 - beta2) * np.square(
            gradients["db" + str(l + 1)])
        corrected_cache_dW = cache["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        corrected_cache_db = cache["db" + str(l + 1)] / (1 - np.power(beta2, t))

        parameters["W" + str(l + 1)] -= learning_rate * (
                corrected_velocities_dW / (np.sqrt(corrected_cache_dW) + epsilon))
        parameters["b" + str(l + 1)] -= learning_rate * (
                corrected_velocities_db / (np.sqrt(corrected_cache_db) + epsilon))

    return parameters, velocities, cache


def mini_batch_gradient_descent_with_L2Regularization(parameters, gradients, learning_rate, m, lambda_=0.01):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                gradients["dW" + str(l + 1)] + (lambda_ / m) * parameters["W" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters


def stochastic_gradient_descent_with_L2Regularization(parameters, gradients, learning_rate, lambda_=0.7):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = (1 - learning_rate * lambda_) * parameters["W" + str(l + 1)] - learning_rate * \
                                       gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters


def momentum_with_L2Regularization(parameters, gradients, learning_rate, velocities, lambda_=0.01, beta=0.5):
    L = len(parameters) // 2
    for l in range(L):
        velocities["dW" + str(l + 1)] = beta * velocities["dW" + str(l + 1)] + (1 - beta) * gradients["dW" + str(l + 1)]
        velocities["db" + str(l + 1)] = beta * velocities["db" + str(l + 1)] + (1 - beta) * gradients["db" + str(l + 1)]

        parameters["W" + str(l + 1)] = (1 - lambda_ * learning_rate) * parameters["W" + str(l + 1)] - learning_rate * \
                                       velocities["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = (1 - lambda_ * learning_rate) * parameters["b" + str(l + 1)] - learning_rate * \
                                       velocities["db" + str(l + 1)]

    return parameters, velocities


def rmsprop_with_L2Regularization(parameters, gradients, learning_rate, cache, beta=0.9, epsilon=1e-8, lambda_=0.01):
    L = len(parameters) // 2
    for l in range(L):
        cache["dW" + str(l + 1)] = beta * cache["dW" + str(l + 1)] + (1 - beta) * np.square(
            gradients["dW" + str(l + 1)])
        cache["db" + str(l + 1)] = beta * cache["db" + str(l + 1)] + (1 - beta) * np.square(
            gradients["db" + str(l + 1)])

        parameters["W" + str(l + 1)] -= learning_rate * (
                    gradients["dW" + str(l + 1)] + lambda_ * parameters["W" + str(l + 1)]) / \
                                        (np.sqrt(cache["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] -= learning_rate * gradients["db" + str(l + 1)] / (
                    np.sqrt(cache["db" + str(l + 1)]) + epsilon)

    return parameters, cache


def adam_with_L2Regularization(parameters, gradients, learning_rate, velocities, cache, t, beta1=0.9, beta2=0.999,
                               epsilon=1e-8):
    L = len(parameters) // 2
    for l in range(L):
        velocities["dW" + str(l + 1)] = beta1 * velocities["dW" + str(l + 1)] + (1 - beta1) * gradients[
            "dW" + str(l + 1)]
        velocities["db" + str(l + 1)] = beta1 * velocities["db" + str(l + 1)] + (1 - beta1) * gradients[
            "db" + str(l + 1)]
        corrected_velocities_dW = velocities["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        corrected_velocities_db = velocities["db" + str(l + 1)] / (1 - np.power(beta1, t))

        cache["dW" + str(l + 1)] = beta2 * cache["dW" + str(l + 1)] + (1 - beta2) * np.square(
            gradients["dW" + str(l + 1)])
        cache["db" + str(l + 1)] = beta2 * cache["db" + str(l + 1)] + (1 - beta2) * np.square(
            gradients["db" + str(l + 1)])
        corrected_cache_dW = cache["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        corrected_cache_db = cache["db" + str(l + 1)] / (1 - np.power(beta2, t))

        parameters["W" + str(l + 1)] -= learning_rate * (
                    corrected_velocities_dW / (np.sqrt(corrected_cache_dW) + epsilon))
        parameters["b" + str(l + 1)] -= learning_rate * (
                    corrected_velocities_db / (np.sqrt(corrected_cache_db) + epsilon))

    return parameters, velocities, cache
