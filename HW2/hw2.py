import numpy as np
from matplotlib import pyplot as plt

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
        velocities["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        velocities["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return velocities

def initialize_cache(parameters):
    L = len(parameters) // 2
    cache = {}
    for l in range(L):
        cache["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        cache["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return cache

def Binary_Cross_Entropy(AL, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(AL) + (1 - y_true) * np.log(1 - AL)) / m
    loss = np.squeeze(loss)
    return loss

def Binary_Cross_Entropy_derivative(AL, y_true):
    return - (np.divide(y_true, AL) - np.divide(1 - y_true, 1 - AL))

def gradient_descent(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * gradients["db" + str(l + 1)]
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
        cache["dW" + str(l + 1)] = beta * cache["dW" + str(l + 1)] + (1 - beta) * np.square(gradients["dW" + str(l + 1)])
        cache["db" + str(l + 1)] = beta * cache["db" + str(l + 1)] + (1 - beta) * np.square(gradients["db" + str(l + 1)])
        parameters["W" + str(l + 1)] -= learning_rate * (gradients["dW" + str(l + 1)] / (np.sqrt(cache["dW" + str(l + 1)]) + epsilon))
        parameters["b" + str(l + 1)] -= learning_rate * (gradients["db" + str(l + 1)] / (np.sqrt(cache["db" + str(l + 1)]) + epsilon))
    return parameters, cache


def adam(parameters, gradients, learning_rate, velocities, cache, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    for l in range(L):
        velocities["dW" + str(l + 1)] = beta1 * velocities["dW" + str(l + 1)] + (1 - beta1) * gradients["dW" + str(l + 1)]
        velocities["db" + str(l + 1)] = beta1 * velocities["db" + str(l + 1)] + (1 - beta1) * gradients["db" + str(l + 1)]
        corrected_velocities_dW = velocities["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        corrected_velocities_db = velocities["db" + str(l + 1)] / (1 - np.power(beta1, t))

        cache["dW" + str(l + 1)] = beta2 * cache["dW" + str(l + 1)] + (1 - beta2) * np.square(gradients["dW" + str(l + 1)])
        cache["db" + str(l + 1)] = beta2 * cache["db" + str(l + 1)] + (1 - beta2) * np.square(gradients["db" + str(l + 1)])
        corrected_cache_dW = cache["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        corrected_cache_db = cache["db" + str(l + 1)] / (1 - np.power(beta2, t))

        parameters["W" + str(l + 1)] -= learning_rate * (corrected_velocities_dW / (np.sqrt(corrected_cache_dW) + epsilon))
        parameters["b" + str(l + 1)] -= learning_rate * (corrected_velocities_db / (np.sqrt(corrected_cache_db) + epsilon))
    return parameters, velocities, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
    elif activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "tanh":
        A = np.tanh(Z)
    elif activation == "leaky_relu":
        A = np.maximum(0.01 * Z, Z)

    cache = (A_prev, W, b, Z)
    return A, cache


def forward(X, parameters, activation_fw):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_fw)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation_fw)
    caches.append(cache)

    return AL, caches

def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]
    if activation == "sigmoid_derivative":
        dZ = dA * (1 / (1 + np.exp(-Z)) * (1 - 1 / (1 + np.exp(-Z))))
    elif activation == "relu_derivative":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    elif activation == "tanh_derivative":
        dZ = 1 - np.power(tanh(Z), 2)
    elif activation == "leaky_relu_derivative":
        dZ = np.where(Z > 0, 1, 0.01)


    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward(AL, Y, caches, activation_function):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = Binary_Cross_Entropy_derivative(AL, Y)
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation_bw)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation_bw)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def predict(X, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))
    probas, _ = forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    return p


def train(X_train, Y_train, layers_dims, learning_rate, num_iterations, activation_function, initialize_parameters,
          update_parameters):
    np.random.seed(1)
    losses = []
    activation_bw = activation_function + "_derivative"
    activation_fw = activation_function

    print(f"{activation_fw = } and {activation_bw = }")

    if initialize_parameters == 'initialize_parameters_zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialize_parameters == 'initialize_parameters_random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialize_parameters == 'initialize_parameters_xavier':
        parameters = initialize_parameters_xavier(layers_dims)
    elif initialize_parameters == 'initialize_parameters_HE':
        parameters = initialize_parameters_HE(layers_dims)

    if update_parameters == 'gradient_descent':
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters = gradient_descent(parameters, grads, learning_rate)
            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                losses.append(loss)

    elif update_parameters == 'momentum':
        velocities = initialize_velocities(parameters)
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters, velocities = momentum(parameters, grads, learning_rate, velocities)

            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                losses.append(loss)

    elif update_parameters == 'rmsprop':
        cache = initialize_cache(parameters)
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters, cache = rmsprop(parameters, grads, learning_rate, cache)

            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                losses.append(loss)

    elif update_parameters == 'adam':
        velocities = initialize_velocities(parameters)
        cache = initialize_cache(parameters)
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters, velocities, cache = adam(parameters, grads, learning_rate, velocities, cache, i)

            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                losses.append(loss)

    plt.plot(np.squeeze(losses))
    plt.ylabel('Loss')
    plt.xlabel('Epochs(x100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    predictions_train = predict(X_train, parameters, activation_fw)
    print("Accuracy: {} %".format(100 - np.mean(np.abs(predictions_train - Y_train)) * 100))

    return parameters

def load_data():
    train_loaded = np.load(f'PandaOrBear/train_data.npz')
    test_loaded = np.load(f'PandaOrBear/test_data.npz')

    train_x, train_y = train_loaded['x'].T, train_loaded['y']
    test_x, test_y = test_loaded['x'].T, test_loaded['y']

    # Standarize data
    train_x = train_x / 255.
    test_x = test_x / 255.

    # Shuffle data (no effect for full batch method)
    train_indices = np.random.permutation(train_x.shape[1])
    train_x = train_x[:, train_indices]
    train_y = train_y[train_indices]

    test_indices = np.random.permutation(test_x.shape[1])
    test_x = test_x[:, test_indices]
    test_y = test_y[test_indices]

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = load_data()


activation_function = 'sigmoid'
input_shape = train_x.shape[1]
layers_dims = [input_shape, 30, 30, 20, 10, 10, 1]
learning_rate = 0.005
num_iterations = 1500
initialize_parameters = 'initialize_parameters_random'
update_parameters = 'gradient_descent'

parameters = train(train_x, train_y, layers_dims, learning_rate, num_iterations, activation_function, initialize_parameters, update_parameters)
predictions_test = predict(test_x, parameters)
# print("Test Accuracy: {} %".format(100 - np.mean(np.abs(predictions_test - test_y)) * 100))

# train_x.shape is (12000, 28, 28)