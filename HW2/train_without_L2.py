from matplotlib import pyplot as plt
from initialize_parameters import *
from optimizer import *
from backward import *
from predict import *


def train(X_train, Y_train, layers_dims, learning_rate, num_iterations, activation_function, initialize_parameters,
          update_parameters):
    np.random.seed(1)
    losses = []
    accuracies = []
    activation_bw = activation_function + "_derivative"
    activation_fw = activation_function

    # print(f"{activation_fw = } and {activation_bw = }")

    if initialize_parameters == 'initialize_parameters_zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialize_parameters == 'initialize_parameters_random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialize_parameters == 'initialize_parameters_xavier':
        parameters = initialize_parameters_xavier(layers_dims)
    elif initialize_parameters == 'initialize_parameters_HE':
        parameters = initialize_parameters_HE(layers_dims)

    # print(f"{parameters.shape = }")
    # print(f"{parameters['W1'].shape = }")
    # print(f"{parameters['W2'].shape = }")
    # print(f"{parameters['W3'].shape = }")
    # print(f"{parameters['b1'].shape = }")
    # print(f"{parameters['b2'].shape = }")
    # print(f"{parameters['b3'].shape = }")

    if update_parameters == 'mini_batch_gradient_descent':
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            # print(f"{AL.shape = }")
            # print(f"{caches.shape = }")

            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters = mini_batch_gradient_descent(parameters, grads, learning_rate)
            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                predictions_train = predict(X_train, parameters, activation_fw)
                accuracy = compute_accuracy(predictions_train, Y_train)
                losses.append(loss)
                accuracies.append(accuracy)


    elif update_parameters == 'stochastic_gradient_descent':
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters = stochastic_gradient_descent(parameters, grads, learning_rate)
            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                predictions_train = predict(X_train, parameters, activation_fw)
                accuracy = compute_accuracy(predictions_train, Y_train)
                losses.append(loss)
                accuracies.append(accuracy)


    elif update_parameters == 'momentum':
        velocities = initialize_velocities(parameters)
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters, velocities = momentum(parameters, grads, learning_rate, velocities)

            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                predictions_train = predict(X_train, parameters, activation_fw)
                accuracy = compute_accuracy(predictions_train, Y_train)
                losses.append(loss)
                accuracies.append(accuracy)


    elif update_parameters == 'rmsprop':
        cache = initialize_cache(parameters)
        for i in range(0, num_iterations):
            AL, caches = forward(X_train, parameters, activation_fw)
            loss = Binary_Cross_Entropy(AL, Y_train)
            grads = backward(AL, Y_train, caches, activation_bw)
            parameters, cache = rmsprop(parameters, grads, learning_rate, cache)

            if i % 100 == 0:
                print(f"epoch {i} | loss : {loss}")
                predictions_train = predict(X_train, parameters, activation_fw)
                accuracy = compute_accuracy(predictions_train, Y_train)
                losses.append(loss)
                accuracies.append(accuracy)


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
                predictions_train = predict(X_train, parameters, activation_fw)
                accuracy = compute_accuracy(predictions_train, Y_train)
                losses.append(loss)
                accuracies.append(accuracy)

    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(losses))
    plt.ylabel('Loss')
    plt.xlabel('Epochs(x100)')
    plt.title("Learning rate =" + str(learning_rate))

    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(accuracies))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs(x100)')

    plt.show()

    predictions_train = predict(X_train, parameters, activation_fw)
    print("Accuracy: {} %".format(100 - np.mean(np.abs(predictions_train - Y_train)) * 100))

    return parameters
