from backward import *
from import_data import load_data
from train_without_L2 import train
from predict import predict, compute_accuracy

# Identity
# sigmoid
# ReLU
# tanh
# leaky_relu

# ------------------ load data
train_x, train_y, test_x, test_y = load_data()

# ------------------ setup
activation_function = 'sigmoid'
input_layer = train_x.shape[0]
print(f"{input_layer =  }")
layers_dims = [input_layer, 5, 5, 1]
learning_rate = 0.09
num_iterations = 800
init_parameter = 'initialize_parameters_random'
update = 'mini_batch_gradient_descent'

# ------------------ run
parameters = train(train_x, train_y, layers_dims, learning_rate, num_iterations, activation_function, init_parameter,
                   update)
predictions_test = predict(test_x, parameters, activation_function)
print("Test Accuracy: {} %".format(100 - np.mean(np.abs(predictions_test - test_y)) * 100))
