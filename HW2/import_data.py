import numpy as np

def load_data():
    train_x = np.load('dataset/x_train.npz')['arr_0']
    train_y = np.load('dataset/y_train.npz')['arr_0']
    test_x = np.load('dataset/x_test.npz')['arr_0']
    test_y = np.load('dataset/y_test.npz')['arr_0']

    train_x = train_x.reshape(12000, 28*28).T
    train_y = train_y.reshape(-1, 1)
    test_x = test_x.reshape(2000, 28*28).T
    test_y = test_y.reshape(-1, 1)

    # print(f"{train_x.shape =  }, {train_y.shape =  }, {test_x.shape =  }, {test_y.shape =  }")

    return train_x, train_y, test_x, test_y