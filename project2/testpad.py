import numpy as np
import mnist_loader
def sigmoid2(z):
    """The sigmoid function."""
    return [1.0 / (1.0 + np.exp(-z[0]))]

def sigmoid(z):
    """The sigmoid function."""
    return [1.0 / (1.0 + np.exp(-z))]

# a = [[1],[-1]]
# b= map(sigmoid2, np.array(a))
# print b
# print sigmoid(np.array(a))
# for i in range(len(a)):
#     print a[i]
#     if a[i][0] > 0: a[i] = [2]
#     elif a[i][0] < 0: a[i] = [0]
#
# print a
training_data, validation_data, tset_data = mnist_loader.load_data_wrapper()

print(2 * np.random.random((5)) + 2 * np.random.random((5))-1)