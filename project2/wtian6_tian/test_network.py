# sample_submission.py
import numpy as np
import math

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - tanh(x) ** 2

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(data, epsilon=0.1):
    return np.maximum(epsilon * data, data)

def relu_prime(data, epsilon=0.1):
    gradients = 1. * (data > 0)
    gradients[gradients == 0] = epsilon
    return gradients

class xor_net(object):
    """
    This is a sample class for miniproject 1.
    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """

    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.params = []  # [(w,b),(w,b)]

        # layers = [784, 392, 196, 30, 2 ,1]
        layers = [2,2,1]
        self.activation = tanh
        self.activation_prime = tanh_prime
        self.weights = []
        self.bias = []

        # range of weight values (-1,1)
        for i in range(1, len(layers)-1):
            r = 2 * np.random.random((layers[i - 1], layers[i])) - 1  # add 1 for bias node
            self.weights.append(r)
            r = 2 * np.random.random((layers[i])) - 1
            self.bias.append(r)

        r = 2 * np.random.random((layers[i], layers[i + 1])) - 1
        self.weights.append(r)
        r = 2 * np.random.random((layers[i+1])) - 1
        self.bias.append(r)
        # r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        # self.weights.append(r)

        # print np.array(self.weights)[0].shape
        # print np.array(self.weights)[0][0].shape
        # print np.array(self.weights)[0][0]

        def fit(X, Y, learning_rate=0.03, epochs=2000):
            # Add column of ones to X
            # This is to add the bias unit to the input layer
            # X = np.hstack([np.ones((X.shape[0])), X])

            for k in range(epochs):
                if k % 1000 == 0: print 'epochs:', k

                # Return random integers from the discrete uniform distribution in the interval [0, low).
                i = np.random.randint(X.shape[0], high=None)
                a = [X[i]]
                # print np.array(a).shape

                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    dot_value += self.bias[l]
                    activation = self.activation(dot_value)
                    a.append(activation)

                error = Y[i] - a[-1]
                # print Y[i]
                # print a[-1][0]
                # error = -a[-1][0]*
                # error = -Y[i]*math.log(a[-1][0])-(1-Y[i])*math.log(1-a[-1][0])
                # print error
                # error = np.sum(np.nan_to_num(-Y[i]*np.log(a[-1])-(1-Y[i])*np.log(1-a[-1])))
                deltas = [error * self.activation_prime(a[-1])]


                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

                deltas.reverse()
                # print np.array(deltas)[0].shape
                # print self.bias[0].shape

                # backpropagation
                # 1. Multiply its output delta and input activation to get the gradient of the weight.
                # 2. Subtract a ratio (percentage) of the gradient from the weight.
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])  # View inputs as arrays with at least two dimensions
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * np.dot(layer.T, delta)
                    self.bias[i] = learning_rate * np.dot(self.bias[i].T, deltas[i])
        fit(self.x, self.y)




    def get_params(self):
        """
        Method that should return the model parameters.
        Returns:
            tuple of numpy.ndarray: (w, b).
        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weoghts and bias for each layer. Ordering should from input to outputt
        """
        return self.params

    def get_predictions(self, x):
        """
        Method should return the outputs given unseen data
        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        Notes:
            Temporarily returns random numpy array for demonstration purposes.
        """
        # Here is where you write a code to evaluate the data and produce predictions.
        # return np.random.randint(low=0, high=2, size=x.shape[0])

        def predict(x):
            a = np.array(x)
            for l in range(0, len(self.weights)):
                a = self.activation(np.dot(a, self.weights[l]))
            return int(round(abs(a[0])))
        return [ predict(i) for i in x]


class mlnn():
    """
    At the moment just inheriting the network above.
    """
    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.params = []  # [(w,b),(w,b)]

        # layers = [784, 392, 196, 30, 2 ,1]
        layers = [784, 392, 100, 1]
        self.activation = tanh
        self.activation_prime = tanh_prime
        self.weights = []
        self.bias = []

        # range of weight values (-1,1)
        for i in range(1, len(layers)):
            r = 2 * np.random.random((layers[i - 1], layers[i])) - 1  # add 1 for bias node
            self.weights.append(r)
            r = 2 * np.random.random((layers[i])) - 1
            self.bias.append(r)

        # r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        # self.weights.append(r)

        # print np.array(self.weights)[0].shape
        # print np.array(self.weights)[0][0].shape
        # print np.array(self.weights)[0][0]

        def fit(X, Y, learning_rate=0.02, epochs=10000):
            # Add column of ones to X
            # This is to add the bias unit to the input layer
            # X = np.hstack([np.ones((X.shape[0])), X])

            for k in range(epochs):
                if k % 1000 == 0: print 'epochs:', k

                # Return random integers from the discrete uniform distribution in the interval [0, low).
                i = np.random.randint(X.shape[0], high=None)
                a = [X[i]]
                # print np.array(a).shape

                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    dot_value += self.bias[l]
                    activation = self.activation(dot_value)
                    a.append(activation)

                error = Y[i] - a[-1]
                # print Y[i]
                # print a[-1][0]
                # error = -a[-1][0]*
                # error = -Y[i]*math.log(a[-1][0])-(1-Y[i])*math.log(1-a[-1][0])
                # print error
                # error = np.sum(np.nan_to_num(-Y[i]*np.log(a[-1])-(1-Y[i])*np.log(1-a[-1])))
                deltas = [error * self.activation_prime(a[-1])]


                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

                deltas.reverse()
                # print np.array(deltas)[0].shape
                # print self.bias[0].shape

                # backpropagation
                # 1. Multiply its output delta and input activation to get the gradient of the weight.
                # 2. Subtract a ratio (percentage) of the gradient from the weight.
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])  # View inputs as arrays with at least two dimensions
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * np.dot(layer.T, delta)
                    self.bias[i] = deltas[i]
        fit(np.array([[i/256 for i in j] for j in self.x]), self.y)
        # [[i/255 for i in j] for j in self.x]
        # fit(self.x, self.y)




    def get_params(self):
        """
        Method that should return the model parameters.
        Returns:
            tuple of numpy.ndarray: (w, b).
        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weoghts and bias for each layer. Ordering should from input to outputt
        """
        return self.params

    def get_predictions(self, x):
        """
        Method should return the outputs given unseen data
        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        Notes:
            Temporarily returns random numpy array for demonstration purposes.
        """
        # Here is where you write a code to evaluate the data and produce predictions.
        # return np.random.randint(low=0, high=2, size=x.shape[0])

        def predict(x):
            x = [i/256 for i in x]
            a = np.array(x)
            for l in range(0, len(self.weights)):
                a = np.dot(a, self.weights[l])
                a += self.bias[l]
                a = self.activation(a)
            return int(round(abs(a[0])))

        return [ predict(i) for i in x]
if __name__ == '__main__':
    pass