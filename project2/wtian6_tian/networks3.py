# sample_submission.py
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - tanh(x) ** 2

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

        layers = [2, 2, 1]
        self.activation = tanh
        self.activation_prime = tanh_prime
        self.weights = []

        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)

        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

        def fit(X, Y, learning_rate=0.03, epochs=10000):
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            for k in range(epochs):
                if k % 1000 == 0: print 'epochs:', k

                # Return random integers from the discrete uniform distribution in the interval [0, low).
                i = np.random.randint(X.shape[0], high=None)
                a = [X[i]]

                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)

                error = Y[i] - a[-1]
                deltas = [error * self.activation_prime(a[-1])]

                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))
                deltas.reverse()

                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * np.dot(layer.T, delta)
        fit(self.x, self.y)
        self.params = [(i[:-1], i[-1]) for i in self.weights]



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
            a = np.concatenate((np.ones(1), np.array(x)))
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
        self.params = []
        layers = [784, 392, 196, 30, 1]
        layers = [784, 450, 200, 50, 1]
        self.activation = tanh
        self.activation_prime = tanh_prime
        self.weights = []

        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1  # add 1 for bias node
            self.weights.append(r)

        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

        def fit(X, Y, learning_rate=0.02, epochs=10000):
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            for k in range(epochs):
                if k % 1000 == 0: print 'epochs:', k

                # Return random integers from the discrete uniform distribution in the interval [0, low).
                i = np.random.randint(X.shape[0], high=None)
                a = [X[i]]

                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)

                error = Y[i] - a[-1]
                deltas = [error * self.activation_prime(a[-1])]

                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))
                deltas.reverse()

                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * np.dot(layer.T, delta)
        fit(np.array([[i/256 for i in j] for j in self.x]), self.y)
        self.params = [(i[:-1], i[-1]) for i in self.weights]

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
            a = np.concatenate((np.ones(1), np.array(x)))
            for l in range(0, len(self.weights)):
                a = self.activation(np.dot(a, self.weights[l]))
            return int(round(abs(a[0])))

        return [ predict(i) for i in x]
if __name__ == '__main__':
    pass