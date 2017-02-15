# sample_submission.py
import numpy as np


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
        # training_data = np.array(zip(self.x, np.array([[i] for i in self.y])))
        training_data = zip([[[k] for k in i] for i in self.x], [[j] for j in self.y])
        sizes = [data.shape[1], 10, 1]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        def sigmoid(z):
            """The sigmoid function."""
            return 1.0 / (1.0 + np.exp(-z))

        def sigmoid_prime(z):
            """Derivative of the sigmoid function."""
            return sigmoid(z) * (1 - sigmoid(z))

        def feedforward(a):
            """Return the output of the network if ``a`` is input."""
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
            return a

        def SGD(training_data, epochs, mini_batch_size, eta):
            """Train the neural network using mini-batch stochastic
            gradient descent.  The ``training_data`` is a list of tuples
            ``(x, y)`` representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If ``test_data`` is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
            n = len(training_data)
            for j in xrange(epochs):
                mini_batches = [
                    training_data[k:k + mini_batch_size]
                    for k in xrange(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    update_mini_batch(mini_batch, eta)

                print 'Epoch {0}'.format(j)

        def update_mini_batch(mini_batch, eta):
            """Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]

        def backprop(x, y):
            """Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x]  # list to store all the activations, layer by layer
            zs = []  # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass n
            delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                # print(delta)
                # print(np.array(activations[-l - 1]).transpose())
                nabla_w[-l] = np.dot(delta, np.array(activations[-l - 1]).transpose())
            return (nabla_b, nabla_w)

        def evaluate(test_data):
            """Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation."""
            test_results = [(np.argmax(feedforward(x)), y)
                            for (x, y) in test_data]
            return sum(int(x == y) for (x, y) in test_results)

        def cost_derivative(output_activations, y):
            """Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations."""
            return (output_activations - y)


        SGD(training_data, 30, 10, 3)









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
        return np.random.randint(low=0, high=2, size=x.shape[0])

        def feedforward(a):
            """Return the output of the network if ``a`` is input."""
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
            return a

        def sigmoid(z):
            """The sigmoid function."""
            return 1.0 / (1.0 + np.exp(-z))

        return [np.argmax(feedforward(i)) for i in x]


class mlnn(xor_net):
    """
    At the moment just inheriting the network above.
    """

    def __init__(self, data, labels):
        super(mlnn, self).__init__(data, labels)


if __name__ == '__main__':
    pass