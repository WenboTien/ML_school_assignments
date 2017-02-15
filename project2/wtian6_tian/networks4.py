# MIT License
#
# Copyright (c) 2017 WenboTian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import math

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
        netStruc = [data.shape[1], 2, 1]
        self.num_layers = len(netStruc)
        self.biases = [np.random.randn(y, 1) for y in netStruc[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(netStruc[:-1], netStruc[1:])]
        training_data = np.array(zip([[[k] for k in i] for i in self.x], [[j] for j in self.y]))

        def SGD(epochs, eta):
            # print([[[k] for k in i] for i in self.x])
            # print([[j] for j in self.y])
            # print([i] for)
            """Train the neural network using mini-batch stochastic
            gradient descent.  The ``training_data`` is a list of tuples
            ``(x, y)`` representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If ``test_data`` is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""

            # if test_data: n_test = len(test_data)
            for j in xrange(epochs):
                # mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
                # for mini_batch in mini_batches:
                #     update_mini_batch(mini_batch, eta)
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in training_data:
                    delta_nabla_b, delta_nabla_w = backprop(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                # for x, y in training_data:
                #     delta_nabla_b, delta_nabla_w = backprop(x, y)
                #     nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                #     nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                self.weights = [w - (eta ) * nw
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (eta ) * nb
                               for b, nb in zip(self.biases, nabla_b)]

            #     print 'Epoch {0}'.format(j)
            # print self.weights
            # print self.biases

                # if test_data:
                #     print "Epoch {0}: {1} / {2}".format(
                #         j, evaluate(test_data), n_test)
                # else:
                #     print "Epoch {0} complete".format(j)

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
            # backward pass
            delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, np.array(activations[-2]).transpose())
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
                nabla_w[-l] = np.dot(delta, np.array(activations[-l - 1]).transpose())
            return (nabla_b, nabla_w)

        # def evaluate(test_data):
        #
        #     """Return the number of test inputs for which the neural
        #     network outputs the correct result. Note that the neural
        #     network's output is assumed to be the index of whichever
        #     neuron in the final layer has the highest activation."""
        #     test_results = [(np.argmax(feedforward(x)), y)
        #                     for (x, y) in test_data]
        #     return sum(int(x == y) for (x, y) in test_results)

        def cost_derivative(output_activations, y):
            """Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations."""
            return (output_activations - y)


        def sigmoid(z):
            """The sigmoid function."""
            return 1.0 / (1.0 + np.exp(-z))
            # if z > 0: return z
            # else: return 0
            # for i in range(len(z)):
            #     if z[i][0] < 0:
            #         z[i] = [0]
            # return z


        def sigmoid_prime(z):
            """Derivative of the sigmoid function."""
            return sigmoid(z) * (1 - sigmoid(z))
            # return np.log(1+np.exp(z))


        SGD(300, 0.03)


  
    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        return self.params

    def get_predictions (self, x):
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

        def feedforward(a):
            """Return the output of the network if ``a`` is input."""
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
            return a

        def sigmoid(z):
            """The sigmoid function."""
            return 1.0 / (1.0 + np.exp(-z))
            # if z > 0: return z
            # else: return 0
            # for i in range(len(z)):
            #     if z[i][0] < 0:
            #         z[i] = [0]
            # return z

        return [np.argmax(feedforward(i)) for i in x]

        # return np.random.randint(low =0, high = 2, size = x.shape[0])

class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """
    def __init__ (self, data, labels):
        super(mlnn,self).__init__(data, labels)


if __name__ == '__main__':
    pass 
