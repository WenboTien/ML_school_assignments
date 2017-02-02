# sample_submission.py
import numpy as np


class regressor(object):
    """
    This is a sample class for miniproject 1.
    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """

    def __init__(self, data):
        self.x, self.y = data
        # Here is where your training and all the other magic should happen.
        # Once trained you should have these parameters with ready.
        # self.w = np.dot(np.dot(np.linalg.inv(np.dot(self.x.T, self.x)), self.x.T), self.y)
        # self.b = self.y - np.dot(self.x, self.w)

        self.b = np.zeros(self.y.shape)
        self.w = np.zeros((self.x.shape[1], self.y.shape[1]))
        epsilon = 0.0001
        error0 = 0

        learning_rate = 0.001
        epochs = 10000

        for _ in range(epochs):
            print('learning_rate:', learning_rate)
            for i in range(self.x.shape[0]):
                diff = (self.b[i] + np.dot(self.x[i], self.w)) - self.y[i]
                self.b[i] -= learning_rate * diff
                for j in range(self.x.shape[1]):
                    self.w[j] -= learning_rate * diff * self.x[i][j]

            error1 = 0
            for k in range(self.x.shape[0]):
                error1 += (self.y[k] - (np.dot(self.x[k], self.w) + self.b[k]))**2/2
            if abs(error1 - error0) < epsilon:
                break
            else:
                print('coss: ', abs(error1 - error0))
                learning_rate *= 0.96
                error0 = error1





    def get_params(self):
        """
        Method that should return the model parameters.
        Returns:
            tuple of numpy.ndarray: (w, b).
        Notes:
            This code will return a random numpy array for demonstration purposes.
        """
        return (self.w, self.b)

    def get_predictions(self, x):
        """
        Method should return the outputs given unseen data
        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of th
            e same length as axis 0 of
                            ``x``
        Notes:
            Temporarily returns random numpy array for demonstration purposes.
        """
        # Here is where you write a code to evaluate the data and produce predictions.
        return np.dot(x, self.w) + self.b


if __name__ == '__main__':
    pass
