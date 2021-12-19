import numpy
import scipy.special
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.pyplot
import datetime


# neural network definition
class mlp:

    # init neural network
    def __init__(self, X, Y, hidden=1, alpha=0.1):
        self.X = X
        self.Y = Y
        # set number of nodes in each input, hidden and output layer
        self.inodes = X.shape[1]
        self.onodes = Y.shape[1]
        self.hnodes = hidden

        # learning rate
        self.lr = alpha

        # linkweight matrices, wih and who
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function (sigmoid function expit)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

    # train the neural network
    def train(self):
        inputs_list = self.X
        targets_list = self.Y
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is target - actual
        output_errors = targets - final_outputs
        # hidden layer error is the outputs error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                                        numpy.transpose(inputs))

        pass

    # query the neural network
    def calc_saida(self, inputs_list):
        # convert inputs_list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

