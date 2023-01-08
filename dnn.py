import numpy as np
from .utils import sigmoid, sigmoid_backward, relu, relu_backward, leaky_relu, leaky_relu_backward, tanh, tanh_backward, softmax, softmax_backward
import math

class dnn :
    # hyper parameters
    learning_rate = 0.05
    layers = [{'size': 0}]
    m = 500
    nx = 0
    # regularization hyper params
    regularization = False
    lambd = 0.1
    dropout = False
    keep_prob = 1
    keep_probs = {}
    # params
    parameters = {}
    cache = {}
    # batch
    mini_batch = False
    batch_size = 500
    # optimization
    adam = False
    momentum = False
    beta_1 = 0.9
    rms = False
    beta_2 = 0.999
    epsilon = 1e-8
    # others
    verbose = False
    verbose_int = 10
    costs = {}

    def __init__(self, X='', Y='') :
        self.X = X
        self.Y = Y
        self.nx = X.shape[0]
        self.layers[0]['size'] = self.nx
        self.initial_factor = 0.5
        np.random.seed(3)

    def add_layer(self, node_size=3, activation='relu') :
        self.layers.append({'size': node_size, 'activation': activation})

    def initialize(self, params={}) :
        if len(params) > 0 :
            self.parameters = params
        else :
            for i in range(1, len(self.layers)) :
                if self.initial_factor :
                    initial_factor = self.initial_factor
                else :
                    initial_factor = np.sqrt(1 / self.layers[i-1]['size'])
                self.parameters['W' + str(i)] = np.random.randn(self.layers[i]['size'], self.layers[i-1]['size']) * initial_factor
                self.cache['dW' + str(i)] = np.zeros((self.layers[i]['size'], self.layers[i-1]['size']))
                self.cache['VdW' + str(i)] = np.zeros((self.layers[i]['size'], self.layers[i-1]['size']))
                self.cache['SdW' + str(i)] = np.zeros((self.layers[i]['size'], self.layers[i-1]['size']))
                self.parameters['b' + str(i)] = np.random.randn(self.layers[i]['size'], 1)
                self.cache['db' + str(i)] = np.zeros((self.layers[i]['size'], 1))
                self.cache['Vdb' + str(i)] = np.zeros((self.layers[i]['size'], 1))
                self.cache['Sdb' + str(i)] = np.zeros((self.layers[i]['size'], 1))

    def forward_prop(self, e) :
        for i in range(1, len(self.layers)) :
            self.cache['Z' + str(i)] = np.dot(self.parameters['W' + str(i)], self.cache['A' + str(i - 1)]) + self.parameters['b' + str(i)]
            if self.layers[i]['activation'] == 'relu' :
                self.cache['A' + str(i)] = relu(self.cache['Z' + str(i)])[0]
            elif self.layers[i]['activation'] == 'leaky_relu' :
                self.cache['A' + str(i)] = leaky_relu(self.cache['Z' + str(i)])[0]
            elif self.layers[i]['activation'] == 'tanh' :
                self.cache['A' + str(i)] = tanh(self.cache['Z' + str(i)])[0]
            elif self.layers[i]['activation'] == 'softmax' :
                self.cache['A' + str(i)] = softmax(self.cache['Z' + str(i)])[0]
            else :
                self.cache['A' + str(i)] = sigmoid(self.cache['Z' + str(i)])[0]
            if self.dropout and 'D' + str(i) in self.keep_probs :
                self.cache['D' + str(i)] = np.random.rand(self.cache['A' + str(i)].shape[0], self.cache['A' + str(i)].shape[1])
                self.cache['D' + str(i)] = self.cache['D' + str(i)] < self.keep_probs['D' + str(i)]
                self.cache['A' + str(i)] = self.cache['A' + str(i)] * self.cache['D' + str(i)] / self.keep_probs['D' + str(i)]

    def backward_prop(self, e) :
        for i in range(len(self.layers) - 1, 0, -1) :
            # last layer
            if i == len(self.layers) - 1 :
                self.cache['dA' + str(i)] = - (self.cache['Y'] / self.cache['A' + str(i)]) + ((1 - self.cache['Y']) / (1 - self.cache['A' + str(i)]))
            else :
                self.cache['dA' + str(i)] = np.dot(self.parameters['W' + str(i + 1)].T, self.cache['dZ' + str(i + 1)])
            if self.dropout and 'D' + str(i) in self.keep_probs :
                self.cache['dA' + str(i)] = self.cache['dA' + str(i)] * self.cache['D' + str(i)] / self.keep_probs['D' + str(i)]

            if self.layers[i]['activation'] == 'relu' :
                self.cache['dZ' + str(i)] = relu_backward(self.cache['dA' + str(i)], self.cache['Z' + str(i)])
            elif self.layers[i]['activation'] == 'leaky_relu' :
                self.cache['dZ' + str(i)] = leaky_relu_backward(self.cache['dA' + str(i)], self.cache['Z' + str(i)])
            elif self.layers[i]['activation'] == 'tanh' :
                self.cache['dZ' + str(i)] = tanh_backward(self.cache['dA' + str(i)], self.cache['Z' + str(i)])
            elif self.layers[i]['activation'] == 'softmax' :
                self.cache['dZ' + str(i)] = softmax_backward(self.cache['dA' + str(i)], self.cache['Z' + str(i)])
            else :
                self.cache['dZ' + str(i)] = sigmoid_backward(self.cache['dA' + str(i)], self.cache['Z' + str(i)])

            self.cache['dW' + str(i)] = 1 / self.m * np.dot(self.cache['dZ' + str(i)], self.cache['A' + str(i - 1)].T)
            if self.regularization :
                self.cache['dW' + str(i)] = self.cache['dW' + str(i)] + self.lambd / self.m * self.parameters['W' + str(i)]
            self.cache['db' + str(i)] = 1 / self.m * np.sum(self.cache['dZ' + str(i)], axis=1, keepdims=True)

        #update
        for i in range(1, len(self.layers)) :
            if self.momentum or self.rms :
                dW_factor = 1
                db_factor = 1
                if self.momentum :
                    self.cache['VdW' + str(i)] = self.beta_1 * self.cache['VdW' + str(i)] + (1 - self.beta_1) * self.cache['dW' + str(i)]
                    dW_factor *= self.cache['VdW' + str(i)] / (1 - self.beta_1 ** (e + 1))
                    self.cache['Vdb' + str(i)] = self.beta_1 * self.cache['Vdb' + str(i)] + (1 - self.beta_1) * self.cache['db' + str(i)]
                    db_factor *= self.cache['Vdb' + str(i)] / (1 - self.beta_1 ** (e + 1))
                if self.rms :
                    self.cache['SdW' + str(i)] = self.beta_2 * self.cache['SdW' + str(i)] + (1 - self.beta_2) * np.power(self.cache['dW' + str(i)], 2)
                    dW_factor *= 1 / (np.sqrt(self.cache['SdW' + str(i)] / (1 - self.beta_2 ** (e + 1))) + self.epsilon)
                    self.cache['Sdb' + str(i)] = self.beta_2 * self.cache['Sdb' + str(i)] + (1 - self.beta_2) * np.power(self.cache['db' + str(i)], 2)
                    db_factor *= 1 / (np.sqrt(self.cache['Sdb' + str(i)] / (1 - self.beta_2 ** (e + 1))) + self.epsilon)
                self.parameters['W' + str(i)] = self.parameters['W' + str(i)] - self.learning_rate * dW_factor
                self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - self.learning_rate * db_factor
            else :
                self.parameters['W' + str(i)] = self.parameters['W' + str(i)] - self.learning_rate * self.cache['dW' + str(i)]
                self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - self.learning_rate * self.cache['db' + str(i)]

    def cost(self) :
        if self.Y.shape[0] > 1 :
            C = np.sum((self.cache['Y'] * np.log(self.cache['A' + str(len(self.layers) - 1)])))
        else :
            C = np.sum((self.cache['Y'] * np.log(self.cache['A' + str(len(self.layers) - 1)])) + ((1 - self.cache['Y']) * np.log(1 - self.cache['A' + str(len(self.layers) - 1)])))
        L = (-1 / self.m) * C
        if self.regularization :
            for j in range(1, len(self.layers)) :
                L = L + np.sum(np.square(self.parameters['W' + str(j)])) * (self.lambd / (2 * self.m))
        L = np.squeeze(L)

        return L

    def train(self, epoch=1000, params={}) :
        self.initialize(params=params)

        # sorting out some prerequisites
        if self.adam :
            self.momentum = True
            self.rms = True

        if self.mini_batch :
            iter = epoch * math.ceil(self.X.shape[1] / self.batch_size)
        else :
            iter = epoch
            self.cache['A0'] = self.X
            self.cache['Y'] = self.Y

        for e in range(epoch) :
            # mini batch enabled
            if self.mini_batch :
                permutation = list(np.random.permutation(self.X.shape[1]))
                shuffled_X = self.X[:, permutation]
                shuffled_Y = self.Y[:, permutation]
                iters_per_epoch = math.ceil(self.X.shape[1] / self.batch_size)
                for k in range(iters_per_epoch) :
                    i = e * iters_per_epoch + k
                    X_batch = shuffled_X[:, k * self.batch_size : (k + 1) * self.batch_size]
                    Y_batch = shuffled_Y[:, k * self.batch_size : (k + 1) * self.batch_size]
                    self.m = X_batch.shape[1]
                    self.cache['A0'] = X_batch
                    self.cache['Y'] = Y_batch
                    #foward prop
                    self.forward_prop(i)
                    #backward prop
                    self.backward_prop(i)
            # mini batch disabled
            else :
                i = e
                self.m = self.X.shape[1]
                #foward prop
                self.forward_prop(i)
                #backward prop
                self.backward_prop(i)

            if self.verbose and e % (epoch // self.verbose_int) == 0 :
                L = self.cost()
                self.costs[e] = L
                print('Cost after ' + str(e) + ' epochs: ' + str(L))

        return self.parameters

    def predict(self, X='') :
        A = X
        for i in range(1, len(self.layers)) :
            Z = np.dot(self.parameters['W' + str(i)], A) + self.parameters['b' + str(i)]
            if self.layers[i]['activation'] == 'relu' :
                A = relu(Z)[0]
            elif self.layers[i]['activation'] == 'leaky_relu' :
                A = leaky_relu(Z)[0]
            elif self.layers[i]['activation'] == 'tanh' :
                A = tanh(Z)[0]
            elif self.layers[i]['activation'] == 'softmax' :
                A = softmax(Z)[0]
            else :
                A = sigmoid(Z)[0]

        return A

if __name__ == '__main__':
    nn = nn(np.array([]), np.array([]))
