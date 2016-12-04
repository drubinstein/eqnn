import numpy as np
import tensorflow as tf

class NaiveNN:
    #h is an orderx1 sized array
    #learning_rate is the forgetting factor
    #delta is the initial value for P(0)
    def __init__(self, n_weights = 5, learning_rate = .1):
        self.n_weights = n_weights
        self.learning_rate = learning_rate
        self.X = tf.placeholder(tf.float32, [self.n_weights,1])
        self.Y = tf.placeholder(tf.float32, [1]) # one output per clock
        self.weights = {
            'out' : tf.Variable(tf.zeros([self.n_weights, 1], dtype=tf.float32))
        }
        self.pred = self.mlp(self.X, self.weights)
        self.cost = tf.pow(tf.abs(self.pred-self.Y),2)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.sess.run(self.init)

    def mlp(self, x, weights):
        return tf.matmul(tf.transpose(x), weights['out'])

    def train(self, x, desired):
        #The size of x and desired must be the same
        assert x.shape == desired.shape, "Shape of x, {0}, must be same as length of desired, {1}".format(x.shape, desired.shape)
        assert x.shape[0] > self.n_weights, "Length dim of x, {0}, must be greater than the lms filter order, {1}".format(x.shape, self.n_weights)

        for n in range(0,x.shape[0]-self.n_weights):
            x_n = x[n:n+self.n_weights,...]
            self.sess.run(self.optimizer, feed_dict={self.X: x_n, self.Y: desired[n,...]})

        return None

    def filter(self, x):
        y = np.zeros(x.shape)
        for n in range(0,x.shape[0]-self.n_weights):
            x_n = x[n:n+self.n_weights,...]
            y[n] = self.sess.run(self.pred, feed_dict={self.X: x_n})
        return np.round(y)

    def get_taps(self):
        return self.weights

    def reset_taps(self):
        self.weights = {
            'out' : tf.Variable(tf.zeros([self.n_weights, 1]))
        }

    def set_learning_rate(self, learning_rate : float):
        self.__init__(self.n_weights, learning_rate)

    def set_order(self, order : int):
        self.__init__(order, self.learning_rate)
