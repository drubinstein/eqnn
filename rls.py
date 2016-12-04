import numpy as np
from scipy import signal

class Rls:
    #h is an orderx1 sized array
    #f_factor is the forgetting factor
    #delta is the initial value for P(0)
    def __init__(self,order=5, f_factor = .97, delta=.5):
        self.order = order
        self.f_factor = f_factor
        self.w = np.zeros((self.order,1),dtype=np.float32)
        self.k = np.zeros((self.order,1),dtype=np.float32)
        self.P = 1/delta * np.eye(self.order)

    def train(self, x, desired):
        #The size of x and desired must be the same
        assert x.shape == desired.shape, "Shape of x, {0}, must be same as length of desired, {1}".format(x.shape, desired.shape)
        assert x.shape[0] > self.order, "Length dim of x, {0}, must be greater than the lms filter order, {1}".format(x.shape, self.order)

        for n in range(0,x.shape[0]-self.order):
            x_n = np.array(x[n:n+self.order,...])
            alpha = desired[n,...] - (np.transpose(x_n) @ self.w)
            g = self.P*np.conj(x_n)*np.linalg.pinv(self.f_factor + np.transpose(x_n)*self.P*np.conj(x_n))
            self.P = 1/self.f_factor*self.P-g*np.transpose(x_n)/self.f_factor*self.P
            self.w = self.w + alpha*g

        return None

    def filter(self, x):
        return signal.lfilter(self.w[:,0], 1, x[:,0])

    def get_taps(self):
        return self.w

    def reset_taps(self):
        self.h = np.zeros((self.order,1))

    def set_f_factor(self, f_factor : float):
        self.f_factor = f_factor

    def set_order(self, order : int):
        self.order = order
        self.reset_taps()
