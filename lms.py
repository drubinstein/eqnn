import numpy as np
from scipy import signal

class Lms:
    #h is an orderx1 sized array
    #mu is the step size
    def __init__(self,order=5, mu=.2):
        self.order = order
        self.h = np.zeros((self.order,1),dtype=complex)
        self.mu = mu

    def train(self, x, desired, return_err = False):
        #The size of x and desired must be the same
        assert x.shape == desired.shape, "Shape of x, {0}, must be same as length of desired, {1}".format(x.shape, desired.shape)
        assert x.shape[0] > self.order, "Length dim of x, {0}, must be greater than the lms filter order, {1}".format(x.shape, self.order)

        if return_err:
            ev = np.zeros(x.shape)
        else:
            ev = None

        for n in range(0,x.shape[0]-self.order):
            x_n = np.array(x[n:n+self.order,...])
            e = desired[n,...] - (np.conj(np.transpose(self.h)) @ x_n)
            if return_err:
                ev[n] = e.real
            self.h = self.h + self.mu * x_n * np.conj(e)

        return ev

    def filter(self, x):
        return signal.lfilter(self.h[:,0], 1, x[:,0])

    def get_taps(self):
        return self.h

    def reset_taps(self):
        self.h = np.zeros((self.order,1))

    def set_mu(self, mu : float):
        self.mu = mu

    def set_order(self, order : int):
        self.order = order
