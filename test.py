import numpy as np
import lms, rls, naive_nn
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

lms_filter = lms.Lms(2,.1)

x = np.random.randint(0, 2, (500,1))
b = np.array([.7, .3], dtype=np.float32)
y = signal.lfilter(b, 1, x)

error_vector = lms_filter.train(y, x, True)
plt.plot(error_vector)
plt.ylabel('Error')
plt.xlabel('Sample')
plt.ion()
plt.show()

y_fix = lms_filter.filter(y)
print('LMS Taps: {0}'.format(lms_filter.get_taps()))
print('LMS mse = {0}'.format(mse(x,y_fix).real))

rls_filter = rls.Rls(2,.97)
rls_filter.train(y,x)
y_fix = rls_filter.filter(y)
print('RLS Taps: {0}'.format(rls_filter.get_taps()))
print('RLS mse = {0}'.format(mse(x,y_fix).real))

nn_filter = naive_nn.NaiveNN(2, .1)
nn_filter.train(y,x)
print('NN Taps: {0}'.format(lms_filter.get_taps()))
y_fix = nn_filter.filter(y)
print('NN mse = {0}'.format(mse(x,y_fix).real))
