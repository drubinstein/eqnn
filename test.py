import numpy as np
import lms
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

lms_filter = lms.Lms(2,.1)

x = np.random.randint(0, 2, (500,1))
b = np.array([.7, .3], dtype=complex)
y = signal.lfilter(b, 1, x)
#print(x.shape)
#print(b.shape)
#print(y)

error_vector = lms_filter.train(y, x, True)
plt.plot(error_vector)
plt.ylabel('Error')
plt.xlabel('Sample')
plt.ion()
plt.show()

y_fix = lms_filter.filter(y)
print('mse = {0}'.format(mse(x,y_fix).real))
