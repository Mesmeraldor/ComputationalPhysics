import numpy as np
import matplotlib.pyplot as plt

def CorrelationFunction(magnetization):
    '''
    Get the correlation function given a set of magnetisation values over different time values.

    Parameters:
    magnetisation : (numpy_array) Equally spaced magnetisation values.

    Returns: (numpy_array) The correlation function for all time differences, where dt=1.
    '''

    N = len(magnetization)

    correlation_function = np.zeros(N-1)

    for i in range(1, N):
        correlation_function[i-1] = (1/(N-i) * np.sum(magnetization[i:] * magnetization[:-i]) -
                                     1/(N-i)**2 * np.sum(magnetization[i:]) * np.sum(magnetization[:-i]))

    return correlation_function

N = 1000
x_array = np.zeros(N)
x = 0
for i in range(N):
    x_array[i] = x
    x += np.random.normal(0, 1)

correlation_function = CorrelationFunction(x_array)

plt.plot(correlation_function)
plt.show()