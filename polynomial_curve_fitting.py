import numpy as np
from numpy.linalg import *
from random import *
from math import *
import matplotlib.pyplot as plt

mu, sigma = 0.0, 0.1 #average and scale of normal distribution
IsRegular, numOfSample, degree = 0.3, 10, 9 #important parameters
sample_xi, sample_y = [], []
for i in range(numOfSample):
    sample_xi.append(uniform(0,2*pi)) #random sampling on the interval [0,2pi]
sample_x = np.array(sorted(sample_xi))
for i in range(numOfSample):
    sample_y.append(sin(sample_x[i]) + np.random.normal(mu,sigma, numOfSample)[i]) #random.normal are noises
#compute MATRIX A  Aw=b
#w=(w0,w1,w2,...,w_degree)
AA = np.ones((numOfSample), dtype = int)
for i in range(1, degree + 1):
    AA = np.vstack((AA, sample_x ** i))
A = np.dot(AA, AA.T) + IsRegular * np.eye(degree + 1)
b = np.dot(AA, sample_y)
w = np.dot(inv(A),b)

y_fit = np.zeros(numOfSample)
for j in range(numOfSample):
    for i in range(degree + 1):
        y_fit[j] = y_fit[j] + w[i] * (sample_x[j] ** i)
#drawing
xx = np.linspace(0, 2*pi ,200)
y_fit_curve = np.zeros(len(xx))
for j in range(len(xx)):
    for i in range(degree + 1):
        y_fit_curve[j] = y_fit_curve[j] + w[i] * (xx[j] ** i)
plt.plot(sample_x, y_fit, 'r^')
plt.plot(xx, np.sin(xx), color = 'blue', linewidth = 1)
plt.plot(xx, y_fit_curve, color = 'green', linewidth = 1)
plt.axis([0, 2*pi, -1.2, 1.2])
plt.show()