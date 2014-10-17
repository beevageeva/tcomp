import numpy as np

mu, sigma = 0.5, 0.2 # mean and standard deviation
for i in range(2,7):
	N = float(10**i)
	s = np.random.normal(mu, sigma, N)
	print len(s[abs(s-mu)<sigma]) / N
