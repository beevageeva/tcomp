import numpy as np

mu, sigma = 0.5, 0.2 # mean and standard deviation
for i in range(2,7):
	s = np.random.normal(mu, sigma, 10**i)
	mx2 = np.mean(s**2)
	print "N=%d, sigma ** 2 = %10.4f,  mean(x**2) - mu**2 = %10.4f, absdif = %10.4f " % (10 ** i, sigma**2 , mx2 - mu**2, abs(sigma**2 + mu **2 - mx2))
