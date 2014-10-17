import numpy as np

mu, sigma = 0.5, 0.2 # mean and standard deviation
s = np.random.normal(mu, sigma, 100)
cmean = np.mean(s)
print "calculated mean = %10.4f, abs dif = %10.4f" % (cmean, abs(mu-cmean))
cstd = np.std(s)
print "calculated std = %10.4f, abs dif = %10.4f" % (cstd, abs(sigma-cstd))
cvar = np.var(s)
print "calculated var = %10.4f, abs dif = %10.4f" % (cvar, abs(sigma**2-cvar))

import pylab
pylab.ion() 
for bins in [5,10, 20]:
	pylab.hist(s, bins, normed=1)
	var = raw_input("Enter something: ")
	print "you entered ", var
	



