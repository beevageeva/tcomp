import numpy as np
from math import sqrt

lam = 5 # Expectation of interval
s = np.random.poisson(lam, 100)


cmean = np.mean(s)
cmedian = np.median(s)
cvar = np.var(s)
cstd = np.std(s)

print "calculated mean = %10.4f, abs dif = %10.4f" % (cmean, abs(lam-cmean))
cstd = np.std(s)
print "calculated std = %10.4f, abs dif = %10.4f" % (cstd, abs(sqrt(lam)-cstd))
cvar = np.std(s)
print "calculated var = %10.4f, abs dif = %10.4f" % (cvar, abs(lam-cvar))

import pylab
pylab.ion() 
for bins in [5,10, 20]:
	pylab.hist(s, bins, normed=1)
	var = raw_input("Enter something: ")
	print "you entered ", var
	



