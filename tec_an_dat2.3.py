import numpy as np


mu = 100
N = 1000000
s = np.random.poisson(mu, N)
print "prob(x==mu):  %10.4f , prob (x==mu-1):  %10.4f" % (len(s[s==mu]) / float(N) , len(s[s==(mu - 1)]) / float(N))

