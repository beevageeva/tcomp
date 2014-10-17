import numpy as np



mu = 100
sigma = 0.2
N = 100000
s = np.random.normal(mu, sigma, N)
hist, bin_edges = np.histogram(s, bins=1000, density=True)
halfmax = 0.5 * max(hist)





print "hist"
print hist
print "halfmax = %10.4f" % halfmax
binwidths = np.diff(bin_edges)

idx = (hist>=halfmax)
print "indices ", idx 

pdf=1/(sigma∗np.sqrt(2∗np.pi))∗np.exp(−(bin_edges−mu)∗∗2/(2∗sigma∗∗2))
halfmax2 = max(pdf) /2.0
idx = (hist>=halfmax)

print "binwidth indices ", binwidths[idx] 
print "hist indices ", hist[idx] 
gamma = sum(binwidths[idx] * hist[idx])

print "Gamma = %10.4f" % gamma
print "2.354 * sigma =  %10.4f" % (2.354 * sigma)


