import numpy as np


for i in range(1,5):
	fn = "fit%d.dat" % i
	data = np.loadtxt(fn)
	x = data[0, :] 
	y = data[1, :] 
	n = len(x)
	
	dx = x - np.mean(x)
	dy = y - np.mean(y)
	covxy = 1.0/n * np.sum(dx*dy)
	corrxy = covxy/(np.std(x) * 	np.std(y))
	
	
	print "corrxy=%10.4f" % corrxy


