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
	covxx= 1.0/n * np.sum(dx*dx)
	
	
	#print "x="
	#print x
	#print "y="
	#print y
	print "covxx=%10.4f" % covxx
	print "varx=%10.4f" % np.var(x)


