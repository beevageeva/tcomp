import os
import numpy as np
import matplotlib.pyplot as plt



outdir = "out"
if not os.path.exists(outdir):
	os.mkdir(outdir)
data = np.loadtxt("fit1.dat")
x = data[0,:]
y = data[1,:]
n = len(x)

#print "x="
#print x
#print "y ="
#print y


#plt.plot(x,y,"ro",ms=1 )
#plt.draw()
#plt.show()


#THIS is maximum degree
maxdegree = 40
for m in range(1,maxdegree):
	c = np.polyfit(x,y,m)
	plt.cla()
	plt.plot(x,y,"ro",ms=1 )
	print "m=%d, coef" % m
	print c
	p = np.poly1d(c)
	print "x0%10.4f" % x[0]
	print "y0%10.4f" % y[0]
	print "calcy0%10.4f" % p(x[0])
	ty = p(x)
	#print "ty ="
	#print ty
	plt.plot(x,ty,"g-")
	#plt.draw()
	#plt.show()
	plt.savefig(os.path.join(outdir,"%d.png" % m))


