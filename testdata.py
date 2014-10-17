import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

data = np.loadtxt("datafit1.dat")
x = data[:,0]
y = data[:,1]
yerr = data[:,2]


exp = True #a0*exp-(a1*x + a2*x**2 + ... am * x**m)  else a0 + a1 * x + ... am * x**m    in function poly1d the order is reversed

def calculateDelta(m):
	if(exp):
		sigma = np.divide(yerr,y)
	else: 
		sigma = yerr
	a = np.zeros([m+1, m+1])
	for i in range(0,m+1):
		for j in range(0,m+1):
			a[i][j] = np.sum(np.divide(x**(i+j), sigma**2 ))
	print "matriz "
	print a	
	return np.linalg.det(a)


def calculateDAi(m,k):
	if(exp):
		sigma = np.divide(yerr,y)
	else: 
		sigma = yerr
	
	a = np.zeros([m+1, m+1])
	for i in range(0,m+1):
		for j in range(0,m+1):
			if(j!=k):	
				a[i][j] = np.sum(np.divide(x**(i+j), sigma ** 2 ))
			else:
				a[i][j] =np.sum(np.divide(y * (x ** i), sigma ** 2))
	print "matriz A%d"%k
	print a	
	return np.linalg.det(a)




maxdeg = 1
for m in range(1, maxdeg+1):
	a = np.zeros(m+1)
	det = calculateDelta(m)
	for i in range(0,m+1):	
		print "m=%d, i=%d" % (m,i)
		a[m-i] =  calculateDAi(m,i) / det
	 	coefErr = sqrt(np.sum( np.divide( ((x ** i) * (x ** i)), yerr ** 2  )) / det)
		if(exp):
			coefErr /=a[m-i]
		print "error coef for a[%d] = %4.4f " % ((m-i), coefErr)	

	
	print "coef a"
	print a

	cp = np.polyfit(x,y,m)
	print "coef poly= "
	print cp
	
	plt.cla()
	plt.errorbar(x, y, yerr=yerr, fmt='o', ms=1)
	if(exp):
		sumty = np.zeros(len(x))
		for i in range(0,m):
			sumty += (x ** (m-i)) * a[i]
		print "sumty="
		print sumty
		ty = a[m] * np.exp(-sumty) 
	else:
		p = np.poly1d(a)
		ty = p(x)
	print "ty = "
	print ty
	plt.plot(x,ty,"g-")
	plt.draw()
	plt.show()
	#var = raw_input("press key to continue")










