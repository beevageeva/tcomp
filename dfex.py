import os
import numpy as np
import matplotlib.pyplot as plt


#http://www.chem.uoa.gr/applets/AppletPoly/Appl_Poly2.html

outdir = "outn"
if not os.path.exists(outdir):
	os.mkdir(outdir)
data = np.loadtxt("fit1.dat")
x = data[0,:]
y = data[1,:]
n = len(x)
print "len(x) = %d, len(y) = %d" %(len(x), len(y))

def calculateSk(k):
	res = 0
	for i in range(0,n):
		res+=x[i] ** k
	return res 

def calculateTk(k):
	res = 0
	for i in range(0,n):
		res+=(x[i] ** k) * y[i]
	return res 


def calcCoef(Dm):
	#gaussian 
	#http://en.wikipedia.org/wiki/Gaussian_elimination
	print "********calcCOef%d" % Dm
	flag = False
	ad1 = np.zeros([Dm+1, Dm+2])
	ad = np.zeros(2*Dm+1)
	ad2 = np.zeros(Dm+2)
	A = np.zeros(Dm+1)
	print "ad"	
	print ad
	#obviously this is n
	ad[0] = n
	for l in range(1,n):
		for i2 in range(1, 2*Dm + 1):
			ad[i2] += x[l] ** i2;
		for i3 in range(0, Dm + 1):
			d = x[l] ** i3
			ad1[i3][Dm + 1] = ad2[i3] + y[l] * d;
			ad2[i3] += y[l] * d
		ad2[Dm + 1] += y[l] * y[l];
	for j2 in range(0,Dm + 1):		
		for j3 in range(0,Dm + 1):	
			ad1[j2][j3] = ad[(j2 + j3) - 1]
	print "ad1 NOW "
	print ad1
	for k2 in range(0,Dm+1):		
		for k3 in range(k2,Dm + 1):	
			print"absin continue test %10.4f" % abs(ad1[k3][k2])	
			if(abs(ad1[k3][k2])<=0.0001):
				continue
			flag = True
			break
		print "afetr break " + str(flag)
		if(not flag):
			return A
		flag = False
		for i1 in range(0,Dm+2):
			d1 = ad1[k2][i1];
			ad1[k2][i1] = ad1[Dm][i1];
			ad1[Dm][i1] = d1;
		d2 = 1.0 / ad1[k2][k2];
		for j1 in range(0,Dm+2):
			ad1[k2][j1] *= d2;
		for l3 in range(0,Dm+1):
			if(l3!=k2):
				d3= - ad1[l3][k2]
				for k1 in range(0, Dm+2):
					ad1[l3][k1] +=d3 * ad1[k2][k1]
	print "*********ad1************"
	print ad1
	print "*********END ad1************"
	for l2 in range(0, Dm+1):
		A[Dm-l2] = ad1[l2][Dm+1]		
	return A


#THIS is maximum degree 
maxdegree = 2
for m in range(1,maxdegree+1):
	plt.cla()
#	s = np.empty([m+1,m+1])
#	t = np.empty([m+1,1])
#	for i in range(0,m+1):
#		s[m,i] = 0
#		for j in range(0,m+1):
#			if(i>j):
#				#its a symmetric matrix  calculate only once
#				s[i,j] = s[j,i]
#			else:
#				if(i>1 and j<m):
#					s[i,j] = s[i-1, j+1]
#				else:
#					s[i,j] = calculateSk(i+j)
#		t[m] = calculateTk(i)
#	print "s="
#	print s
#	print "t="
#	print t
	#c = np.linalg.solve(s, t)
	#print "coef = "
	#print c
	#print "verify s * c"
	#print np.dot(s, c)
	plt.plot(x,y,"ro" , ms=1)
	#p = np.poly1d(c[0])
	coefjava = calcCoef(m)
	p = np.poly1d(coefjava)
	ty = p(x)
	plt.plot(x,ty,"g-")
	cp = np.polyfit(x,y,m)
	print "coef poly= "
	print cp
	print "coef java= "
	print coefjava


	plt.savefig(os.path.join(outdir,"%d.png" % m))


