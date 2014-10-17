import os
import numpy as np
import matplotlib.pyplot as plt


#http://www.chem.uoa.gr/applets/AppletPoly/Appl_Poly2.html

fit = False

outdir = "outn"
if not os.path.exists(outdir):
	os.mkdir(outdir)
data = np.loadtxt("fit1.dat")
if(fit):
	x = data[0,:]
	y = data[1,:]
else:
	x = data[:,0]	
	x = data[:,1]	
n = len(x)
print "len(x) = %d, len(y) = %d" %(len(x), len(y))


def calcCoef(Dm):
	print "********calcCOef%d" % Dm
	flag = False
	flag1 = False
	ad1 = np.zeros([Dm+2, Dm+3])
	ad = np.zeros(2*Dm+2)
	ad2 = np.zeros(Dm+3)
	A = np.zeros(Dm+1)
	print "ad"	
	print ad
	ad[1] = n
	for l in range(1,n):
		for i2 in range(2, 2*Dm + 2):
			ad[i2] += x[l] **( i2 - 1);
		for i3 in range(1, Dm + 2):
			d = x[l] ** (i3-1)
			ad1[i3][Dm + 2] = ad2[i3] + y[l] * d;
			ad2[i3] += y[l] * d
		ad2[Dm + 2] += y[l] * y[l];
	for j2 in range(1,Dm + 2):		
		for j3 in range(1,Dm + 2):	
			ad1[j2][j3] = ad[(j2 + j3) - 1]
	print "ad1 NOW "
	print ad1
	for k2 in range(1,Dm+2):		
		for k3 in range(k2,Dm + 2):	
			print"absin continue test %10.4f" % abs(ad1[k3][k2])	
			if(abs(ad1[k3][k2])<=0.0001):
				continue
			flag = True
			break
		print "afetr break " + str(flag)
		if(not flag):
			return A
		flag = False
		k3=Dm + 1
		for i1 in range(1,Dm+3):
			d1 = ad1[k2][i1];
			ad1[k2][i1] = ad1[k3][i1];
			ad1[k3][i1] = d1;
		d2 = 1.0 / ad1[k2][k2];
		for j1 in range(1,Dm+3):
			ad1[k2][j1] *= d2;
		for l3 in range(1,Dm+2):
			if(l3==k2):
				flag = True				
			if(not flag):
				d3= - ad1[l3][k2]
				for k1 in range(1, Dm+3):
					ad1[l3][k1] +=d3 * ad1[k2][k1]
			flag = False
	print "*********ad1************"
	print ad1
	print "*********END ad1************"
	for l2 in range(0, Dm+1):
		A[Dm-l2] = ad1[l2+1][Dm+2]		
	return A


#THIS is maximum degree 
maxdegree = 2
for m in range(1,maxdegree+1):
	plt.cla()
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


