from math import cos,  pi, e, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def f(t):
	return cos(6*pi*t) * e**(-pi * t**2)

T = 2

x = np.arange(-T, T, 0.1)
#TODO WHY?????
yval = []
for xval in x:
	yval.append(f(xval))
y = np.array(yval)
#y = f(x)

plt.plot(x, y)



def anFunc(n):
	func = lambda t: f(t) * cos(2.0 * n * pi * t / T)  	
	return (2.0 / T) * integrate.quad(func, -0.5 * T, 0.5 * T)[0] 

def bnFunc(n):
	func = lambda t: f(t) * sin(2.0 * n * pi * t / T)  
	resIntegr = 	integrate.quad(func, -0.5 * T, 0.5 * T)[0]	
	print("Integrating result %4.3f" % resIntegr)
	return (2.0 / T) * resIntegr 



N = 50
an = []
bn = [] 
for n in range(0,N):
	anVal = anFunc(n)
	bnVal = bnFunc(n)
	print ("a%d=%4.3f, b%d=%4.3f" % (n,anVal,n,bnVal))
	an.append(anVal)
	bn.append(bnVal)

def g(t):
	res = 0.5 * an[0]
	for n in range(1,N):
		res+=an[n] * cos(2 * n * pi * t / T) + bn[n] * sin(2 * n * pi * t / T)
	return res


yval = []
for xval in x:
	yval.append(g(xval))
yf = np.array(yval)

plt.plot(x, yf)

plt.draw()
plt.show()



