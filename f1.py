from math import cos,  pi, e, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sys

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




def anFunc(n):
	func = lambda t: f(t) * cos(2.0 * n * pi * t / T)  	
	return (2.0 / T) * integrate.quad(func, -0.5 * T, 0.5 * T)[0] 

def bnFunc(n):
	func = lambda t: f(t) * sin(2.0 * n * pi * t / T)  
	resIntegr = 	integrate.quad(func, -0.5 * T, 0.5 * T)[0]	
	#print("Integrating result %4.3f" % resIntegr)
	return (2.0 / T) * resIntegr 




def calculate(N):
	print("N is %d" % N)
	an = []
	bn = [] 
	for n in range(0,N+1):
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
	plt.cla()
	plt.plot(x, y)
	plt.plot(x, yf)
	plt.draw()
	plt.show()

N = 1
calculate(N)

stop = False
while not stop:
	vstr = input("q for quit, p inc N, l dec N")
	print("Valor inroducido: %s" % vstr)
	if(vstr == "q"):
		sys.exit(0)
	if(vstr == "p"):
		N+=1
	elif (vstr == "l"):
		N-=1		
	if(vstr == "p" or vstr == "l"):
		calculate(N)
		





