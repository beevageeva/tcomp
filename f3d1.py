from math import cos,  pi, e, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.fftpack import fft,fftfreq#forFourierTransform

f = 400
ns = 16
numPoints = 2**6  #power of 2!!
#from def above
T = 1.0 / f
dt = T / ns
t = np.linspace(0, (numPoints - 1) * dt, numPoints )


def fun(t):
	return np.cos(2 * pi * f * t) * np.exp(pi * t**2)

def fun2(t):
	return np.sin(2 * np.pi * f * t ) - 0.5*np.sin( 2 * np.pi *2* f * t )




def getCoefExpl():

	def anFunc(n):
		func = lambda t: fun(t) * cos(2.0 * n * pi * t / T)  	
		return (2.0 / T) * integrate.quad(func, -0.5 * T, 0.5 * T)[0] 
	
	def bnFunc(n):
		func = lambda t: fun(t) * sin(2.0 * n * pi * t / T)  
		resIntegr = 	integrate.quad(func, -0.5 * T, 0.5 * T)[0]	
		#print("Integrating result %4.3f" % resIntegr)
		return (2.0 / T) * resIntegr 

	an = []
	bn = []
	 
	#for n in range(0,N):
	#for n in range(0, numPoints + 1):
	for n in range(0, numPoints):
		anVal = anFunc(n)
		bnVal = bnFunc(n)
		#print ("a%d=%4.3f, b%d=%4.3f" % (n,anVal,n,bnVal))
		an.append(anVal)
		bn.append(bnVal)

	return an,bn



def fourierFunExpl(t):

		an,bn = getCoefExpl()

		print("size of an")
		print(len(an))
		print("size of bn")
		print(len(bn))
		

		print("an calc")
		print(an)
		print("bn calc")
		print(bn)

		#res = 0.5 * an[0]
		res = 0.0
		#for n in range(1,numPoints + 1):
		#for n in range(1,numPoints):
		for n in range(0,numPoints):
			res+=an[n] * np.cos(2 * n * pi * t / T) + bn[n] * np.sin(2 * n * pi * t / T)
		return res

def fourierFunFFT(t):
		#with scipy
		rfft = 	fft(fun(t)) 
		#normalize ??
		#rfft = 	fft(fun(t)) / numPoints
		an = rfft.real
		bn = rfft.imag
		print("shape an")
		print(an.shape)
		print("shape bn")
		print(bn.shape)
		print("an fft")
		print(an)
		print("bn fft")
		print(bn)
		ff = fftfreq(numPoints)		
		print("frequencies shape")
		print(ff.shape)
		print("frequencie")
		print(ff)
		#res = 0.0
		res = np.zeros(numPoints)
		#res += an[0]
		#for n in range(1,numPoints + 1):
		#for n in range(1,numPoints):
		middle = int(numPoints/2)
		for n in range(0, numPoints):
			val = 0
			for k in range(1, middle):
				val +=(an[k]  + an[numPoints - k])* np.cos(2 * n * pi * k * dt /numPoints)
			res[n] = val
		print("res = ")
		print(res) 
		return res



#to use fun2:
fun = fun2

plt.plot(t, fun(t), "r")
#plt.plot(t, fourierFunExpl(t), "b")
plt.plot(t, fourierFunFFT(t), "b")

plt.draw()
plt.show()


