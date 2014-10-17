from math import cos,  pi, e, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.fftpack import fft,fftfreq#forFourierTransform



def fun2( t , f) :
	return np.sin(2 * np.pi * f * t ) - 0.5*np.sin( 2 * np.pi *2* f * t )



def inpvar(n2,f,ns):
	pn=np.arange(1,2**n2+1)
	T=1.0/f
	dt=1.0/(f*ns)
	return pn,T,dt

n2=6#itwillgive2**n2points
f=400#FrequencyinHz(f=1/T)
ns=16
pn,T,dt=inpvar(n2,f,ns)
ln=len(pn)
x=np.linspace(0,(ln-1)*dt,ln)
print("T=%4.3f,ln=%d, dt=%4.3f, xstart: " % (T,ln, dt))
print(x)
print("xend")


def plotWindowFunc():
	plt.clf()
	plt.plot(np.arange(20), np.kaiser(20,3.5))
	plt.plot(np.arange(20), np.bartlett(20))
	plt.plot(np.arange(20), np.blackman(20))
	plt.plot(np.arange(20), np.hamming(20))
	plt.plot(np.arange(20), np.hanning(20))



def anFunc(n):
	func = lambda t: fun2(t,f) * cos(2.0 * n * pi * t / T)  	
	return (2.0 / T) * integrate.quad(func, -0.5 * T, 0.5 * T)[0] 

def bnFunc(n):
	func = lambda t: fun2(t,f) * sin(2.0 * n * pi * t / T)  
	resIntegr = 	integrate.quad(func, -0.5 * T, 0.5 * T)[0]	
	#print("Integrating result %4.3f" % resIntegr)
	return (2.0 / T) * resIntegr 


def calculate(N,x):
	an = []
	bn = [] 
	for n in range(0,N):
		anVal = anFunc(n)
		bnVal = bnFunc(n)
		#print ("a%d=%4.3f, b%d=%4.3f" % (n,anVal,n,bnVal))
		an.append(anVal)
		bn.append(bnVal)
	
	def g(t):
		res = 0.5 * an[0]
		for n in range(1,N):
			res+=an[n] * np.cos(2 * n * pi * t / T) + bn[n] * np.sin(2 * n * pi * t / T)
		return res
	return g(x)



#plt.clf()

Y=fft(fun2(x,f))/ln
F=fftfreq(ln,dt)
#plt.vlines(F,0,Y.imag)






print("Y")
print(Y)
print("Yend")
print("F")
print(F)
print("Fend")




#plt.annotate(s=u'f=400Hz',xy=(400.0,-0.5),xytext=(400.0+1000.0,-0.5-0.35),arrowprops=dict(arrowstyle="->"))
#plt.annotate(s=u'f=-400Hz',xy=(-400.0,0.5),xytext=(-400.0-2000.0,0.5+0.15),arrowprops=dict(arrowstyle="->"))
#plt.annotate(s=u'f=800Hz',xy=(800.0,0.25),xytext=(800.0+600.0,0.25+0.35),arrowprops=dict(arrowstyle="->"))
#plt.annotate(s=u'f=-800Hz',xy=(-800.0,-0.25),xytext=(-800.0-1000.0,-0.25-0.35),arrowprops=dict(arrowstyle="->"))
#plt.ylim(-1,1)
#plt.xlabel('Frequency(Hz)')
#plt.ylabel('Im($Y$)')


n2=2**5
x2=np.linspace(0,0.012,n2)#Timeinterval
dt2=x2[1]-x2[0]
y2=	fun2(x2,f)
Y2=fft(y2)/n2#FastFourierTransformnormalized
F2=fftfreq(n2,dt2)#Frequencies
#plt.vlines(F2,0,Y2.imag)

t3=np.linspace(0,0.012+9*dt2,10*n2)#Timeinteval
y3=np.append(y2,np.zeros(9*n2))#Signal
Y3=fft(y3)/(10*n2)
F3=fftfreq(10*n2,dt2)#Frequencies
#plt.vlines(F3,0,Y3.imag)

#plotWindowFunc()

n4=2**8
t4=np.linspace(0,0.05,n4)
dt4=t4[1]-t4[0]
y4=fun2(t4,f)
y5=y4*np.blackman(n4)
t4=np.linspace(0,0.12+4*dt4,5*n4)
y4=np.append(y4,np.zeros(4*n4))
y5=np.append(y5,np.zeros(4*n4))
Y4=fft(y4)/(5*n4)
Y5=fft(y5)/(5*n4)
F4=fftfreq(5*n4,dt4)
#plt.plot(t4,y4)

#plt.vlines(F4,0,Y4.imag)
#plt.plot(t4,y5)

plt.vlines(F4,0,Y5.imag)






#plt.plot(x2, fun2(x2,f), "ro")
#plt.plot(x2, calculate(50,x2), "g-")


#plt.plot(x, fun2(x,f), "ro")
#plt.plot(x, calculate(50,x), "g-")


plt.draw()
plt.show()



