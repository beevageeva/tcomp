import numpy as np
from scipy.fftpack import fft,fftfreq#forFourierTransform
import matplotlib.pyplot as plt




def findFreq(x,y):
	print("x")
	print(x)	
	print("y")
	print(y)

	
	
	#plt.plot(x,y, "ro", ms=1)
	minX = np.min(x)
	print("minX = %4.3f" % minX	)
	maxX = np.max(x)
	print("maxX = %4.3f" % maxX	)
	numPoints = 2 ** 8
	dx = (maxX - minX) / float(numPoints)
	newX = np.arange(minX, maxX, dx)
	newY = np.interp(newX, x, y)

	newYWindow=newY*np.blackman(numPoints)
	
	print("lenNewX=%d" % len(newX))
	print("lenNewY=%d" % len(newY))


	#Y=fft(newY)/(numPoints)
	Y=fft(newYWindow)/(numPoints)
	F=fftfreq(numPoints,dx)
	
	print("F=")
	print(F)
	print("Y=")
	print(Y.imag)

	plt.vlines(F,0,Y.imag)

	plt.draw()
	plt.show()


#data = np.loadtxt("composite_d25_07_0310a.dat")
#data = data[data[:,2]>0]
#if(np.all(np.diff(data[:,1]) > 0)):
#	print("array x ordered")
#else:
#	print("array x NOT ordered")
#	np.sort(data, order=[1])
#findFreq(data[:,1], data[:,2])

#for filename in ["sunspots_day.dat", "sunspots_year.dat"]:
for filename in ["sunspots_year.dat"]:
	data = np.loadtxt("sunspots_day.dat")
	if(np.all(np.diff(data[:,0]) > 0)):
		print("array x ordered")
	else:
		print("array x NOT ordered")
		np.sort(data, order=[0])
	findFreq(data[:,0], data[:,1])
