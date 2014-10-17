import sys,getopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider





class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def set_val(self, val):
        discrete_val = int(val)
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.items():
            func(discrete_val)




def readFile(filename, cols, useSigma):
	data = np.loadtxt(filename)
	if(len(data.shape)!=2):
		print("data must be bidim. data.shape = %s" % str(data.shape))
		
	sigma = None
	#test data read from file (shape) 
	if(cols):
		if(data.shape[1]<2):
			print("data must have at least 2 columns, data.shape = %s" % str(data.shape))
			sys.exit(0)
		elif(useSigma and data.shape[1]<3):
			print("if useSigma data must have at least 3 columns, data.shape = %s" % str(data.shape))
			sys.exit(0)
		#sort by first column (x)
		data = data[data[:,0].argsort()]
		x = data[:,0]
		y = data[:,1]
		if(useSigma):
			sigma = data[:,2]
	else:	
		#test data read from file (shape) 
		if(data.shape[0]<2):
			print("data must have at least 2 rows, data.shape = %s" % str(data.shape))
			sys.exit(0)
		elif(useSigma and data.shape[0]<3):
			print("if useSigma data must have at least 3 rows, data.shape = %s" % str(data.shape))
			sys.exit(0)
		#sort by first line (x)
		data = data[:,data[0].argsort()]
		x = data[0]
		y = data[1]
		if(useSigma):
			sigma = data[2]
	n = len(x)
	#test len(x) = len(y)
	if(len(y)!=n):
		print("x and y must have same length len(x) = %d, len(y) = %d" % (n , len(y)))
		sys.exit(0)
	#test len(sigma) = len(x) if sigma is defined
	if(not sigma is None and len(sigma)!=n):
		print("if sigma defined it has to have the same length as x: len(x) = %d, len(sigma) = %d" % (n , len(sigma)))
		sys.exit(0)
	#print "x="
	#print x
	#print "y="
	#print y
	#print "sigma="
	#print sigma
	fig, ax = plt.subplots(1)
	plt.subplots_adjust(left=0.15, bottom=0.4)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.grid(True)
	ax.autoscale(True)
	ax.plot(x,y,"ro", ms=1, lw=0)
	l, = ax.plot(x,y, "g-")

	def drawPlot(m):
		print("Poly degree is %d" % m)
		A = getAMatrix(x,m,sigma)
		B = getBColumn(x,y,m,sigma)
		#print("A matrix is")
		#print(A)
		#print("B column is")
		#print(B)
		c,v = solveKramer(A, B)
		polyCurve = np.poly1d(c)
		ty = polyCurve(x) #teoretic y
		print("calculated coeficients are:")
		print(c)
		print("coef variance:")
		print(v)
		#polyfit throws this error sometimes: when rank(A)< m(equiv A cannot be inversed and there is no unique solution)?
		try:
			polyC, polyV = np.polyfit(x,y,m, full=False, cov=True)
			print("coef from polyfit are:")
			print(polyC)
			print("covariance matrix from polyfit are:")
			print(polyV)
		except ValueError as err:
			print("Error in numpy.polyfit " )
			print(err)
		print("-------------------------------------------------------------------")
		print("goodness=R**2=%4.3f"% (1  - np.var(np.subtract(ty,y)) / np.var(y)) )
		#corrected division by 0 error when m==n
		print("avg unc=sigma**2=%4.3f"% ((1.0 / ((n-m) if n!=m  else 1) ) *  np.sum(np.power(np.subtract(ty,y),2))) )
		l.set_ydata(ty)
		ax.relim()
		ax.autoscale_view(True,True,True)
		plt.draw()
	varHash = {'m':1} # I have to keep track of the current slider value in order not to update if value is not changing
	#I want a discrete slider so several values of the slider will be converted to the same integer value
	#I have to use the hash because m is changed in the listener function and m variable would be local to this function
	#I don't use a global variable m. In python 3 there is the nonlocal statement which causes the listed identifiers to refer to previously bound variables in the nearest enclosing scope.
	drawPlot(1)	
	axSlider = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg='white')
	#mSlider =  Slider(axSlider, 'Degree', 1, 23, valinit=1, valfmt='%d')#max degree 23
	#the difference in using Slider or DiscreteSlider is only in the visual update of the slider bar
	#if discrete it will update only with portions corresponding to 1, I still have to keep track of the value
	#because the actual value of slider.val will always be the float
	mSlider =  DiscreteSlider(axSlider, 'Degree', 1, 23, valinit=1, valfmt='%d')#max degree 23
	

	def sliderChanged(val):
		print("SLIDER CHANGED EVENT " + str(val))
		intVal = int(val)
		#I don't want an update if values are equal (as integers)
		if(intVal!=varHash["m"]):
			print("Integer Value of slider CHANGED set slider to %d" % intVal)
			varHash["m"] = intVal
			drawPlot(intVal)
		else:
			print("BUT integer VALUE did NOT change")

	mSlider.on_changed(sliderChanged) 	
	plt.show()
	



#sigma array standard deviation of y(from measurement) , if None not taken into account(assume all sigma_i = 1)
#m degree
def getAMatrix(x,m,sigma=None):
	if(len(x.shape)!=1):
		print("x must be a unidimesional array and has %d dim"%len(x.shape))
		sys.exit(0)
	n = x.shape[0]
	if((not sigma is None) and (len(sigma.shape)!=1 or sigma.shape[0]!=n) ):
		print("sigma defined and must be a unidimesional array(has %d dim) and have same elements as x"%len(sigma.shape))
		sys.exit(0)
	A = np.zeros(shape=(m+1,m+1))
	for k in range(0, m+1):	
		for l in range(0, m+1):	
			if(k>0 and l<m):
				A[k][l] = A[k-1][l+1]
			else:
				if(sigma is None):
					A[k][l] = np.sum(np.power(x, (k+l)))
				else:
					A[k][l] = np.dot(np.power(sigma,-2.0), np.power(x, (k+l)))	
		
	return A


#sigma array standard deviation of y(from measurement) , if None not taken into account(assume all sigma_i = 1)
#m degree
def getBColumn(x, y, m, sigma=None):			
	B = np.zeros(m+1)
	for k in range(0, m+1):
		if(sigma is None):
			t = np.power(x, k)
		else:
			t = np.multiply(np.power(sigma,-2.0), np.power(x, k))
		B[k] = np.dot(t,y)
	return B	


#returns an array of 2 elements: the first is the coefficients array and the second is the coefficient variances array
#both in order from higher to lower degree 
def solveKramer(A, B):
	#Kramer method even if we have the unique solution if and only if detA!=0 (equiv A has an inverse and I could get the coef as A ** (-1) * B)
	#I have to calculate A ** (-1) because is the covariance matrix
	#I already know that A is a square matrix of dim m y B a vector of dim m, but check
	if((len(A.shape)!=2) or (len(B.shape)!=1) or(A.shape[0]!=B.shape[0]) or(A.shape[1]!=B.shape[0]) ):
		print("dimesions of arrays incorect A.shape = %s, B.shape = %s" % (str(A.shape), str(B.shape)))
		sys.exit(0)
	m = B.shape[0]
	c = np.zeros(m)
	detA = np.linalg.det(A)
	if(detA == 0):
		"detA = 0, matrix A is not inversible, no coef..exit"
		sys.exit(0)	
	for i in range(0,m):
		Ai = np.array(A)
		Ai[:,i] = B
		#print("Kramer: A%d is" % i)
		#print(Ai)
		detAi =  np.linalg.det(Ai)
		c[i] =  detAi / detA
	invA = np.linalg.inv(A)
	print("coef 0..m  as A ** (-1) * B ")
	print(np.dot(invA, B))
	print("coef 0..m  from kramer ")
	print(c)
	#reverse order to be used directly in numpy.poly1d like the coefficients returned by numpy.polyfit
	return [c[::-1], invA.diagonal()[::-1]] 


def usage():
	print("Usage: %s --input=<filename> [--rows] [--sigma]\n If option --rows not present it will assume that data is by columns like in datafit1.dat otherwise by rows like in fit1.dat\n If option --sigma present it will attempt to read the sigma column or row and use it" % sys.argv[0])



def main():
	try:
		opts, args = getopt.getopt(sys.argv[1:], "", ["help", "input=", "rows", "sigma"])
	except getopt.GetoptError as err:
		# print help information and exit:
		print(str(err)) # will print something like "option -a not recognized"
		usage()
		sys.exit(2)
	cols = True #if False like in fit1.dat x and y defined in a row, if True like in datafit1 where x and y are defined in a column
	useSigma = False #if True will try to read sigma : from a row or a column depending on value of varable cols
	inputFilename = None
	for o, a in opts:
		if o in("--rows"):
			cols = False
		elif o in("--sigma"):
			useSigma = True
		elif o in ("--help"):
			usage()
			sys.exit()
		elif o in ("--input"):
			inputFilename = a
		else:
			print("option %s not recognized " % o)
	if(inputFilename is None):
		usage()
		sys.exit()
	print("InputFile=%s, cols=%s, useSigma=%s" % (inputFilename, str(cols), str(useSigma)))	
	readFile(inputFilename, cols, useSigma)

if __name__ == "__main__":
    main()


