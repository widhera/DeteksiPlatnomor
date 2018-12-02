from skimage import feature
import numpy as np

class LBP:
	"""docstring for LBP"""
	def __init__(self, numPoints, radius):
		self.numPoints = numPoints
		self.radius = radius
	def describe(self,image,eps=1e-7):
		#histogram
		lbp = feature.local_binary_pattern(image,self.numPoints, self.radius,method="uniform")
		(hist,_) = np.histogram(lbp.ravel(), bins=np.arange(0,self.numPoints+3), range=(0, self.numPoints+2))
		#normalize
		hist = hist.astype("float")
		hist /= (hist.sum()+eps)
		#retrun histogram lbp
		return hist
		
		