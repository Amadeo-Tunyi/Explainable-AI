
import numpy as np
import pandas as pd
from helpers import CustomScaler


from sklearn.neighbors import KernelDensity

from scipy.special import gamma as gamma_function


EPSILON = 1e-8

### KDE kernel
class Kernel_obj:
	def __init__(self, data, distrib = None, bandwidth=0.5, Num_points=1, knnK=1):
		self.b = bandwidth
		self._distribution = distrib
		self._kernel = KernelDensity(bandwidth=self.b , kernel='tophat')
		self.Num_points = Num_points
		self.knnK = knnK
		self.volume = None
		self.data = data



		

	def kernelUniform(self, xi, xj):
		"""
		KDE kernel value
		"""
		mean = 0.5*(xi + xj)
		dist = np.linalg.norm(xi - xj, 2)
		density_at_mean = self._distribution.pdf(mean)
		return density_at_mean*dist
	def K(self, x):
		return np.exp(-x**2/2)/np.sqrt(2*np.pi)
	def KDE(self, xi):
		norm = distance_obj()
		inner = np.array([self.K(norm.computeDistance(self.data[i],  xi)/self.b) for i in range(self.data.shape[0])])
		
		return inner.sum()/(len(self.data)*self.b)
	
	def scale(self, data, xi):
		scaler  = CustomScaler()
		scaler.fit(data)
		return scaler.transform(xi)


	def kernelKNN(self, xi):
		"""
		kNN kernel
		"""
		norm = distance_obj()
		distances = np.array([norm.computeDistance(xi, self.data[i]) for i in range(self.data.shape[0]) if np.array_equal(xi, self.data[i]) == False])
		sorted_distances = np.sort(distances) 
		dim = len(xi)
		if self.volume is None:
			self.volume = np.pi**(dim//2) / gamma_function(dim//2 + 1)
		dist = sorted_distances[:self.Num_points].max()
		density_at_mean = self.knnK/((self.Num_points*self.volume)*dist)
		return density_at_mean
	

	def knn_density(self,xi):
		lst = []
		for i in range(len(self.data)):
			lst.append((self.kernelKNN(np.array(self.data)[i])))
		x = self.kernelKNN(xi)
		return self.scale(np.array(lst), x)
	




class distance_obj:
	def __init__(self):
		return

	def computeDistance(self, xi, xj):
		dist = np.linalg.norm(xi - xj, 2)
		return dist
	

