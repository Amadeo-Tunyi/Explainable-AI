import scipy
from scipy.stats import multivariate_normal, multivariate_t, multivariate_hypergeom, multinomial
import numpy as np
from collections import Counter



class distribution():

    def __init__(self, Ansatz = 'Gaussian'):
        self.Ansatz = Ansatz




    def fit_data(self, data):
        self.data = data


    def get_distribution_pdf(self, x):
        if self.Ansatz == 'Gaussian':
                # Calculate the mean and covariance matrix of the data
            mean = np.mean(self.data, axis=0)
            cov_matrix = np.cov(self.data, rowvar=False)
            print(cov_matrix.shape)

            # Fit a multivariate normal distribution to the data
            distribution = multivariate_normal(mean=mean, cov=cov_matrix)


            return distribution.pdf(x)
        


                

        




