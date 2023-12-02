import networkx as nx
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import gamma as gamma_function
import scipy.stats as st
from sklearn.mixture import GaussianMixture
import pandas as pd
class FACE:
    def __init__(self, epsilon, density_threshold,  classifier, number_of_paths, classification_threshold = 0.5,  density_estimator = None, 
                 kde_bandwidth = None, knn_number_of_points = None, knnK = None, knn_volume = None):
        self.eps = epsilon
        self.t_p = density_threshold
        self.t_d = classification_threshold
        self.p = density_estimator
        self.bandwidth = kde_bandwidth
        self.Num_points = knn_number_of_points
        self.volume = knn_volume
        self.K = knnK
        self.clf = classifier
        self.n = number_of_paths
        self.G = nx.DiGraph()
        self.stored_indices = []
        self.visited = []

    def find_closest(self, unit, data):
        
        index = self.get_index(unit)
        self.stored_indices.append(index)
        wanted_indices = [i for i in range(len(data)) if i not in self.stored_indices]
        distances = [self.distance(unit, data[wanted_indices][i]) for i in range(len(data[wanted_indices]))]
        wanted_index = np.argsort(np.array(distances))[1]
        return data[wanted_indices][wanted_index]
    
    def find_neighbours(self, unit, data):
        self.visited.append(self.get_index(unit))
        wanted_indices = [i for i in range(len(data)) if i not in self.visited]
        distances = [self.distance(unit, data[wanted_indices][i]) for i in range(len(data[wanted_indices]))]
        wanted_index = np.argsort(np.array(distances))[1:self.n+1]
        return data[wanted_indices][wanted_index]
    def distance(self, x_1, x_2):
        return np.linalg.norm(x_1 - x_2, 2)
    
    def normalize(self, data):
    
   
        
    
    # Normalizing all the n features of X.
        
        data = (data - data.mean(axis=0))/data.std(axis=0)
        
        return data
    



    
    def _kernel_function(self, data, function, xi, xj):
        

        if function == 'KDE':
            mean = 0.5*(xi + xj)
            dist = np.linalg.norm(xi - xj, 2)
            kde = KernelDensity(kernel='gaussian', bandwidth= self.bandwidth).fit(data)
            density_at_mean = np.exp(kde.score_samples([mean]))
            return (1/(density_at_mean + self.eps))*dist
        elif function == 'kNN':
            dim = len(xi)
            if self.volume is None:
                self.volume = np.pi**(dim//2) / gamma_function(dim//2 + 1)
            dist = np.linalg.norm(xi - xj, 2)
            density_at_mean = self.K/(self.Num_points*self.volume)*dist
            return density_at_mean
        # elif function == 'Gaussian':
        #     mean = 0.5*(xi + xj)
        #     dist = np.linalg.norm(xi - xj, 2)
        #     density_at_mean = distribution.pdf(mean)
        #     return density_at_mean*dist



    def get_index(self, target_array):
        for i, row in enumerate(self.train_data):
            if all(target_array == row):
                return i  # Return the index of the row if the array is found
        return -1 





    def fit(self, train_data, label_data):
        self.train_data = train_data
        self.training_labels = label_data
    def create_path_to_counterfactual_class(self, x_i, target_label):
        self.t = target_label
       
        i = 0
        self.G.add_node(self.get_index(x_i))
        while i < len(self.train_data):
            
            x_j = self.find_closest(x_i, self.train_data)
            #print(x_j)
            i += 1
            if self.distance(x_i, x_j) < self.eps:
                self.G.add_node(self.get_index(x_j))
                
                if self.p is None:
                    self.p = 'KDE'
                self.G.add_edge(self.get_index(x_i), self.get_index(x_j), density = self._kernel_function(self.train_data, self.p, x_i, x_j))
                if self.clf.predict(x_j) != self.t:
                    x_i = x_j
                else:
                    break
        return x_j

                
        
    
    def check_constraints(self, a, b):
        density = self._kernel_function(self.train_data, self.p, a, b)
        dist = self.distance(a, b)
        classified_prob = self.clf.proba_predict(b)[self.t]
        if dist< self.eps:
            if density > self.t_p:
                if classified_prob < self.t_d:
                    return True
            
    def generate_paths(self, set, data):
        net = np.ones((self.n, self.n + 1, data.shape[1]))
        for i in range(len(set)):
            neighbours = self.find_neighbours(set[i], data)
            for blet in neighbours:
                generated_nodes = np.vstack([set[i], self.find_neighbours(blet, data)])
                net[i] = generated_nodes
        return net






    # def generate_counterfactual(self, x_i, target_label, max_iter = 10):
  
    #     x_j = self.create_path_to_counterfactual_class(x_i, target_label)
    #     self.t = target_label
    #     Ict = []
    #     iter = 0
    #     while iter < max_iter:
    #         others = self.find_neighbours(x_j, self.train_data[np.flatnonzero(self.training_labels == self.t)])
    #         self.visited = np.concatenate([self.visited, np.flatnonzero(others == self.train_data)])
    #         for x in others:
                
    #             if self.check_constraints(x_j, x) == True:
    #                 Ict.append(x)
                    
                   
    #         if Ict == []:
    #             for x in others:
    #                 x_j = x
    #         iter += 1
    #     return Ict, self.G
    


    def generate_counterfactual(self, x_i, target_label, max_iter = 10):
        self.x_i = self.get_index(x_i)
  
        x_j = self.create_path_to_counterfactual_class(x_i, target_label)
        self.visited.append(self.get_index(x_j))

        self.t = target_label
        Ict = 0
        iter = 0
        paths = []
        others = self.find_neighbours(x_j, self.train_data[np.flatnonzero(self.training_labels == self.t)])
        self.visited = self.visited + [self.get_index(arr) for arr in others]
        path_tensor = self.generate_paths(others, self.train_data[np.flatnonzero(self.training_labels == self.t)])
        path = np.zeros((1, self.train_data.shape[1]))

        for i in range(path_tensor.shape[0]):
            self.G.add_node(self.get_index(path_tensor[i][0]))
            self.G.add_edge(self.get_index(x_j), 
                            self.get_index(path_tensor[i][0]),
                            density = self._kernel_function(self.train_data, self.p, x_j, path_tensor[i][0]) )
            path_a = np.vstack([x_j, path_tensor[i][0]])
            for k in range(1, path_tensor.shape[1]):
                
                if self.check_constraints(path_tensor[i][0], path_tensor[i][k]) == True:
                    self.G.add_node(self.get_index(path_tensor[i][k]))
                    self.G.add_edge(self.get_index(path_tensor[i][0]), 
                                    self.get_index(path_tensor[i][k]),
                                    density = self._kernel_function(self.train_data, self.p, path_tensor[i][0], path_tensor[i][k]) )
                    #Ict += 1
                    
                    path = np.vstack([path_a,  path_tensor[i][k]])
                    while iter < max_iter:
                        new_others = self.find_neighbours(path[-1], self.train_data[np.flatnonzero(self.training_labels == self.t)])
                        self.visited = self.visited + [self.get_index(arr) for arr in new_others]
                        path_proxy = []
                        
                        for i in range(len(new_others)):
                            
                            if self.check_constraints(path[-1], new_others[i]) == True:
                                Ict += 1

                                if np.array([np.array_equal(new_others[i], p) for p in path]).sum() == 0:
                                
                                    if self.get_index(path[-1]) != self.get_index(path_tensor[i][k]):
                                        self.G.add_node(self.get_index(path[-1]))
                                    self.G.add_node(self.get_index(new_others[i]))
                                    self.G.add_edge(self.get_index(path[-1]),
                                                    self.get_index(new_others[i]), 
                                                    density = self._kernel_function(self.train_data, self.p, path_tensor[i][k], path[-1]) )

                                    path = np.vstack([path,  new_others[i]])
                                    if iter == max_iter - 1:
                                        path_proxy.append(path)
                                    self.visited.append(self.get_index(path[-1]))
                            else:
                                break
                                
                        
                            
                        iter += 1

        counterfactuals_indices = [node for node in self.G.nodes if self.G.out_degree(node) == 0]
        counterfactuals = self.train_data[counterfactuals_indices]

        return counterfactuals, Ict, self.G
    

    def recourse_paths(self):
        counterfactuals_indices = [node for node in self.G.nodes if self.G.out_degree(node) == 0]
        paths = []
        for ind in counterfactuals_indices:
            indices = list(nx.all_simple_paths(self.G, source = self.x_i, target=ind))[0]
            paths.append(self.train_data[indices])

        return paths