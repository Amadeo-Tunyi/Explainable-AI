import numpy as np
from tqdm import tqdm
class RSLVQ:
    def __init__(self, num_prototypes_per_class, initialization_type = 'mean', sigma = 1, learning_rate = 0.05,max_iter = 100, cat_full = False, test_data = None, test_labels = None):

        self.max_iter = max_iter 
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_prototypes = num_prototypes_per_class
        self.cat_full = cat_full
        self.sigma = sigma
        self.initialization_type = initialization_type
        self.alpha = learning_rate
    
    
    

    def initialization(self, train_data, train_labels):
        if self.initialization_type == 'mean':
            """Prototype initialization: if number of prototypes is 1, prototype initialised is the mean
            if prototype is n>1, prototype initilised is the mean plus n-1 points closest to mean"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    list1.append(mu)#.astype(int))
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                if self.cat_full == True:
                    P = np.array(prototypes)
                else:
                    P = np.array(prototypes) + (0.01 *self.sigma* np.random.uniform(low = -1.0, high = 1.0, size = 1)*np.array(prototypes))
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    if self.cat_full == True:
                        mu = mu#.astype(int)
                        distances = [self.indicator_dist(mu, c) for c in class_data]
                    else:
                        distances = [(mu-c)@(mu-c).T for c in class_data]
                    index = np.argsort(distances)
                    indices = index[1:self.num_prototypes]
                    prototype = class_data[indices]
                    r = np.vstack((mu, prototype))
                    list2.append(r)
                    ind = []
                    for j in range(self.num_prototypes ):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)#.flatten()   
                prototypes = M.reshape(num_protos,num_dims)
                if self.cat_full == True:
                    P = np.array(prototypes)
                else:
                    P = np.array(prototypes) + (0.01 *1* np.random.uniform(low = -1.0, high = 1.0, size = 1)*np.array(prototypes))
            return np.array(new_labels).flatten(), P
        
        elif self.initialization_type == 'random':
            """Prototype initialization random: randomly chooses n points per class"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    random_int = np.random.choice(np.array(index))
                    prototype = train_data[random_int]
                    list1.append(prototype)
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                #regulate the prototypes, could also be done with GMM
                P = np.array(prototypes) + (0.01 *1* np.random.uniform(low = -1.0, high = 1.0, size = 1)*np.array(prototypes))
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    random_integers = np.random.choice(np.array(index), size=self.num_prototypes)
                    prototype = train_data[random_integers]
                    list2.append(prototype)
                    ind = []
                    for j in range(self.num_prototypes):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)  
                prototypes = M.reshape(num_protos,num_dims)
                P = np.array(prototypes) + (0.01 *1* np.random.uniform(low = -1.0, high = 1.0, size = 1)*np.array(prototypes))
            return np.array(new_labels).flatten(), P           
    
    def indicator_dist(self, a, b):
        l = 0
        for i in range(len(a)):
            if a[i] == b[i]:
                l += 0
            else:
                l += 1
        return l
    
    def indicator_differenc(self, a, b):
        arr = []
        for i in range(len(a)):
            if a[i] == b[i]:
                arr.append(0)
            else:
                arr.append(1)
        return np.array(arr)

    #prototype update       

    def gradient_ascent(self, train_data, train_labels, prototypes, proto_labels):
        """prototype optimization through gradient ascent"""
    

        #self.train_data = self.normalize(self.train_data)
    
        for i in range(len(train_data)):
            xi = train_data[i]
            x_label = train_labels[i]
            
            for j in range(prototypes.shape[0]):

                d = (xi - prototypes[j])
                c = 1/(self.sigma*self.sigma) 
                if self.proto_labels[j] == x_label:
                    self.prototypes[j] += self.alpha*(np.subtract(self.Pl_y(xi, j,  x_label), self.Pl(xi,j)))*c*d
                else:
                    self.prototypes[j] -= self.alpha*(self.Pl(xi,j))*c*d
    
        return self.prototypes 
    

    def predict_all(self, data, return_scores = False):

        """predict an array of instances""" 
        label = []
        #prototypes, _ = RSLVQ(data, labels, num_prototypes, max_iter)
        if return_scores == False:
            for i in range(data.shape[0]):
                xi = data[i]
                distances = np.array([np.linalg.norm(xi - p) for p in self.prototypes])
                index = np.argwhere(distances == distances.min())
                x_label = self.proto_labels[index]
                label.append(x_label)
            return np.array(label).flatten()
        else:
            predicted = []
            for i in range(len(data)):
                predicted.append(self.proba_predict(data[i]))
            return predicted 

        

    def likelihood_ratio(self, prototypes, train_data, train_labels):
    
        numerator = []
        denominator = []

    
    
         
        for i in range(len(train_data)):
            
            xi = train_data[i]
            x_label = train_labels[i]
            for j in range(len(prototypes)):
                if x_label == self.proto_labels[j]:
                    numerator.append(np.log(np.exp(self.inner_f(xi, prototypes[j]))))
        
            
                denominator.append(np.log(np.exp(self.inner_f(xi, prototypes[j]))))
        a = np.sum(np.array(numerator))
        b = np.sum(np.array(denominator))

    
            
        return a-b


    def fit(self, train_data, train_labels, show_plot = False):
        self.proto_labels, self.prototypes = self.initialization(train_data, train_labels)
        self.prototypes = self.prototypes.astype(float)
        import matplotlib.pyplot as plt
        loss =[]
        #iter = 0

        for iter in tqdm(range(self.max_iter), desc="Training", unit="epoch"):
            self.prototypes = self.gradient_ascent(train_data, train_labels, self.prototypes, self.proto_labels)
            predicted = []
            for i in range(len(train_data)):
                predicted.append(self.predict(train_data[i]))
            val_acc = (np.array(predicted) == np.array(train_labels).flatten()).mean() * 100  
            lr = self.likelihood_ratio(self.prototypes, train_data, train_labels)
            loss.append(lr)
            #iter += 1
            
        if show_plot  == True:
            plt.plot(loss)
            plt.ylabel('log likelihood ratio')
            plt.xlabel(' number of iterations')
        return self.prototypes, self.proto_labels
    
    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = []
        for i in range(len(test_data)):
            predicted.append(self.predict(test_data[i]))
        val_acc = (np.array(predicted) == np.array(test_labels).flatten()).mean() * 100 
        return val_acc

    def predict_cat(self, input, prototypes, proto_labels):
   
    #prototypes, _ = RSLVQ(data, labels, num_prototypes, max_iter)

       
         
   
        distances = np.array([self.indicator_dist(input, p) for p in prototypes])
        index = np.argmin(distances)
        x_label = proto_labels[index]
        
        return x_label


    def predict(self, input):
        """predicts only one output at the time, numpy arrays only, 
        might want to convert"""
        
   


       
         
   
        distances = np.array([np.linalg.norm(input - p) for p in self.prototypes])
        index = np.argmin(distances)
        x_label = self.proto_labels[index]
        
        return x_label
    
    def proba_predict(self, input, softmax = False):
        """probabilistic prediction of a point by approximation of distances of a point to closest prototypes
        the argmin is the desired class. If softmax is true, then predicted class is argmax"""
        scores = []
        closest_prototypes = []
        for i in np.unique(self.proto_labels):
            label_prototypes = self.prototypes[np.flatnonzero(self.proto_labels == i)]
            distances = np.array([np.linalg.norm(input - label_prototypes[j]) for j in range(label_prototypes.shape[0])])
            closest_prototype = label_prototypes[np.argmin(distances)]
            closest_prototypes.append(closest_prototype)
        dists = np.array([np.linalg.norm(input - prototype) for prototype in closest_prototypes])
        scores = np.array([d/dists.sum() for d in dists])
        if softmax == True:
            score = scores.copy()
            scores = [np.exp(-z)/(np.array(np.exp(-1*score)).sum()) for z in score]
        return scores 

    
  
    def inner_f(self, x, p):
        

        coef = -1/(2*(self.sigma *self.sigma))

        dist = (x -p)@(x- p).T
        return coef*dist

    def inner_derivative(self, x, p):
    
        coef = 1/(self.sigma *self.sigma)

        diff = (x -p) 
        return coef*diff
    



    
    def Pl_loss(self, unit, target_class):
        #updated_prototypes = self.fit()
        index = np.flatnonzero(self.proto_labels == target_class)[0]

        u = []
        for i in range(len(self.prototypes)):
            if target_class == self.proto_labels[i]:
                u.append(np.exp(self.inner_f(unit, self.prototypes[i])))
            else:
                u.append(0)
            numerator = np.array(u).sum()
        denominator = np.sum(np.array([np.exp(self.inner_f(unit, p)) for p in self.prototypes]))
        
        return numerator/denominator 

    
    def Pl_y(self, x, index, x_label):
        """probability of a point being correctly classified """
        u = np.exp(np.array([self.inner_f(x, self.prototypes[i]) for i in range(len(self.prototypes)) if x_label == self.proto_labels[i]]))
        numerator = np.exp(np.array(self.inner_f(x, self.prototypes[index])))
        denominator = u.sum()
        return numerator/denominator    
    def Pl(self, x, index):
        """probability of a point being classified"""
        inner = np.exp(np.array([self.inner_f(x, p) for p in  self.prototypes]))
        numerator = np.exp(np.array(self.inner_f(x, self.prototypes[index])))
        denominator = inner.sum()
        return numerator/denominator 