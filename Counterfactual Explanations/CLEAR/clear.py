import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from collections import Counter
from scipy.spatial import distance
from helpers import check_sparsity
import random
from density_check import Kernel_obj
from helpers import generate_subsets, weighted_l1_norm
class CLEAR1:
    def __init__(self,model, num_points_neighbourhood,neighbourhood = 'unbalanced', backend = 'lvq', synthetic_generator_scheme = 'Gaussian', 
                 classification_threshold = 0.4, number_of_CFEs = 1, num_cols = None, cat_cols = None, set_immutable = [],  
                 regression = None, wachter_search_max = None, learning_rate = None, batch_size = None,
                 number_of_reg_runs = 50):
            
            #self.unit = (unit - self.train_data.mean(axis=0))/self.train_data.std(axis=0
            self.N = num_points_neighbourhood
            self.neighbourhood = neighbourhood
            self.model = model
            self.wachter_search_max = wachter_search_max
            #self.prototypes = self.model.fit()
            self.lr = learning_rate
            self.batch_size = batch_size
            self.r = regression
            self.num_cf = number_of_CFEs
            self.scheme = synthetic_generator_scheme
            self.num_cols = num_cols
            self.cat_cols = cat_cols
            self.c_t = classification_threshold
            self.immutable = set_immutable
            self.backend = backend
            self.n_runs = number_of_reg_runs 
            






    def perturb_vector(self, vector, num_to_perturb, target_class, mean=0, std_dev=1):
        """
        Perturb a vector by replacing a specified number of its values with random samples
        from a Gaussian distribution.

        Args:
        - vector: The input vector (list) to be perturbed.
        - num_to_perturb: The number of values to perturb.
        - mean: The mean of the Gaussian distribution (default is 0).
        - std_dev: The standard deviation of the Gaussian distribution (default is 1).

        Returns:
        - The perturbed vector with some values replaced by Gaussian samples.
        """
        if num_to_perturb < 0 or num_to_perturb > len(vector):
            raise ValueError("num_to_perturb should be between 0 and the length of the vector")

        # Create a copy of the input vector to avoid modifying the original
        perturbed_vector = vector.copy()
        seed = 42

        rng = np.random.default_rng(seed)
        counterfactual_class = self.train_data.loc[self.training_labels['labels'] == target_class]
        mean_counterfactual_class= counterfactual_class.mean(axis = 0)
        perturbation_factor = 0.8 #rng.uniform(low = 0, high = 1)
        difference_vector = mean_counterfactual_class - perturbed_vector

        # Choose random indices to perturb
        indices_to_perturb = random.sample(range(len(perturbed_vector)), num_to_perturb)

        # Perturb the selected indices
        for index in indices_to_perturb:
            
            perturbed_vector[index] += perturbation_factor * difference_vector[index]

        return perturbed_vector
    

    def perturb_col(self, unit, num_to_perturb, target_class, mean=0, std_dev=1):
        counterfactual_class = self.train_data.loc[self.training_labels['labels'] == target_class]
        columns = self.train_data.columns
        perturbed_vector = unit.copy()
        indices_to_perturb = random.sample(range(len(perturbed_vector)), num_to_perturb)
        cols_to_perturb = columns[indices_to_perturb]
        for col in cols_to_perturb:
            if col in self.cat_cols:
                perturbed_vector[col] = counterfactual_class.mode().iloc[0][col]
            else:
                
                seed = 42

                rng = np.random.default_rng(seed)
                #counterfactual_class = self.train_data.loc[self.training_labels['labels'] == target_class]
                mean_counterfactual_class= counterfactual_class.mean(axis = 0)
                perturbation_factor = 0.8 #rng.uniform(low = 0, high = 1)
                difference_vector = mean_counterfactual_class - perturbed_vector

                # Choose random indices to perturb
                #indices_to_perturb = random.sample(range(len(perturbed_vector)), num_to_perturb)
                perturbed_vector[col] += perturbation_factor * difference_vector[col]


        return perturbed_vector

            

        
    

   


    def synthetic_generator(self, train_data, training_labels, unit, target_class ):


        seed = 42
        unit_df = pd.DataFrame(np.array(unit).reshape((1, train_data.shape[1])), columns = self.train_data.columns)

        rng = np.random.default_rng(seed)
        number_of_points = (len(train_data)//3)*4
        if self.scheme == 'Point-Specific':
        
            
            components = []
            components_labels = []
            df = pd.DataFrame()
            if self.cat_cols is not None :
                print('no numerical column specified')
                num_cols = [col for col in train_data.columns if col not in self.cat_cols]
                n_classes = len(training_labels['labels'].unique())
                new_data = []
                new_labels = []
                for label in training_labels['labels'].unique():
                    #target_indices = np.flatnonzero(self.training_labels == i)
                    #target_points = self.train_data[target_indices]
                    target_points = train_data.loc[training_labels['labels'] == label]
                    #new_data = pd.DataFrame(columns = target_points.columns)

                    components_labels.extend([label]*number_of_points)
                    unit_df = pd.DataFrame(np.array(unit).reshape((1, train_data.shape[1])), columns = self.train_data.columns)
                    n = 0
                    while n<=number_of_points//n_classes:                    
                        for col in self.train_data.columns:
                            if col in num_cols:
                                if col not in self.immutable:
                                    unit_df[col] += rng.normal(target_points[col].mean(), target_points[col].std(),1)
                                else:
                                    unit_df[col] += 0
                            
                            elif col in self.cat_cols:
                                if col not in self.immutable:
                                    values = target_points[col].unique()
                                    index = random.sample(range(len(values)), 1)
                                    unit_df[col] = values[index]
                                else:
                                    unit_df[col] += 0
                            if self.backend == 'sklearn':
                                #if self.model.predict(pd.DataFrame(np.array(unit_df).reshape((1, train_data.shape[1])), columns= train_data.columns)) == label:
                                new_data.append(np.array(unit_df))
                                new_labels.append(label)

                                n += 1
                            elif self.backend == 'lvq':
                                #if self.model.predict(np.array(unit_df)) == label:
                                new_data.append(np.array(unit_df))
                                new_labels.append(label)

                                n += 1


            
                generated_data = pd.DataFrame(np.array(new_data).reshape((np.array(new_data).shape[0], np.array(new_data).shape[2])), columns= train_data.columns)
                synthetic_data1 = pd.concat([generated_data, train_data], axis = 0)
                generated_labels =pd.DataFrame(new_labels, columns= training_labels.columns)
                synthetic_label = pd.concat([generated_labels, training_labels], axis = 0)
                return generated_data, generated_labels
            


    

                

            elif self.cat_cols is None:

                for label in training_labels['labels'].unique():
                    #target_indices = np.flatnonzero(self.training_labels == i)
                    #target_points = self.train_data[target_indices]
                    target_points = train_data.loc[training_labels['labels'] == label]
                    mu, sigma = target_points.mean(axis = 0), target_points.cov()
                    components.append(rng.multivariate_normal(mu, sigma, number_of_points))
                    #components_labels.extend([label]*number_of_points)

                    
                data_new = np.vstack(components)
                data = np.vstack((data_new, train_data))
                #labels = np.hstack((np.array(components_labels),np.array(training_labels).reshape((training_labels.shape[0],))))
                synthetic_data = np.column_stack((data, labels))
                rng.shuffle(synthetic_data)
                generated_data = pd.DataFrame(synthetic_data[:,:synthetic_data.shape[1] - 1], columns= train_data.columns)
                #generated_labels =pd.DataFrame(synthetic_data[:,-1], columns= training_labels.columns)
                for col in self.immutable:
                    generated_data[col] = generated_data[col].apply(lambda x: unit_df[col])
                if self.backend == 'lvq':
                    labels = np.array([self.model.predict(np.array(generated_data)[i]) for i in range(len(generated_data))])
                    generated_labels =pd.DataFrame(labels, columns= training_labels.columns)
                elif  self.backend == 'sklearn':
                    labels = np.array([self.model.predict(self.model.predict(pd.DataFrame(np.array(generated_data)[i].reshape((1, train_data.shape[1])), columns= train_data.columns))) \
                                        for i in range(len(generated_data))])
                    generated_labels =pd.DataFrame(labels, columns= training_labels.columns) 
                return generated_data, generated_labels

        elif self.scheme == 'Gaussian':
            components = []
            components_labels = []
            df = pd.DataFrame()
            if self.cat_cols is not None :
                print('no numerical column specified')
                num_cols = [col for col in train_data.columns if col not in self.cat_cols]
                for label in training_labels['labels'].unique():
                    #target_indices = np.flatnonzero(self.training_labels == i)
                    #target_points = self.train_data[target_indices]
                    target_points = train_data.loc[training_labels['labels'] == label]
                    new_data = pd.DataFrame(columns = target_points.columns)
                    #components_labels.extend([label]*number_of_points)

                    
                    for col in num_cols:
                        mu = target_points[col].mean()
                        sigma = target_points[col].std()
                        #new_data[col] = np.random.normal(mu, sigma, number_of_points)
                        if col not in self.immutable:
                            new_data[col] = np.random.normal(mu, sigma, number_of_points)
                        else:
                            new_data[col] = new_data[col].apply(lambda x: unit_df[col])
                    for col in self.cat_cols:
                        values = target_points[col].unique()
                        count = Counter(target_points[col])
                        probabilities = tuple([count[values[i]]/len(target_points) for i in range(len(values))])
                        custm = stats.rv_discrete(name='custm', values=(values, probabilities))
                        if col not in self.immutable:
                            new_data[col] =  custm.rvs(size = number_of_points)
                        else:
                            new_data[col] = new_data[col].apply(lambda x: unit_df[col])
                
                        df_created = pd.concat([target_points, new_data], ignore_index=True, sort=False)
                    df_new = pd.concat([df, df_created], ignore_index=True, sort=False)
                    df = df_new
                # labels = np.hstack((np.array(components_labels),np.array(training_labels).reshape((training_labels.shape[0],))))
                # generated_labels = pd.DataFrame(labels, columns = training_labels.columns)
                generated_data = df
                if self.backend == 'lvq':
                    labels = np.array([self.model.predict(np.array(generated_data)[i]) for i in range(len(generated_data))])
                    generated_labels =pd.DataFrame(labels, columns= training_labels.columns)
                elif  self.backend == 'sklearn':
                    labels = np.array([self.model.predict(self.model.predict(pd.DataFrame(np.array(generated_data)[i].reshape((1, train_data.shape[1])), columns= train_data.columns))) \
                                        for i in range(len(generated_data))])
                    generated_labels =pd.DataFrame(labels, columns= training_labels.columns) 
                return generated_data, generated_labels

            

            elif self.cat_cols is None:

                for label in training_labels['labels'].unique():
                    #target_indices = np.flatnonzero(self.training_labels == i)
                    #target_points = self.train_data[target_indices]
                    target_points = train_data.loc[training_labels['labels'] == label]
                    mu, sigma = target_points.mean(axis = 0), target_points.cov()
                    components.append(rng.multivariate_normal(mu, sigma, number_of_points))
                    components_labels.extend([label]*number_of_points)

                    
                data_new = np.vstack(components)
                data = np.vstack((data_new, train_data))
                labels = np.hstack((np.array(components_labels),np.array(training_labels).reshape((training_labels.shape[0],))))
                synthetic_data = np.column_stack((data, labels))
                rng.shuffle(synthetic_data)
                generated_data = pd.DataFrame(synthetic_data[:,:synthetic_data.shape[1] - 1], columns= train_data.columns)
                #generated_labels =pd.DataFrame(synthetic_data[:,-1], columns= training_labels.columns)
                for col in self.immutable:
                    generated_data[col] = generated_data[col].apply(lambda x: unit_df[col])
                if self.backend == 'lvq':
                    labels = np.array([self.model.predict(np.array(generated_data)[i]) for i in range(len(generated_data))])
                    generated_labels =pd.DataFrame(labels, columns= training_labels.columns)
                elif  self.backend == 'sklearn':
                    labels = np.array([self.model.predict(self.model.predict(pd.DataFrame(np.array(generated_data)[i].reshape((1, train_data.shape[1])), columns= train_data.columns))) \
                                        for i in range(len(generated_data))])
                    generated_labels =pd.DataFrame(labels, columns= training_labels.columns) 
                return generated_data, generated_labels    
            

        elif self.scheme == 'Perturbation':
            lst = []
            synthetic_array = []
            label = []
            n_runs = 0
            if self.cat_cols is None:
                while n_runs < 10:
                    for i in range(train_data.shape[1]):
                        for j in range(train_data.shape[1]):
                            alpha = 1
                            perturbed = self.perturb_vector(np.array(unit), i+1, target_class, mean=0, std_dev=alpha*(0.9**(n_runs)))
                            lst.append(perturbed)
                    n_runs += 1
            else:
                while n_runs < 10:
                    for i in range(train_data.shape[1]):
                        for j in range(train_data.shape[1]):
                            alpha = 1
                            perturbed = self.perturb_col(unit, i+1, target_class, mean=0, std_dev=alpha*(0.9**(n_runs)))
                            lst.append(np.array(perturbed))
                    n_runs += 1
                       
            for x in lst:
                if self.backend == 'lvq':
                    label.append(self.model.predict(np.array(x)))
                    synthetic_array.append(x)

                elif self.backend == 'sklearn':
                    label.append(self.model.predict(pd.DataFrame(np.array(x).reshape((1, train_data.shape[1])), columns= train_data.columns)))
                    synthetic_array.append(x)
        
            generated_data = pd.DataFrame(synthetic_array, columns= train_data.columns)
            synthetic_data1 = pd.concat([generated_data, train_data], axis = 0, ignore_index =True)
            generated_labels =pd.DataFrame(label, columns= training_labels.columns)
            synthetic_label = pd.concat([generated_labels, training_labels], axis = 0, ignore_index= True)
            return synthetic_data1, synthetic_label
            
            
        

    
            

  
    
    def Mean_absolute_deviation(self, data):
        feature_medians = np.median(data, axis = 0)
        diff = np.absolute(data - feature_medians)
        MAD = np.median(diff, axis = 0)
        return MAD

        
    def cat_con_dist(self,a, b):
        if self.num_cols is None:
            self.num_cols = [col for col in self.train_data.columns if col not in self.cat_cols]
        num1 = a[self.num_cols]
        num2 = b[self.num_cols]
        cat1 = a[self.cat_cols]
        cat2 = b[self.cat_cols]
        manhattan_distance = distance.cityblock(cat1, cat2)
        MAD = ((abs(num1 - num2))/self.Mean_absolute_deviation(self.train_data[self.num_cols])).sum()
        return manhattan_distance + MAD
    
    def MAD (self, data, a, b):
        return ((abs(a - b))/self.Mean_absolute_deviation(data)).sum()

    

    def fit(self, data, labels):
        self.train_data = data
        self.training_labels = labels



    def boundary_set(self, lower_bound, upper_bound, target_class ):
        boundary_set = []
        for i in range(self.synthetic_data.shape[0]):
            if self.backend == 'lvq':
                if lower_bound < self.model.proba_predict(self.synthetic_data.iloc[i])[target_class] <= upper_bound:
                    boundary_set.append(self.synthetic_data.iloc[i])
            elif self.backend == 'sklearn':
                if lower_bound < self.model.predict_proba(pd.DataFrame(np.array(self.synthetic_data.iloc[i]).reshape((1, self.synthetic_data.shape[1])), columns = self.train_data.columns))[0][target_class] <= upper_bound:
                    boundary_set.append(self.synthetic_data.iloc[i])

        return boundary_set








    def Balanced_Neighbourhood(self, unit, target_class):
        neighbourhood = []

        self.synthetic_data, self.synthetic_labels = self.synthetic_generator(self.train_data, self.training_labels, unit, target_class)
        if target_class == None:
            target_class = self.target_class
        



        if self.backend == 'lvq':
            first_cut = self.boundary_set(0, self.c_t - 0.01, target_class)
            
            second_cut = self.boundary_set(self.c_t - 0.01, self.c_t + 0.01, target_class)
            third_cut = self.boundary_set(self.c_t + 0.01, 1, target_class)


            if self.neighbourhood == 'balanced':
                self.N = np.array([len(first_cut), len(second_cut), len(third_cut)]).min()*3
                for cut in [first_cut, second_cut, third_cut]:
                    print(len(cut))
                    
                    distances = []
                    for j in range(len(cut)):
                        if self.cat_cols is not None:
                            self.num_cols = [col for col in self.train_data.columns if col not in self.cat_cols]
                            distances.append([self.cat_cols(self.train_data, unit, cut[j]), j])
                        else:

                            distances.append([self.MAD(self.train_data, unit, cut[j]), j])
                    dist = np.array(distances).reshape((len(cut), 2))
                
                    sorted_distances = dist[dist[:,0].argsort()]
                    
                    index_list = sorted_distances[:,1].astype(int)[:self.N//3]
                        
                    selected_points = np.array(cut)[index_list]
                    
                    
                    neighbourhood.append(selected_points)
            else:
                
                for cut in [first_cut, second_cut, third_cut]:
                    print(len(cut))
                    
                    distances = []
                    for j in range(len(cut)):
                        if self.cat_cols is not None:
                            self.num_cols = [col for col in self.train_data.columns if col not in self.cat_cols]
                            distances.append([self.cat_cols(self.train_data, unit, cut[j]), j])
                        else:

                            distances.append([self.MAD(self.train_data, unit, cut[j]), j])
                    dist = np.array(distances).reshape((len(cut), 2))
                
                    sorted_distances = dist[dist[:,0].argsort()]
                    if self.N//3 <  len(cut):
                        index_list = sorted_distances[:,1].astype(int)[:self.N//3]
                        
                        selected_points = np.array(cut)[index_list]
                    else:
                        selected_points = np.array(cut)
                    
                    neighbourhood.append(selected_points)
                
                
            result = [item for sublist in neighbourhood for item in sublist]

            neighbourhood_d = np.array(result)
            neighbourhood_l = self.model.predict_all(neighbourhood_d)
        
        
        
        
        if self.backend == 'sklearn':
            first_cut = self.boundary_set(0, 0.3, target_class)
            
            second_cut = self.boundary_set(0.3, 0.7, target_class)
            third_cut = self.boundary_set(0.7, 1, target_class)
            

        
            
            for cut in [first_cut, second_cut, third_cut]:
                if self.neighbourhood == 'balanced':
                    self.N = np.array([len(first_cut), len(second_cut), len(third_cut)]).min()*3
                    for cut in [first_cut, second_cut, third_cut]:
                        
                        
                        distances = []
                        for j in range(len(cut)):
                            if self.cat_cols is not None:
                                self.num_cols = [col for col in self.train_data.columns if col not in self.cat_cols]
                                distances.append([self.cat_cols(self.train_data, unit, cut[j]), j])
                            else:

                                distances.append([self.MAD(self.train_data, unit, cut[j]), j])
                        dist = np.array(distances).reshape((len(cut), 2))
                    
                        sorted_distances = dist[dist[:,0].argsort()]
                        
                        index_list = sorted_distances[:,1].astype(int)[:self.N//3]
                        #print(len(index_list))
                            
                        selected_points = np.array(cut)[index_list]
                        
                        
                        neighbourhood.append(selected_points)
                else:
                    
                    for cut in [first_cut, second_cut, third_cut]:
                        #print(len(cut))
                        
                        distances = []
                        for j in range(len(cut)):
                            if self.cat_cols is not None:
                                self.num_cols = [col for col in self.train_data.columns if col not in self.cat_cols]
                                distances.append([self.cat_con_dist(unit, cut[j]), j])
                            else:

                                distances.append([self.MAD(self.train_data, unit, cut[j]), j])
                        dist = np.array(distances).reshape((len(cut), 2))
                    
                        sorted_distances = dist[dist[:,0].argsort()]
                        if self.N//3 <  len(cut):
                            index_list = sorted_distances[:,1].astype(int)[:self.N//3]
                            
                            selected_points = np.array(cut)[index_list]
                        else:
                            selected_points = np.array(cut)
                        
                        neighbourhood.append(selected_points)
                    
                    
            result = [item for sublist in neighbourhood for item in sublist]

            neighbourhood_d = np.array(result)
            neighbourhood_l = self.model.predict(pd.DataFrame(neighbourhood_d.reshape((len(neighbourhood_d), self.train_data.shape[1])), columns = self.train_data.columns))
        
            
        
        

        neighbourhood_set = pd.DataFrame(neighbourhood_d, columns= self.synthetic_data.columns)
        neighbourhood_set_labels = pd.DataFrame(neighbourhood_l, columns=self.synthetic_labels.columns)

            
        return neighbourhood_set, neighbourhood_set_labels


 
    def wachter_search(self, unit, target_class):
        data, labels = self.synthetic_generator(self.train_data, self.training_labels, unit, target_class)
        if self.immutable is not None:
            #immutable_features = unit[self.immutable]
            for feature in self.immutable:
                data = data.loc[data[feature] == unit[feature]]
            
        if self.backend == 'lvq':
            CF_space = data.loc[labels['labels'] == target_class]
            if self.cat_cols is not None:
                self.num_cols = [col for col in self.train_data.columns if col not in self.cat_cols]
                distances = np.array([self.cat_con_dist(CF_space.iloc[i], unit) for i in range(len(CF_space))])
            else:
                distances = np.array([self.MAD(data, CF_space.iloc[i], unit) for i in range(len(CF_space))])
            loss = np.array([self.model.Pl_loss(CF_space.iloc[i], target_class) for i in range(len(CF_space))])
            
            #loss = np.array([self.model.Pl_loss(x, self.target_class, self.prototypes) for x in CF_space])
            
            lst = np.array([np.linalg.norm(-x+ y) for x,y  in zip(distances, loss)])
            sorted_list = lst.argsort()[::2][::-1]
            #if self.wachter_search_max is not None:
            if self.wachter_search_max is not None:
                index = sorted_list[:self.wachter_search_max]
                return CF_space.iloc[index]
            else:
                index = sorted_list
                return CF_space.iloc[index]
            

        elif self.backend == 'sklearn':
            CF_space = data[labels['labels'] == target_class]
            if self.cat_cols == None:
                distances = np.array([self.MAD(data, CF_space.iloc[i], unit) for i in range(len(CF_space))])
            else:

                distances = np.array([self.cat_con_dist(CF_space.iloc[i], unit) for i in range(len(CF_space))])
            n_classes = len(self.training_labels['labels'].unique())
            t = np.zeros((1,n_classes))
            t[0][target_class] = 1.0
            loss = np.array([self.cross_entropy(t, self.model.predict_proba(pd.DataFrame(np.array(CF_space)[i].reshape((1,self.train_data.shape[1])), columns = data.columns))) for i in range(len(CF_space))])
            #loss = np.array([self.model.Pl_loss(x, self.target_class, self.prototypes) for x in CF_space])
            
            lst = np.array([np.linalg.norm(x+ y) for x,y  in zip(distances, loss)])
            sorted_list = lst.argsort()[::2][::-1]
            #if self.wachter_search_max is not None:
            if self.wachter_search_max is not None:
                index = sorted_list[:self.wachter_search_max]
                return CF_space.iloc[index]
            else:
                index = sorted_list
                return CF_space.iloc[index]
         




            

        

    def find_weights(self, unit, target_class):
        balenced_data, balanced_labels = self.Balanced_Neighbourhood(unit,  target_class)
        if self.r == None:
            from sklearn.linear_model import LogisticRegression
            log_model = LogisticRegression()
            x_train, x_test, y_train, y_test = train_test_split(balenced_data, balanced_labels)
            n = 0
            weights = []
            recalls = []
            while n < self.n_runs:
                n += 1
                log_model.fit(x_train, y_train)
                y_pred = log_model.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                if acc >= 0.8 and recall >= 0.8:
                    w = log_model.coef_
                    #print(acc, recall)
                    break
                else:
                    weight = log_model.coef_
                    weights.append(weight)
                    recalls.append(recall)
                    w = weights[np.argmax(np.array(recalls))]
                    #print(np.max(np.array(recalls)))
                    
            return w[0]
           
        elif self.r == 'lasso':
            from sklearn.linear_model import Lasso
            log_model = Lasso(alpha = 0.1)
            x_train, x_test, y_train, y_test = train_test_split(balenced_data, balanced_labels)
            n = 0
            weights = []
            recalls = []
            while n < self.n_runs:
                n += 1
                log_model.fit(x_train, y_train)
                y_pred = log_model.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                if acc >= 0.8 and recall >= 0.8:
                    w = log_model.coef_
                    break
                else:
                    weight = log_model.coef_
                    weights.append(weight)
                    recalls.append(recall)
                    w = weights[np.argmax(np.array(recalls))]
                    #print(np.max(np.array(recalls)))
            return w
        elif self.r == 'SVM':
            from sklearn.svm import SVR
            log_model = SVR(kernel = 'linear')
            x_train,  x_test,y_train, y_test = train_test_split(balenced_data, balanced_labels)
            n = 0
            weights = []
            recalls = []
            while n < self.n_runs:
                n += 1
                log_model.fit(x_train, y_train)
                y_pred = log_model.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                if acc >= 0.8 and recall >= 0.8:
                    w = log_model.coef_
                    #print('blat')
                    break
                else:
                    weight = log_model.coef_
                    weights.append(weight)
                    recalls.append(recall)
                    w = weights[np.argmax(np.array(recalls))]
                    #print(np.max(np.array(recalls)))
        
            return w[0]



        
    

    def estimated_b_counterfactual(self, unit, target_class):
        counterfactual_list = np.array(self.wachter_search(unit,  target_class))
        w = self.find_weights(unit, target_class)
        
        estimate = np.zeros((counterfactual_list.shape))
        for i in range(counterfactual_list.shape[0]):
            #b_counter = b_counterfactual.copy()
            #b_counter = b_counter.reshape((1,counterfactual_list.shape[1]))
            for k in range(counterfactual_list.shape[1]):
                b_counter = counterfactual_list[i].copy()
                b_counter[k] = 0
                #counterfactual_list[i][k] = 0
                
                estimate[i][k] = (-1*(np.multiply(w, b_counter)).sum())/(w[k] + 1e-8)

        return estimate
    

    def check_counterfactual(self, counterfactual, target):
        if self.backend == 'lvq':
            if self.model.proba_predict(counterfactual)[target] < self.c_t:
                return True
            else: 
                return False 
        elif self.backend == 'sklearn':
            if self.model.predict_proba(counterfactual)[0][target] > self.c_t:
                return True
            else: 
                return False        



    def cross_entropy(self, y_pred, y_true):
 
    # computing softmax values for predicted values

        loss = 0
        
        # Doing cross entropy Loss
        for i in range(len(y_pred[0])):
    
            # Here, the loss is computed using the
            # above mathematical formulation.
            loss = loss + (-1 * y_true[0][i]*np.log(y_pred[0][i] + 1e-8))
    
        return loss
 
        

    
    def generate_counterfactual(self,unit, target_class = 'opposite'):
        if self.backend == 'lvq':
            self.unit_class = self.model.predict(unit)
        elif self.backend == 'sklearn':
            #print(pd.DataFrame(np.array(unit).reshape((1, self.train_data.shape[1])), columns=self.train_data.columns))
            self.unit_class = self.model.predict(pd.DataFrame(np.array(unit).reshape((1, self.train_data.shape[1])), columns=self.train_data.columns))
            
        if target_class == 'opposite':
            if len(np.unique(np.array(self.training_labels))) != 2:
                raise ValueError('cannot be opposite number of classes greater than 2')
            else:
                self.target_class = np.array([i for i in np.unique(np.array(self.training_labels)) if i!=self.unit_class])[0]

        else:
            self.target_class = target_class
        
        # unit_class = self.model.predict(unit, self.prototypes, proto_labels)
        counterfactual_list = self.wachter_search(unit, self.target_class)
        estimations = self.estimated_b_counterfactual(unit,  self.target_class)
        FEs = []
        for i in range(len(estimations)):
            fidelity_error = self.fidelity_error(counterfactual_list.iloc[i], estimations[i], unit)
            FEs.append(fidelity_error)
            
        Errors = np.array(FEs)
        best_CFEs = Errors.argsort()
        chosen = counterfactual_list.iloc[best_CFEs]
        chosen_estimates = estimations[best_CFEs]

        #chosen = pd.DataFrame(best_CFEs, train_data.columns)
        indices = []
        for i in range(chosen.shape[0]):
            if self.backend =='lvq':

                if self.check_counterfactual(chosen.iloc[i], self.target_class) == True: #and self.check_counterfactual(chosen_estimates[i], self.target_class) == True:
                    num_features = 1
                    while num_features <= self.train_data.shape[1] + 1:
                        if check_sparsity(num_features).is_sparse(unit, chosen.iloc[i]):
                            indices.append(i)
                            #break
                        num_features += 1
                        
                    

            

            elif self.backend == 'sklearn':
                if self.check_counterfactual(pd.DataFrame(np.array(chosen)[i].reshape((1, self.train_data.shape[1])), columns = self.train_data.columns), self.target_class) == True \
                    and self.check_counterfactual(pd.DataFrame(np.array(chosen_estimates)[i].reshape((1, self.train_data.shape[1])), columns = self.train_data.columns), self.target_class) == True:
                    num_features = 1
                    while num_features <= self.train_data.shape[1] + 1:
                        if check_sparsity(num_features).is_sparse(unit, chosen.iloc[i]):
                            indices.append(i)
                            
                            break
                        num_features += 1
        if indices == []:
            print('something went wrong, repeat')
        else:
            norms = np.array([self.MAD(self.train_data[self.training_labels['labels'] == self.target_class], unit, np.array(chosen.iloc[i])) for i in indices])
            min_norm_index = np.argmin(norms)
            cf_index = indices[min_norm_index]

        
        return chosen.iloc[cf_index], pd.DataFrame(chosen_estimates[cf_index].reshape((1, self.train_data.shape[1])), columns=self.train_data.columns)
    
        

    def fidelity_error(self, x, y, unit):
        b_pert = np.linalg.norm(x - unit)
        est_b_pert = np.linalg.norm(y - unit)
        
        return abs(b_pert - est_b_pert)
    