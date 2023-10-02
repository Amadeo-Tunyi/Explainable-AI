import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from scipy.spatial import distance
class CLEARM:
    def __init__(self,model, num_points_neighbourhood, classification_threshold = 0.4, number_of_CFEs = 1, num_cols = None, cat_cols = None, set_immutable = None,  regression = None, wachter_search_max = None, learning_rate = None, batch_size = None):
            
            #self.unit = (unit - self.train_data.mean(axis=0))/self.train_data.std(axis=0
            self.N = num_points_neighbourhood
            self.model = model
            self.wachter_search_max = wachter_search_max
            self.lr = learning_rate
            self.batch_size = batch_size
            self.r = regression
            self.num_cf = number_of_CFEs
            self.num_cols = num_cols
            self.cat_cols = cat_cols
            self.c_t = classification_threshold
            self.immutable = set_immutable
            



    def synthetic_generator(self, train_data, training_labels):


        seed = 42

        rng = np.random.default_rng(seed)
        
        number_of_points = (len(train_data)//3)*2
        components = []
        components_labels = []
        df = pd.DataFrame()
        if self.cat_cols is not None and self.num_cols is None:
            print('no numerical column specified')
            num_cols = [col for col in train_data.columns if col not in self.cat_cols]
            for label in training_labels['labels'].unique():
                #target_indices = np.flatnonzero(self.training_labels == i)
                #target_points = self.train_data[target_indices]
                target_points = train_data.loc[training_labels['labels'] == label]
                new_data = pd.DataFrame(columns = target_points.columns)
                components_labels.extend([label]*number_of_points)

                   
                for col in num_cols:
                    mu = target_points[col].mean()
                    sigma = target_points[col].std()
                    new_data[col] = np.random.normal(mu, sigma, number_of_points)
                for col in self.cat_cols:
                    values = target_points[col].unique()
                    print(values)
                    count = Counter(target_points[col])
                    probabilities = tuple([count[values[i]]/len(target_points) for i in range(len(values))])
                    custm = stats.rv_discrete(name='custm', values=(values, probabilities))
                    new_data[col] = custm.rvs(size = number_of_points)
            
                    df_created = pd.concat([target_points, new_data], ignore_index=True, sort=False)
                df_new = pd.concat([df, df_created], ignore_index=True, sort=False)
                df = df_new
            labels = np.hstack((np.array(components_labels),np.array(training_labels).reshape((training_labels.shape[0],))))
            generated_labels = pd.DataFrame(labels, columns = training_labels.columns)
            generated_data = df
            return generated_data, generated_labels
        elif self.num_cols is not None and self.cat_cols is None:
            print('no categorical column specified')
            cat_cols = [col for col in train_data.columns if col not in self.num_cols]
            
            df = pd.DataFrame()
            for label in training_labels['labels'].unique():
                #target_indices = np.flatnonzero(self.training_labels == i)
                #target_points = self.train_data[target_indices]
                target_points = train_data.loc[training_labels['labels'] == label]
                new_data = pd.DataFrame(columns = target_points.columns)
                components_labels.extend([label]*number_of_points)
                
                for col in self.num_cols:
                    mu = target_points[col].mean()
                    sigma = target_points[col].std()
                    new_data[col] = np.random.normal(mu, sigma, number_of_points)
                for col in cat_cols:
                    values = target_points[col].unique()
                    print(values)
                    count = Counter(target_points[col])
                    probabilities = tuple([count[values[i]]/len(target_points) for i in range(len(values))])
                    custm = stats.rv_discrete(name='custm', values=(values, probabilities))
                    new_data[col] = custm.rvs(size = number_of_points)
            
                    df_created = pd.concat([target_points, new_data], ignore_index=True, sort=False)
                df_new = pd.concat([df, df_created], ignore_index=True, sort=False)
                df = df_new
            labels = np.hstack((np.array(components_labels),np.array(training_labels).reshape((training_labels.shape[0],))))
            generated_labels = pd.DataFrame(labels, columns = training_labels.columns)
            generated_data = df
            return generated_data, generated_labels


        elif self.num_cols is not None and self.cat_cols is not None:
            df = pd.DataFrame()
            for label in training_labels['labels'].unique():
                #target_indices = np.flatnonzero(self.training_labels == i)
                #target_points = self.train_data[target_indices]
                target_points = train_data.loc[training_labels['labels'] == label]
                new_data = pd.DataFrame(columns = target_points.columns)
                components_labels.extend([label]*number_of_points)
                
                for col in self.num_cols:
                    mu = target_points[col].mean()
                    sigma = target_points[col].std()
                    new_data[col] = np.random.normal(mu, sigma, number_of_points)
                for col in self.cat_cols:
                    values = target_points[col].unique()
                    print(values)
                    count = Counter(target_points[col])
                    probabilities = tuple([count[values[i]]/len(target_points) for i in range(len(values))])
                    custm = stats.rv_discrete(name='custm', values=(values, probabilities))
                    new_data[col] = custm.rvs(size = number_of_points)
            
                df_created = pd.concat([target_points, new_data], ignore_index=True, sort=False)
                df_new = pd.concat([df, df_created], ignore_index=True, sort=False)
                df = df_new
            labels = np.hstack((np.array(components_labels),np.array(training_labels).reshape((training_labels.shape[0],))))
            generated_labels = pd.DataFrame(labels, columns = training_labels.columns)
            generated_data = df
        

            return generated_data, generated_labels

            

        elif self.num_cols is None and self.cat_cols is None:

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
            generated_labels =pd.DataFrame(synthetic_data[:,-1], columns= training_labels.columns)
            return generated_data, generated_labels

    # def synthetic_generator(self):
    
        
    #     seed = 42

    #     rng = np.random.default_rng(seed)
    #     self.train_data = self.normalize(self.train_data)
    #     number_of_points = (len(self.train_data)//3)*2
    #     components = []
    #     components_labels = []
    #     for i in np.unique(self.training_labels):
    #         target_indices = np.flatnonzero(self.training_labels == i)
    #         target_points = self.train_data[target_indices]
    #         mu, sigma = np.mean(target_points, axis = 0), np.cov(target_points.T)
    #         components.append(rng.multivariate_normal(mu, sigma, number_of_points))
    #         components_labels.extend([i]*number_of_points)

     
    #     data_new = np.vstack(components)
    #     data = np.vstack((data_new, self.train_data))
    #     labels = np.hstack((np.array(components_labels),self.training_labels.reshape((self.training_labels.shape[0],))))
    #     synthetic_data = np.column_stack((data, labels))
    #     rng.shuffle(synthetic_data)
    #     return synthetic_data[:,:synthetic_data.shape[1] - 1], synthetic_data[:,-1]
        
    
    def Mean_absolute_deviation(self, data):
        feature_medians = np.median(data, axis = 0)
        diff = np.absolute(data - feature_medians)
        MAD = np.median(diff, axis = 0)
        return MAD

        
    def cat_con_dist(self, data, a, b):
        if self.num_cols is not None and self.cat_cols is not None:
            num1 = a[self.num_cols]
            num2 = b[self.num_cols]
            cat1 = a[self.cat_cols]
            cat2 = b[self.cat_cols]
            manhattan_distance = distance.cityblock(cat1, cat2)
            MAD = ((abs(num1 - num2))/self.Mean_absolute_deviation(data)).sum()
            return manhattan_distance + MAD
        elif self.num_cols is not None:
            print('No categorical Values provided')
            num1 = a[self.num_cols]
            num2 = b[self.num_cols]  
            return ((abs(num1 - num2))/self.Mean_absolute_deviation(data)).sum()   
        elif self.cat_cols is not None:
            cat1 = a[self.cat_cols]
            cat2 = b[self.cat_cols]
            return distance.cityblock(cat1, cat2)
        elif self.num_cols is None and self.cat_cols is None:
            #print('Numerical and Categorical Values not specified, assumed to be all continuous')
            columns =  data.columns
            num1 = a[columns]
            num2 = b[columns]
            return ((abs(num1 - num2))/self.Mean_absolute_deviation(data)).sum()  
        

    
    
    # def normalize(self, data):
    
   
        
    
    # # Normalizing all the n features of X.
        
    #     data = (data - data.mean(axis=0))/data.std(axis=0)
        
    #     return data





    def Balanced_Neighbourhood(self, train_data, training_labels,unit, unit_class, target_class):
        neighbourhood = []
        neighbourhood_labels = []
        synthetic_data, synthetic_labels = self.synthetic_generator(train_data, training_labels)
        
            
        for i in [unit_class, target_class]:
            class_labels = np.flatnonzero(np.array(synthetic_labels) == i)
            label_data = synthetic_data[synthetic_labels == i]
            class_distances = []
            for j in class_labels:
                class_distances.append([self.cat_con_dist(label_data, unit, synthetic_data.iloc[j]), j])
               
                
            class_dist = np.array(class_distances).reshape((len(class_labels), 2))
            sorted_class_distances = class_dist[class_dist[:,0].argsort()]
            index_list = sorted_class_distances[:,1].astype(int)[:self.N//2]
            selected_points = np.array(synthetic_data)[index_list]
            neighbourhood.append(selected_points)
            neighbourhood_labels.append(np.array(synthetic_labels)[index_list])
                
    
        neighbourhood_data = np.column_stack((np.array(neighbourhood).reshape(np.array(neighbourhood).shape[0]*np.array(neighbourhood).shape[1], np.array(neighbourhood).shape[2]), np.array(neighbourhood_labels).reshape(self.N, 1)))
        np.random.shuffle(neighbourhood_data)
        neighbourhood_set = pd.DataFrame(neighbourhood_data[:, 0:neighbourhood_data.shape[1] - 1], columns= synthetic_data.columns)
        neighbourhood_set_labels = pd.DataFrame(neighbourhood_data[:,-1], columns=synthetic_labels.columns)

            
        return neighbourhood_set, neighbourhood_set_labels

    def cross_entropy(self, y_pred, y_true):
 
    # computing softmax values for predicted values

        loss = 0
        
        # Doing cross entropy Loss
        for i in range(len(y_pred[0])):
    
            # Here, the loss is computed using the
            # above mathematical formulation.
            loss = loss + (-1 * y_true[0][i]*np.log(y_pred[0][i] + 1e8))
    
        return loss
 
    def wachter_search(self, unit, train_data, training_labels, target_class):
        from sklearn.metrics import log_loss
        data, labels = self.synthetic_generator(train_data, training_labels)
        if self.immutable is not None:
            #immutable_features = unit[self.immutable]
            for feature in self.immutable:
                data = data[data[feature] == unit[feature]]
            

            
        #data = self.normalize(data)
        CF_space = data[labels['labels'] == target_class]
        distances = np.array([self.cat_con_dist(data, CF_space.iloc[i], unit) for i in range(len(CF_space))])
        n_classes = len(training_labels['labels'].unique())
        t = np.zeros((1,n_classes))
        t[0][target_class] = 1.0
        loss = np.array([self.cross_entropy(t, self.model.predict_proba(pd.DataFrame(np.array(CF_space)[i].reshape((1,train_data.shape[1])), columns = data.columns))) for i in range(len(CF_space))])
        #loss = np.array([self.model.Pl_loss(x, self.target_class, self.prototypes) for x in CF_space])
        
        lst = np.array([np.linalg.norm(x+ y) for x,y  in zip(distances, loss)])
        sorted_list = lst.argsort()[::2][::-1]
        #if self.wachter_search_max is not None:
        if self.wachter_search_max is not None:
            index = sorted_list[:self.wachter_search_max]
            return CF_space.iloc[index]
        else:
            index = sorted_list[0]
            return CF_space.iloc[index]

    # def wachter_search(self):
    #     data, labels = self.synthetic_generator()
    #     #data = self.normalize(data)
    #     CF_space = data[np.flatnonzero(labels == self.target_class)]
    #     distances = np.array([((abs(x - self.unit))/self.Mean_absolute_deviation(CF_space)).sum() for x in CF_space])
        
    #     loss = np.array([self.model.Pl_loss(x, self.target_class, self.prototypes) for x in CF_space])
        
    #     lst = np.array([np.linalg.norm(-x+ y) for x,y  in zip(distances, loss)])
    #     sorted_list = lst.argsort()[::2][::-1]
    #     if self.wachter_search_max is not None:
    #         index = sorted_list[:self.wachter_search_max]
    #         return CF_space[index]
    #     else:
    #         index = sorted_list[0]
    #         return np.array([CF_space[index]])

            

        

    def find_weights(self, train_data, training_labels,unit, unit_class, target_class):
        balenced_data, balanced_labels = self.Balanced_Neighbourhood(train_data, training_labels,unit, unit_class, target_class)
        if self.r == None:
            from sklearn.linear_model import LogisticRegression
            log_model = LogisticRegression()
            log_model.fit(balenced_data, balanced_labels)
            w = log_model.coef_
            return w[0]
        elif self.r == 'lasso':
            from sklearn.linear_model import Lasso
            log_model = Lasso(alpha = 0.1)
            log_model.fit(balenced_data, balanced_labels)
            w = log_model.coef_
            return w
        elif self.r == 'SVM':
            from sklearn.svm import SVR
            log_model = SVR(kernel = 'linear')
            log_model.fit(balenced_data, balanced_labels)
            w = log_model.coef_
            return w[0]



        
    

    def estimated_b_counterfactual(self, train_data, training_labels,unit, unit_class, target_class):
        counterfactual_list = np.array(self.wachter_search(unit, train_data, training_labels, target_class))
        w = self.find_weights(train_data, training_labels,unit, unit_class, target_class)
        
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

        if self.model.predict_proba(counterfactual)[0][target] > self.c_t:
            return True
        else: 
            return False 
       
        

    
    def generate_counterfactual(self, train_data, training_labels,unit, unit_class, target_class):

        counterfactual_list = self.wachter_search(unit, train_data, training_labels, target_class)
        estimations = self.estimated_b_counterfactual(train_data, training_labels,unit, unit_class, target_class)
        FEs = []
        for i in range(len(estimations)):
            fidelity_error = self.fidelity_error(counterfactual_list.iloc[i], estimations[i], unit)
            FEs.append(fidelity_error)
            
        Errors = np.array(FEs)
        best_CFEs = Errors.argsort()[:self.num_cf]
        chosen = counterfactual_list.iloc[best_CFEs]

        #chosen = pd.DataFrame(best_CFEs, train_data.columns)
        indices = []
        for i in range(chosen.shape[0]):
            if self.check_counterfactual(pd.DataFrame(np.array(chosen)[i].reshape((1, train_data.shape[1])), columns = train_data.columns), target_class) == True:
                indices.append(i)
        if indices == []:
            print('something went wrong, repeat')
                #print(self.model.proba_predict(chosen.iloc[i], self.prototypes, proto_labels))
        
        return chosen.iloc[indices]
    


    def fidelity_error(self, x, y, unit):
        b_pert = np.linalg.norm(x - unit)
        est_b_pert = np.linalg.norm(y - unit)
        
        return abs(b_pert - est_b_pert)