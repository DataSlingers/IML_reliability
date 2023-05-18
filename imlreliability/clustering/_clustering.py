from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.preprocessing import normalize,scale
from .util_clustering import add_noise,internal_resample
import pandas as pd
import numpy as np

class clustering():
    """ 
    Parameters
    ----------
    data: array of shape (N, M)
            
    estimator: estimator object
        This is assumed to implement the scikit-learn estimator interface.
    
    K: int
        The number of clusters. 
    
    label: array of shape (N,1) or None. default = None.
        True cluster labels

    perturbation: {'noise','split'}
        Controls the way of perturbation. 
            noise: conduct noise addition.
            split: conduct data splitting. 
            
    noise_type: {'normal','laplace'}. need to specify if noise_type=='noise'
        Distribution type of noise. 

    sigma: float. need to specify if noise_type=='noise'
        Controls variance of noise distribution 
    
    n_repeat: int, default=100
        The number of repeats to measure consistency (run in parallel).

    split_proportion: float in (0,1). default=0.7. need to specify if noise_type=='split'
        The proportion of training set in data splitting.
        
    user_metric: callable. default = None.
        User-defined evaluation metric for consistency. 
        
        
    user_metric_name: str. default = 'user_metric'.
        Name of user-defined metric. 
        
    norm: {True,False}
        Controls whether to conduct data normalization. 

    stratify: {True,False}
          Controls whether to conduct stratified sampling     
          
    rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.

    verbose: {True,False}
        Controls the verbosity.

    Returns
    ----------
    

    accuracy_values: pandas dataframe of shape (n_repeat,7), columns = [data, method, perturbation, noise, sigma, criteria, Accuracy]    
        IML model clustering accuracy of each repeat.
            data: name of data set
            method: IML methods. 
            perturbation: type of perturbation.
            noise: type of noise added.
            sigma: level of variance of noise added. 
            criteria: consistency metrics. 
            Accuracy: clustering accuracy scores of each repeat.
    
    results: pandas data frame of shape (n_repeat,8), columns = [data,method, perturbation, noise, sigma, criteria, Consistency  Accuracy] 
        IML model interpretation consistency and clustering accuracy 
            data: name of data set
            method: IML methods. 
            perturbation: type of perturbation.
            noise: type of noise added.
            sigma: level of variance of noise added. 
            criteria: consistency metrics. 
            Consistency: average pairwise consistency scores. 
            Accuracy: average clustering accuracy scores. 
 
 
        Can be saved and uploaded to the dashboard. 
    
    """
    

    def __init__(self,data,estimator,K,
                 label=None,
                 perturbation = 'noise',
                 sigma=1,
                 noise_type='normal',  
                 n_repeat=50,
                 split_proportion=0.7,
                 user_metric= None,
                 user_metric_name='user_metric',
                 rand_index=None,
                 norm=True,
                 stratify=True,
                 verbose=True):
        
#          if perturbation not in ['noise','split']:
             
#                 raise ValueError("results: perturbation must be one of %r." % ['noise','split'])
            
        self.perturbation=perturbation
        
        self.sigma=sigma
        self.noise_type=noise_type
        self.split_proportion=split_proportion
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.K = K
        self.X=data
        self.label=label
        self.estimator=estimator
        self.user_metric=user_metric
        self.user_metric_name=user_metric_name
        self.rand_index=rand_index
            
        if norm == True:
             self.X = normalize(scale(self.X))
    def fit(self, *args,**kwargs):
        self.predicted_label = []
        self.split_train_ind= []
        for i in range(self.n_repeat):
            if self.verbose==True:
                print('Iter: ',i)
           
            if self.perturbation =='noise':
                x_new = add_noise(self.X,noise_type=self.noise_type,
                           sigma=self.sigma,
                           random_index=i*self.rand_index)
            else:
                x_new, x_test, y_train, y_test,indices_train,indices_test  = internal_resample(self.X, self.label, 
                                           proportion=self.split_proportion,
                           random_index=i*self.rand_index)                
                self.split_train_ind.append(indices_train)
            
            
            fitted = self.estimator.fit(x_new)
            self.predicted_label.append(fitted.labels_.astype(str))
            
            



    def get_consistency(self,data_name,method_name):
        ####### pairwise consistency 
        
        
        self.consistency_values={}
        criterias = [('ARI',adjusted_rand_score),
                    ('Mutual Information',adjusted_mutual_info_score),
                    ('V Measure Score',v_measure_score),
                    ('Fowlkes Mallows Score',fowlkes_mallows_score)]
        if self.user_metric is not None:
            criterias=criterias+[(self.user_metric_name,self.user_metric)]
            
            
        for cri in criterias:
             self.consistency_values[cri[0]] = pd.DataFrame(columns = ['data','method','perturbation','noise','sigma','criteria','Consistency']) ## initiate consistency pd
        
        for a in range(self.n_repeat):
            for b in range(a+1,self.n_repeat):
                ## label of each repeats 
                print(self.perturbation)
                if self.perturbation!='noise':
                    subset=(list(set(self.split_train_ind[a]) & set(self.split_train_ind[b])))
                    label1= [self.predicted_label[a][w] for w in [self.split_train_ind[a].index(v) for v in subset]]
                    label2= [self.predicted_label[b][w] for w in [self.split_train_ind[b].index(v) for v in subset]]
                else:
                    subset = range(len(self.X))
                    label1 = self.predicted_label[a]
                    label2 = self.predicted_label[b]
   
                for cri,cri_func in criterias:
                    self.consistency_values[cri].loc[len(self.consistency_values[cri])]=[data_name,
                                                                                         method_name,
                                                                                         self.perturbation,
                                                                                         self.noise_type,
                                                                                         self.sigma,
                                                                                         cri,
                                                                                         round(cri_func(label1,label2),3)] 
                                     
                             


                    
        self.consistency_values2 = pd.concat(self.consistency_values.values(), ignore_index=True)
        self.consistency_mean = self.consistency_values2.groupby(['data','method','perturbation','noise','sigma','criteria'],as_index=False).mean()
        
         ### Accuracy 
        if self.label is not None:        
            self.accuracy_values={}
            
            
            if self.perturbation=='noise':
                label_true = [self.label for i in range(self.n_repeat)]
            else:
                label_true = [[self.label[a] for a in self.split_train_ind[i]] for i in range(self.n_repeat)]

           
            for cri,cri_func in criterias:
                self.accuracy_values[cri] = pd.DataFrame(
                                    {'data':data_name,
                                      'method':method_name,
                                     'perturbation':self.perturbation,
                                      'noise':self.noise_type,
                                      'sigma':self.sigma,
                                      'criteria':cri,
                                    'Accuracy':[round(cri_func(label_true[y], self.predicted_label[y]),3) for y in range(self.n_repeat)]                                       
                                     }
                             )            
  
            self.accuracy_values = pd.concat(self.accuracy_values.values(), ignore_index=True)
  
            self.accuracy_mean = self.accuracy_values.groupby(['data','method','perturbation','noise','sigma','criteria'],as_index=False).mean()

            self.results = pd.merge(self.accuracy_mean,self.consistency_mean,
                                how='left',on = ['data','method','perturbation','noise','sigma','criteria'])

        else: 
            #### if no label input
            self.results  = self.consistency_mean.copy()
            self.results['Accuracy']= np.nan
        
        