from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize,scale
from .util_dr import (jaccard_similarity,add_noise,internal_resample,get_auc,find_neighbor,knn_neighbor)
class dimension_reduction():
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
    
    rank: int. 
        The number of reduced dimensions, default=2.
        
    split_proportion: float in (0,1). default=0.7. need to specify if noise_type=='split'
        The proportion of training set in data splitting.

    n_repeat: int. default=100
        The number of repeats to measure consistency (run in parallel).

    norm: {True,False}
        Controls whether to conduct data normalization. 

    stratify: {True,False}
          Controls whether to conduct stratified sampling, default= False    
          
    rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.

    verbose: {True,False}
        Controls the verbosity.

    Returns
    ----------
    accuracy_values: pandas data frame of shape (n_repeat,7), columns = [data, method, perturbation, clustering, noise, sigma, rank, criteria, Accuracy]    
        IML model clustering accuracy of each repeat.
            data: name of data set
            method: IML methods. 
            perturbation: type of perturbation.
            clustering: clustering method applied on reduced dimensions.
            noise: type of noise added.
            sigma: level of variance of noise added. 
            rank: rank of reduced dimensions.
            criteria: consistency metrics. 
            Accuracy: clustering accuracy scores of each repeat.
    
    results: pandas data frame of shape (n_repeat,8), columns = [data, method, perturbation, clustering, noise, sigma, rank,  criteria, Consistency, Accuracy] 
        IML model interpretation consistency and prediction accuracy 
            data: name of data set
            method: IML methods. 
            perturbation: type of perturbation.
            clustering: clustering method applied on reduced dimensions.
            noise: type of noise added.
            sigma: level of variance of noise added. 
            rank: rank of reduced dimensions.
            criteria: consistency metrics. 
            Consistency: average pairwise consistency scores. 
            Accuracy: average clustering accuracy scores. 
 
 
        Can be saved and uploaded to the dashboard. 
   

    
    """
    def __init__(self,data,estimator,
                 K,
                 label=None,
                 perturbation = 'noise',
                 noise_type='normal',
                 sigma=1,
                 rank=2,
                 split_proportion=0.7,
                 n_repeat=50,
                 norm=True,
                 stratify=False,
                 rand_index=1,
                 verbose=True):
        
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

        self.rand_index=rand_index
            
        self.rank=rank

        if norm == True:
             self.X = normalize(scale(self.X))
        
        
        
    def fit(self, *args,**kwargs):
        self.embedding = []
        self.split_train_ind= []
        ## if estimator is a function
        if callable(self.estimator):
            self.embedding_no_noise=self.estimator(self.X,n_components=self.rank)

        self.embedding_no_noise=self.estimator.fit_transform(self.X)[:,:self.rank]

        for i in range(self.n_repeat):
            if self.verbose==True:
                print('Iter: ',i)
           
            if self.perturbation =='noise':
                x_new = add_noise(self.X, noise_type=self.noise_type,
                           sigma=self.sigma,
                           random_index=i*self.rand_index)
            else:
                x_new, x_test, y_train, y_test,indices_train,indices_test  = internal_resample(self.X,
                                                                                               self.label,                                                                                       proportion=self.split_proportion,stratify=self.stratify,
                                                                                       random_index=i*self.rand_index)  
                
                self.split_train_ind.append(indices_train)
            
            
            self.embedding.append (self.estimator.fit_transform(x_new))
            



    def get_consistency_clustering(self,data_name,method_name,cluster_func=None,
                               cluster_func_name='HC (ward)',user_metric=None,user_metric_name='user_metric'
                              ):

        
        """ 
        Parameters
        ----------
        data_name: str. 
            Name of the data set. 

        method_name: str. 
            Name of dimension reduction method. 

        cluster_func: callable. 
            clustering object applied to the reduced dimension

        cluster_func_name: str.  
            Name of clustering function. 

        user_metric: callable. 
            The user-defined evaluation metric for consistency, default = None.


        user_metric_name: str. default = 'user_metric'.
            Name of user-defined metric. 


        Returns
        ----------


        accuracy_values: pandas data frame of shape (n_repeat,7), columns = [data,method, perturbation,clustering, noise, sigma, rank, criteria,Accuracy]    
            IML model clustering accuracy of each repeat.
                data: name of data set
                method: IML methods. 
                perturbation: type of perturbation.
                clustering: clustering method applied on reduced dimensions.
                noise: type of noise added.
                sigma: level of variance of noise added. 
                rank: rank of reduced dimensions.
                criteria: consistency metrics. 
                Accuracy: clustering accuracy scores of each repeat.

        results: pandas dataframe of shape (n_repeat,8), columns = [data,method, perturbation,clustering, noise, sigma, rank,  criteria, Consistency , Accuracy] 
            IML model interpretation consistency and clustering accuracy 
                data: name of data set
                method: IML methods. 
                perturbation: type of perturbation.
                clustering: clustering method applied on reduced dimensions.
                noise: type of noise added.
                sigma: level of variance of noise added. 
                rank: rank of reduced dimensions.
                criteria: consistency metrics. 
                Consistency: average pairwise consistency scores. 
                Accuracy: average clustering accuracy scores. 


            Can be saved and uploaded to the dashboard. 

        """        
        
 
        if cluster_func is None:
            cluster_func=AgglomerativeClustering(n_clusters=self.K)

        self.predicted_label=[]
        for embed in self.embedding:
            fitted = cluster_func.fit(embed)
            self.predicted_label.append(fitted.labels_.astype(str))


          ####### pairwise consistency 
        self.consistency_values={}
        criterias = [('ARI',adjusted_rand_score),
                    ('Mutual Information',adjusted_mutual_info_score),
                    ('V Measure Score',v_measure_score),
                    ('Fowlkes Mallows Score',fowlkes_mallows_score)]
        if user_metric is not None:
            criterias=criterias+[(user_metric_name,user_metric)]
            
            
        for cri in criterias:
             self.consistency_values[cri[0]] = pd.DataFrame(columns = ['data','method','perturbation','clustering','noise','sigma','rank','criteria','Consistency']) ## initiate consistency pd

        for a in range(self.n_repeat):
            for b in range(a+1,self.n_repeat):
                ## label of each repeats 
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
                                                                                            cluster_func_name,

                                                                                         self.noise_type,
                                                                                         self.sigma,
                                                                                           self.rank, 
                                                                                         cri,
                                                                                         round(cri_func(label1,label2),3)] 
                 
  
        self.consistency_values = pd.concat(self.consistency_values.values(), ignore_index=True)

        self.consistency_mean = self.consistency_values.groupby(['data','method','perturbation','clustering','noise','sigma','rank','criteria'],as_index=False).mean()

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
                                     'clustering':cluster_func_name,
                                     'noise':self.noise_type,
                                     'sigma':self.sigma,
                                     'rank':self.rank,
                                     'criteria':cri,
                                     'Accuracy':[round(cri_func(label_true[y], self.predicted_label[y]),3) for y in range(self.n_repeat)]                                       
                                     }
                             )            

                    
            self.accuracy_values = pd.concat(self.accuracy_values.values(), ignore_index=True)
            self.accuracy_mean = self.accuracy_values.groupby(['data','method','perturbation','clustering','noise','sigma','rank','criteria'],as_index=False).mean()
            self.results = pd.merge(self.accuracy_mean ,self.consistency_mean ,
                                how='left',on = ['data','method','perturbation','clustering','noise','sigma','rank','criteria'])

        else: 
            #### if no label input
            self.results  = self.consistency_mean.copy()
            self.results['Accuracy']= np.nan

        
    def get_consistency_knn(self,data_name,method_name,Kranges=None):
        """ 
        Parameters
        ----------
        data_name: str. 
            Name of the data set. 

        method_name: str. 
            Name of dimension reduction method. 

        Kranges: list of int.  
            list of numbers of nearest neighbors. 


        Returns
        ----------


        consistency_knn: pandas dataframe of shape (n_repeat,9), columns = [data,method,noise, sigma, rank, K, criteria,Consistency]    

            Jaccard consistency scores in reduced dimensions under each K of each repeat.
                data: name of data set
                method: IML methods. 
                noise: type of noise added.
                sigma: level of variance of noise added. 
                rank: rank of reduced dimensions.
                K: number of nearest neighbors. 
                criteria: consistency metrics. 
                Consistency: consistency scores of each repeat.

        AUC: pandas dataframe of shape (n_repeat,8), columns = [data,method, noise, sigma, rank,  criteria, Consistency] 
            AUC scores of model interpretation consistency. 
                data: name of data set
                method: IML methods. 
                clustering: clustering method applied on reduced dimensions.
                noise: type of noise added.
                sigma: level of variance of noise added. 
                rank: rank of reduced dimensions.
                criteria: consistency metrics. 
                Consistency: average AUC consistency scores. 


            Can be saved and uploaded to the dashboard. 

        """        
        
        x = self.X
        if self.perturbation!='noise':
            raise ValueError("KNN consistency requires perturbation being 'noise'")
        else:
            if Kranges is None:
                Kranges = [int(np.round(i)) for i in np.linspace(2,x.shape[0]-1,num=50)]
            N = len(self.X)
            if N>500:
                r = np.random.RandomState()
                idx_I = np.sort(r.choice(N, size=500, replace=False)) # uniform sampling of subset of observations
            else:
                idx_I = range(N)

            self.consistency_knn = pd.DataFrame(columns = ['data','method','noise','sigma','rank','K','criteria','Consistency'])
            for embed in self.embedding:
                for Kr in Kranges:
                    nei1 = knn_neighbor(self.embedding_no_noise, Kr)
                    nei2 = knn_neighbor(embed, Kr)
                    jaccard=[]
                    for i in idx_I:
                        jaccard.append(jaccard_similarity(nei1[i], nei2[i]))
                    self.consistency_knn.loc[len(self.consistency_knn)]=[data_name,method_name,
                                               self.noise_type,
                                               self.sigma, self.rank,
                                               Kr,'NN-Jaccard-AUC',np.mean(jaccard)]

            self.consistency_knn_mean =self.consistency_knn.groupby(['data','method','noise','sigma','rank','K','criteria'],as_index=False).mean('Consistency')
            self.AUC = self.consistency_knn_mean.groupby(['data','method','noise','sigma','rank','criteria'])['Consistency'].apply(get_auc).reset_index()


    
    
    
    
    
