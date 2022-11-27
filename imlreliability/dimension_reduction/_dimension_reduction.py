from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from numpy import trapz
import pandas as pd
import numpy as np

from .util_dr.py import (jaccard_similarity,add_noise,get_auc,find_neighbor,knn_neighbor)
class dimension_reduction():
    """ 
    Parameters
    ----------
    data: 
        X: N*M numpy array
    label:    
        Y: N*1 numpy array 
        
    estimator : dimension reduction estimator object
        This is assumed to implement the scikit-learn estimator interface.
   
    cluster_func: clustering object applied to reduced dimension
    
    noise_type: str (train/test split or noise addiction)
    
    split_proportion: float in (0,1). need to specify if noise_type=='split'
    
    sigma: float, level of noise. need to specify if noise_type!='split'
    
    
    n_repeat: int, default=100
        Number of repeats to measure consistency (run in parallel).
    
    
    
    Attributes
    ----------
    
    clustering_accuracy: ARI with label  
        a list with length = n_repeat
    
    consistency: 
        a list with length = n_repeat*(n_repate-1)/2

    
    """
    def __init__(self,data,estimator,
                 K,
    #             cluster_func= None,
                 label=None,
                 sigma=1,rank=2,
                 noise_type='normal',
                 n_repeat=50,
                 scale=True,
                 rand_index=None,
#                  user_metric= None,
#                  user_metric_name='user_metric',
                 verbose=True):
 
        self.sigma=sigma
        self.noise_type=noise_type
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.rank=rank
        self.K = K
        self.X=data
        self.label=label
        self.estimator=estimator

  
        if scale:
            self.X = preprocessing.scale(self.X)

    def fit(self, *args,**kwargs):
        self.embedding = []
        
        ## if estimator is a function
        if callable(self.estimator):
            self.embedding_no_noise=self.estimator(self.X,n_components=self.rank)
            
            
            
            
        self.embedding_no_noise=self.estimator.fit_transform(self.X)[:,:self.rank]

        for i in range(self.n_repeat):
            if self.verbose==True:
                print('Iter: ',i)
           
         
            x_new = add_noise(self.X, noise_type=self.noise_type,
                           sigma=self.sigma,
                           random_index=i*4)
            self.embedding.append (self.estimator.fit_transform(x_new))
            



    def consistency_clustering(self,data_name,method_name,cluster_func=None,
                               cluster_func_name='HC (ward)',user_metric=None,user_metric_name=None
                              ):

        if cluster_func is None:
            cluster_func=AgglomerativeClustering(n_clusters=self.K)

        self.predicted_label=[]
        for embed in self.embedding:
            fitted = cluster_func.fit(embed)
            self.predicted_label.append(fitted.labels_.astype(str))

         ### Accuracy 
        if self.label is not None:             
            
            self.accuracy_values={}
            for cri,cri_func in [('ARI',adjusted_rand_score),
                ('Mutual Information',adjusted_mutual_info_score),
                ('V Measure Score',v_measure_score),
                ('Fowlkes Mallows Score',fowlkes_mallows_score)]:
                self.accuracy_values[cri] = pd.DataFrame(
                                    {'data':data_name,
                                      'method':method_name,
                                    'clustering':cluster_func_name,
                                     'noise':self.noise_type,
                                      'sigma':self.sigma,
                                         'rank':self.rank,
                                      'criteria':cri,
                                    'Accuracy':[cri_func(self.label, y) for y in self.predicted_label]                                       
                                     }
                             )

         
            
            
            
            if user_metric is not None:
                self.accuracy_values[user_metric_name] = pd.DataFrame(
                                        {'data':data_name,
                                          'method':method_name,
                                   'clustering':cluster_func_name,
                                          'noise':self.noise_type,
                                          'sigma':self.sigma,
                                         'rank':self.rank,
                                        'criteria':user_metric_name,
                                          'Accuracy':[user_metric(self.label, y) for y in self.predicted_label]                          
                                         }
                                 )
            self.accuracy_values = pd.concat(self.accuracy_values.values(), ignore_index=True)

            self.accuracy_mean = self.accuracy_values.groupby(['data','method','clustering','noise','sigma','rank','criteria'],as_index=False).mean()
                
          ####### pairwise consistency 
        self.consistency_values={}
        for cri,cri_func in [('ARI',adjusted_rand_score),
                ('Mutual Information',adjusted_mutual_info_score),
                ('V Measure Score',v_measure_score),
                ('Fowlkes Mallows Score',fowlkes_mallows_score)]:
            self.consistency_values[cri] = pd.DataFrame(
                                    {'data':data_name,
                                      'method':method_name,
                                    'clustering':cluster_func_name,
                                     'noise':self.noise_type,
                                      'sigma':self.sigma,
                                         'rank':self.rank,
                                      'criteria':cri,
                                      'Consistency':[cri_func(self.predicted_label[i],self.predicted_label[j]) for i in range(self.n_repeat) for j in range(i+1,self.n_repeat)]                                        
                                     }
                             )

         
            
            
            
        if user_metric is not None:
            self.consistency_values[self.user_metric_name] = pd.DataFrame(
                                        {'data':data_name,
                                          'method':method_name,
                                   'clustering':cluster_func_name,
                                          'noise':self.noise_type,
                                          'sigma':self.sigma,
                                         'rank':self.rank,
                                        'criteria':user_metric_name,
                                  'Consistency':[user_metric(self.predicted_label[i],self.predicted_label[j]) for i in range(self.n_repeat) for j in range(i+1,self.n_repeat)]                                        
                                         }
                                 )
        self.consistency_values = pd.concat(self.consistency_values.values(), ignore_index=True)

        self.consistency_mean = self.consistency_values.groupby(['data','method','clustering','noise','sigma','rank','criteria'],as_index=False).mean()

            
        self.results = pd.merge(self.accuracy_mean ,self.consistency_mean ,
                                how='left',on = ['data','method','clustering','noise','sigma','rank','criteria'])

        self.results['Accuracy'] = [round(i,3) for i in self.results['Accuracy']]
        self.results['Consistency'] = [round(i,3) for i in self.results['Consistency']]

        
    def consistency_knn(self,data_name,method_name,Kranges=None):
        if Kranges is None:
            Kranges = [np.int(np.round(i)) for i in np.linspace(2,x.shape[0]-1,num=50)]
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
                                           Kr,'jaccard',np.mean(jaccard)]

        self.consistency_knn_mean =self.consistency_knn.groupby(['data','method','noise','sigma','rank','K','criteria'],as_index=False).mean('Consistency')
        self.aucc = self.consistency_knn_mean.groupby(['data','method','noise','sigma','criteria','rank'])['Consistency'].apply(get_auc).reset_index()

        self.aucc  =self.aucc .rename(columns={'Consistency': 'AUC'}) 

    
    
    
    
    