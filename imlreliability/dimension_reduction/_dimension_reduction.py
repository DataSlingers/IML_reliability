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
                 label=None,
                 perturbation = 'noise',
                 sigma=1,rank=2,
                 noise_type='normal',
                 split_proportion=0.7,
                 n_repeat=50,
                 rand_index=1,
#                  user_metric= None,
#                  user_metric_name='user_metric',
                 norm=True,
                 stratify=True,
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
                                                                                               self.label, 
                                                                                      proportion=self.split_proportion,
                                                                                       random_index=i*self.rand_index)  
                
                self.split_train_ind.append(indices_train)
            
            
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

        
    def consistency_knn(self,data_name,method_name,Kranges=None):
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
                                               Kr,'Jaccard',np.mean(jaccard)]

            self.consistency_knn_mean =self.consistency_knn.groupby(['data','method','noise','sigma','rank','K','criteria'],as_index=False).mean('Consistency')
            self.aucc = self.consistency_knn_mean.groupby(['data','method','noise','sigma','criteria','rank'])['Consistency'].apply(get_auc).reset_index()


    
    
    
    
    
