from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn import preprocessing
from .util_clustering import add_noise


class clustering():
    """ 
    Parameters
    ----------
    data: 
        X: N*M numpy array
    label:    
        Y: N*1 numpy array 
        
    estimator : clustering estimator object
        This is assumed to implement the scikit-learn estimator interface.
   
        
    noise_type: str, noise or laplace noise 
    
    
    sigma: float, variance of noise addition. 
    
    
    n_repeat: int, default=100
        Number of repeats to measure consistency (run in parallel).
    
    
    
    Attributes
    ----------
    
    clustering_accuracy: ARI with label  
        a list with length = n_repeat
    
    consistency: 
        a list with length = n_repeat*(n_repate-1)/2

    
    """
    def __init__(self,data,estimator,K,
                 label=None,
                 sigma=1,
                 noise_type='normal',  
                 n_repeat=50,
                    rand_index=None,
                 user_metric= None,
                 user_metric_name='user_metric',
                 scale=True,
                    verbose=True):
 
        self.sigma=sigma
        self.noise_type=noise_type
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.K = K
        self.X=data
        self.label=label
        self.estimator=estimator
        self.user_metric=user_metric
        self.user_metric_name=user_metric_name
 
        if scale:
            self.X = preprocessing.scale(self.X)
            
            
    def fit(self, *args,**kwargs):
        self.predicted_label = []
        for i in range(self.n_repeat):
            if self.verbose==True:
                print('Iter: ',i)
           
         
            x_new = add_noise(self.X, noise_type=self.noise_type,
                           sigma=self.sigma,
                           random_index=i*4)
            
            fitted = self.estimator.fit(x_new)
            self.predicted_label.append(fitted.labels_.astype(str))
            
            



    def consistency(self,data_name,method_name):
       
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
                                      'noise':self.noise_type,
                                      'sigma':self.sigma,
                                      'criteria':cri,
                                    'Accuracy':[cri_func(self.label, y) for y in self.predicted_label]                                       
                                     }
                             )            
            
            
            if self.user_metric is not None:
                self.accuracy_values[self.user_metric_name] = pd.DataFrame(
                                        {'data':data_name,
                                          'method':method_name,
                                          'noise':self.noise_type,
                                          'sigma':self.sigma,
                                        'criteria':self.user_metric_name,
                                          'Accuracy':[self.user_metric(self.label, y) for y in self.predicted_label]                          
                                         }
                                 )
            self.accuracy_values = pd.concat(self.accuracy_values.values(), ignore_index=True)
  
            self.accuracy_mean = self.accuracy_values.groupby(['data','method','noise','sigma','criteria'],as_index=False).mean()
    
          ####### pairwise consistency 
        self.consistency_values={}
        for cri,cri_func in [('ARI',adjusted_rand_score),
            ('Mutual Information',adjusted_mutual_info_score),
            ('V Measure Score',v_measure_score),
            ('Fowlkes Mallows Score',fowlkes_mallows_score)]:
            self.consistency_values[cri] = pd.DataFrame(
                                {'data':data_name,
                                  'method':method_name,
                                  'noise':self.noise_type,
                                  'sigma':self.sigma,
                                  'criteria':cri,
                                  'Consistency':[cri_func(self.predicted_label[i],self.predicted_label[j]) for i in range(self.n_repeat) for j in range(i+1,self.n_repeat)]                                        
                                 }
                         )

        if self.user_metric is not None:
            self.consistency[self.user_metric_name] = d.DataFrame(
                                        {'data':data_name,
                                          'method':method_name,
                                          'noise':self.noise_type,
                                          'sigma':self.sigma,
                                        'criteria':self.user_metric_name,
                                          'Consistency':[self.user_metric(self.predicted_label[i],self.predicted_label[j]) for i in range(self.n_repeat) for j in range(i+1,self.n_repeat)]                                        
                    }
                                 )
            
        self.consistency_values = pd.concat(self.consistency_values.values(), ignore_index=True)
        self.consistency_mean = self.consistency_values.groupby(['data','method','noise','sigma','criteria'],as_index=False).mean()
        self.results = pd.merge(self.accuracy_mean,self.consistency_mean,
                                how='left',on = ['data','method','noise','sigma','criteria'])

        self.results['Accuracy'] = [round(i,3) for i in self.results['Accuracy']]
        self.results['Consistency'] = [round(i,3) for i in self.results['Consistency']]

        
