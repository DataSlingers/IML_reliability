import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from numpy import trapz

def add_noise( x, noise_type,sigma,random_index=None):
    """Noise addition perturbation. 
    
    Parameters
    -------------
      x: arrary of shape (n_sampple, n_feature)
          Predictors 
      
      noise_type: str
          Distirbution type of noise, chosen from ['normal','laplace']
          
      sigma: float
          Conrtols variance of noise distribution 

      random_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.
    
    Returns 
    -------------
    newx: arrary of shape (n_sampple, n_feature)
        perturbed data after noise addition 

    """  
    noise_fun = getattr(np.random, noise_type)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)
    newx=add_noise+x
    return(newx)
def internal_resample(x,y=None,random_index=None, proportion=0.7,stratify=False):
    """Data spliting perturbation. 
    
    Parameters
    -------------
      x: arrary of shape (n_sampple, n_feature)
          predictors 
      
      y: array of shape (n_sampple,)
          response 
      
      proportion: float
              training data proportion in data spliting 

      stratify: {True,False}
          Controls whether to conduct stratified sampling 

      rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.

    Returns 
    -------------
    x_train: arrary of shape (n_sampple*proportion, n_feature)
        predictors in training set 
        
    
    x_test: arrary of shape (n_sampple*(1-proportion),)
        predictors in test set 
        
        
    y_train: arrary of shape (n_sampple*(1-proportion), n_feature)
        response in training set
    y_test:: arrary of shape (n_sampple*(1-proportion),)
        response in test set
    
    indices_train: array of shape (n_sampple*proportion,)
        indices of training samples
        
    indices_test: array of shape (n_sampple*(1-proportion),)
        indices of test samples
    """    
    indices =range(len(x))
    if y is None:
        y=indices
    if stratify:
        x_train, x_test, y_train, y_test,indices_train,indices_test = train_test_split(x, y,indices,stratify=y,random_state=random_index,test_size=0.3)
    else:
        x_train, x_test, y_train, y_test,indices_train,indices_test = train_test_split(x, y,indices,random_state=random_index,test_size=0.3)

    return (x_train, x_test, y_train, y_test,indices_train,indices_test)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

                
def get_auc(y,dx=0.02):
    y = np.array(y)
    area = trapz(y, dx=dx)
    if area>1:
        area=1
    return area
def find_neighbor(x, radiu):
    neigh = NearestNeighbors(radius=radiu)
    neigh.fit(x)
    rng = neigh.radius_neighbors()
    return(rng[1])
def knn_neighbor(x, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(x)
    rng = neigh.kneighbors()
    return(rng[1])