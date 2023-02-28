from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
def internal_resample(x,y,random_index=None, proportion=0.7,stratify=False):
    """Data splitting perturbation. 
    
    Parameters
    -------------
      x: array of shape (n_sampple, n_feature)
          predictors 
      
      y: array of shape (n_sampple,)
          response 
      
      proportion: float
          training data proportion in data splitting 
      
      rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.

      
    
    Returns 
    -------------
    x_train: array of shape (n_sampple*proportion, n_feature)
        predictors in the training set 
        
    
    x_test: array of shape (n_sampple*(1-proportion),)
        predictors in the test set 
        
        
    y_train: array of shape (n_sampple*(1-proportion), n_feature)
        response in the training set
    y_test:: array of shape (n_sampple*(1-proportion),)
        response in the test set
    
    indices_train: array of shape (n_sampple*proportion,)
        indices of training samples
        
    indices_test: array of shape (n_sampple*(1-proportion),)
        indices of test samples
    """
    indices =range(len(x))
    if stratify:
        x_train, x_test, y_train, y_test,indices_train,indices_test = train_test_split(x, y,indices,stratify=y,random_state=random_index,test_size=0.3)
    else:
        x_train, x_test, y_train, y_test,indices_train,indices_test = train_test_split(x, y,indices,random_state=random_index,test_size=0.3)

    return (x_train, x_test, y_train, y_test,indices_train,indices_test)
def add_noise(x, noise_type,sigma,random_index=None):
    """Noise addition perturbation. 
    
    Parameters
    -------------
      x: array of shape (n_sampple, n_feature)
          Predictors 
      
      noise_type: str
          Distribution type of noise, chosen from ['normal', 'laplace']
          
      sigma: float
          Controls variance of noise distribution 

      random_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.
    
    Returns 
    -------------
    newx: array of shape (n_sampple, n_feature)
        perturbed data after noise addition 

    """
    
    noise_fun = getattr(np.random, noise_type)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)
    newx=add_noise+x
    return(newx)

def clean_score(s):
    s=np.array(s)
    if len(s.shape)>1:
        s = s.mean(0)
    s=s.astype(float)
    return s

def get_rank(score):
    score[score == 0] = 'nan'
    score=score.astype(float)

    rank=np.zeros(len(score))
    rank[~np.isnan(score)]=(-np.abs(score[~np.isnan(score)])).argsort().argsort()+1

    rank[rank==0]=len(score)

    return rank
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))



