from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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
    print(noise_fun)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)
    newx=add_noise+x
    return(newx)

def internal_resample(x,y=None, proportion=0.7,stratify=False,random_index=None):
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

