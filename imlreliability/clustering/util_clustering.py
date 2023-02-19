import pandas as pd
import numpy as np

def add_noise( x, noise_type,sigma,random_index=None):
    
    noise_fun = getattr(np.random, noise_type)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)

    return(add_noise+x)

def internal_resample(data,random_index=None, proportion=0.7,stratify=False):
    """
    Args:
      data: (x,y) format
      proportion:  percentage to split
      random_index: set random state 
      
    """
        
    (x,y)=data
    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y,random_state=random_index,test_size=0.3)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=random_index,test_size=0.3)

    return (x_train, x_test, y_train, y_test)