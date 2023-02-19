from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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
def add_noise( x, noise_type,sigma,random_index=None):
    
    noise_fun = getattr(np.random, noise_type)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)

    return(add_noise+x)

def clean_score(s):
    s=np.array(s)
    if len(s.shape)>1:
        s = s.mean(0)
    s=s.astype(np.float)
    return s

def get_rank(score):
    score[score == 0] = 'nan'
    score=score.astype(np.float)

    rank=np.zeros(len(score))
    rank[~np.isnan(score)]=(-np.abs(score[~np.isnan(score)])).argsort().argsort()+1

    rank[rank==0]=len(score)

    return rank
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))



