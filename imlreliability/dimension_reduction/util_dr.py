import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from numpy import trapz

def add_noise( x, noise_type,sigma,random_index=None):
    
    noise_fun = getattr(np.random, noise_type)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)

    return(add_noise+x)
def internal_resample(x,y=None,random_index=None, proportion=0.7,stratify=False):
    """
    Args:
      data: (x,y) format
      proportion:  percentage to split
      random_index: set random state 
      
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