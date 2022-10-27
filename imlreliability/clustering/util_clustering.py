import pandas as pd
import numpy as np

def add_noise( x, noise_type,sigma,random_index=None):
    
    noise_fun = getattr(np.random, noise_type)
    np.random.seed(random_index)
    add_noise = noise_fun(0, np.std(x)*sigma, x.shape)

    return(add_noise+x)

