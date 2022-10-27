import IMLconsistency

import pandas as pd
import numpy as np


#####################
## import spambase data 
########################
spam = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',header=None).to_numpy()
x = spam[:, :-1]
y = spam[:, -1]
    
    
data=(x,y)
noise_type='normal'
K=len(set(y))


#######################
#### PCA
########################

## estimator has to be an object
from sklearn.decomposition import PCA
esti=PCA()

model = dr(x,estimator=esti,K=len(set(y)),
                 label=y,
                rank=2,
                 sigma=1,
                 noise_type='normal',
                 n_repeat=3,
                    rand_index=None,
                    verbose=True)

model.fit()

#### DR+K-Means clustring
model.consistency_clustering('Spambase','PCA',KMeans(n_clusters=4),'KM')
print(model.consistency_values)



## Nearest neighbor 

model.consistency_knn('Spambase','PCA')
print(model.consistency_knn_mean)
print(model.aucc)