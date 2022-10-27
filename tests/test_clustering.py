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
#### KMeans
########################
esti_km = KMeans(n_clusters=K,init='k-means++')



model_km = clustering(x,estimator=esti,K=len(set(y)),
                 label=y,
                 sigma=1,
                 noise_type='normal',
                 n_repeat=10,
                    rand_index=None,
                    verbose=True)

model_km.fit()
model_km.consistency(dd,method_name='K-means')
print(model_km.results)
#######################
#### HC
########################
from sklearn.cluster import AgglomerativeClustering
esti_hc = AgglomerativeClustering(n_clusters=K,linkage='average',affinity='euclidean')


model_hc = clustering(x,estimator=esti_hc,K=len(set(y)),
                 label=y,
                 sigma=1,
                 noise_type='normal',
                 n_repeat=10,
                    rand_index=None,
                    verbose=True)

model_hc.fit()
model_hc.consistency(dd,method_name='HC (average)')
print(model_hc.results)

