import IMLconsistency

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression



#####################
## import bike data 
########################
data = pd.read_csv('/Users/melinda/Downloads/Bike-Sharing-Dataset/day.csv')
data = data.drop(columns=['instant','dteday'])
y = np.array(data['cnt'])
x= np.array(data.drop(columns='cnt'))



##############################
# model specific IML method
#############################
#######################
#### linear regression 
########################

estimator=RidgeCV()
importance_func=None



model_reg=feature_impoReg(data_reg,estimator,sigma=1,
                 importance_func=importance_func,
                 evaluate_fun=mean_squared_error,
                 n_repeat=3,split_proportion=0.7,
                rand_index=None)
model_reg.fit()
model_reg.consistency(data_name='new_reg', estimator_name='ridge',impotance_func_name=impo_name)
print(model_reg.accuracy)
print(model_reg.consistency)



#######################
#### tree-based regression 
########################

estimator_tree=RandomForestRegressor()
importance_func=None

model_reg_tree=feature_impoReg(data_reg,estimator_tree,sigma=1,
                 importance_func=importance_func,
                 evaluate_fun=mean_squared_error,
                 n_repeat=3,split_proportion=0.7,
                rand_index=None)
model_reg_tree.fit()
model_reg_tree.consistency(data_name='bike', estimator_name='RF,impotance_func_name=impo_name)
print(model_reg_tree.accuracy)
print(model_reg_tree.consistency)



#######################
#### model agnostic regression 
########################

estimator_tree=RandomForestRegressor()
importance_func = permutation_importance
## TreeExplainer
# importance_func = PermutationExplainer 


model_reg_per=feature_impoReg(data_reg,estimator_tree,sigma=1,
                 importance_func=importance_func,
                 evaluate_fun=mean_squared_error,
                 n_repeat=3,split_proportion=0.7,
                rand_index=None)
model_reg_per.fit()
model_reg_per.consistency(data_name='bike', estimator_name='RF (permutation)',impotance_func_name=impo_name)
print(model_reg_per.accuracy)
print(model_reg_per.consistency)














