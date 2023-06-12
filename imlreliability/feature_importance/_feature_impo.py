import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import scale,normalize
import collections
import tensorflow as tf
# tf.enable_eager_execution()
# tf.executing_eagerly()
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
from .rbo import RankingSimilarity
from .util_feature_impo import (internal_resample,clean_score,get_rank,jaccard_similarity)

from deepexplain.tensorflow import DeepExplain


def _consistency(estimator, scores, accuracys, data_name,estimator_name,impotance_func_name=None, Ks=range(1,31,1)):
        
        
        
        method_name=impotance_func_name+'_'+estimator_name if impotance_func_name is not None else estimator_name
 
        print('Importance Function is ',method_name )
           
        ### jaccard/rbo to top ranked features         
        jaccard,RBO = pd.DataFrame(columns = ['data','method','criteria','K','Consistency']),pd.DataFrame(columns = ['data','method','criteria','K','Consistency'])
        ranks = [get_rank(s) for s in scores]
        nIter = len(ranks)
        for i in range(nIter):
            relevance_rank = ranks[i]
            for j in range(i+1,nIter):
                compare_rank=ranks[j]

                for K in Ks:
                    x1 = np.where(relevance_rank<=K)[0]
                    x2 = np.where(compare_rank<=K)[0]
                    rk1=sorted(x1, key=lambda d: relevance_rank[d])
                    rk2=sorted(x2, key=lambda d: compare_rank[d])

                    if len(x1)==0 or len(x2)==0:
                        jaccard.loc[len(jaccard)]=[data_name,method_name,'Jaccard',K,0]
                        RBO.loc[len(RBO)]=[data_name,method_name,'RBO',K,0]
                    else:
     
                        jaccard.loc[len(jaccard)]=[data_name,method_name,'Jaccard',K,jaccard_similarity(x1,x2)]
                        RBO.loc[len(RBO)]=[data_name,method_name,'RBO',K,RankingSimilarity(rk1, rk2).rbo()]
            
            RBO['Consistency'],jaccard['Consistency']=[float(i) for i in RBO['Consistency']],[float(i) for i in jaccard['Consistency']]
            RBO_values = RBO
            jaccard_values=jaccard
            
            RBO_mean = RBO.groupby(['data','method','criteria','K'],as_index=False).mean('Consistency')
            jaccard_mean=jaccard.groupby(['data','method','criteria','K'],as_index=False).mean('Consistency')
            
            accuracy = pd.DataFrame({'data':data_name,
                                          'model':estimator_name,
                                          'Accuracy':accuracys
                                         
                                         })
            results = pd.DataFrame(np.vstack((RBO_mean,jaccard_mean)),columns = RBO_mean.columns)
            results['Accuracy'] = round(np.mean(accuracys),3)
            results['Consistency'] = [round(i,3) for i in results['Consistency']]
            
            
            
            return results,accuracy
        
### prediction consistency by classification purity 
def _pred_consistency_class(test_yhat, 
                            data_name,estimator_name):
    
        v= np.apply_along_axis(_get_entropy_class, 0, test_yhat)
        entropy=pd.DataFrame(v)
        entropy.columns=['Entropy']
        entropy['data']=data_name
        entropy['model']=estimator_name
        entropy['Purity'] = 1-(v-min(v))/max(v)-min(v)
        entropy = entropy.groupby(['data','model']).mean().reset_index()
        return entropy
def _pred_consistency_reg(test_yhat, 
                            data_name,estimator_name):
    
        v= np.apply_along_axis(_get_std_reg, 0, test_yhat)
        entropy=pd.DataFrame(v.mean())
        entropy.columns=['sd']
        entropy['data']=data_name
        entropy['model']=estimator_name
        entropy['Purity']= np.exp(-v)
        entropy = entropy.groupby(['data','model']).mean().reset_index()
        return entropy        
    
def _get_entropy_class(x):
    x = x[x!='NA']
    count=collections.Counter(x)
    entropy=0

    for i in count:
        p =count[i]/len(x)
        entropy += p*np.log(p)
    return -entropy

def _get_std_reg(x):
    x = x[x!='NA']
    x=[float(i) for i in x]
    return np.std(x)        



class feature_impoReg():
    """ 
    Parameters
    ----------
    data: (X, Y)
        X: array of shape (N, M)
        Y: array of shape (N,)
        
    estimator: estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function
        or ``scoring`` must be passed.
        
        
    importance_func: str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see:ref:`importance_parameter`);
        - a callable (see:ref:`importance`) that returns a list of values.
        
    evaluate_fun: str, callable, list, tuple or dict, default=mean_squared_error
        Function to evaluate prediction performance.
        
    K_max: int 
        The number of top features of interest.

    n_repeat: int, default=100
        The number of repeats to measure consistency (run in parallel).

    split_proportion: float in (0,1), default=0.7.
        The proportion of training set in data splitting. 
    
    get_prediction_consistency: {True,False}
        Controls whether to calculate prediction consistency, default = True.
     
    norm: {True,False}
        Controls whether to conduct data normalization, default = True.
    
    rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.
    
    verbose: {True,False}
        Controls the verbosity, default = True.

    Returns
    ----------
    
    prediction_score: array of shape (n_repeat, )
        Prediction error
            
    importance_score: array of shape (n_repeat, M)
        Importance scores of each feature in each repeat 
    
    accuracy: pandas dataframe of shape (n_repeat,3), columns = [data,model,Accuracy]    
        IML model prediction accuracy of each repeat.
            data: name of data set
            model:  the ML prediction model.  
            Accuracy: prediction accuracy scores of each repeat.
    
    consistency: pandas data frame of shape (n_repeat,6), columns = [data, method, criteria,K, Consistency,Accuracy] 
        IML model interpretation consistency and prediction accuracy 
            data: name of the data set
            method: the IML methods. 
            criteria: consistency metrics. 
            K: number of top features.
            Consistency: average pairwise consistency scores. 
            Accuracy: average prediction accuracy scores. 
 
 
        Can be saved and uploaded to the dashboard. 
    
    prediction_consistency: pandas dataframe of shape (n_repeat,4), columns = [data,model, Entropy, Purity] 
        IML model prediction consistency score
        
            data: name of data set
            model: an ML prediction model.  
            Entropy: prediction entropy, higher entropy indicates lower consistency.
            Purity: consistency score converted from entropy, ranges in [0,1] and higher purity indicates higher consistency.
    """
    def __init__(self,data,
                 estimator,
                 importance_func=None,
                 evaluate_fun=mean_squared_error,
                 K_max = 30,
                 n_repeat=100,
                 split_proportion=0.7,
                 get_prediction_consistency=True,
                 norm=True,
                rand_index=None,
                verbose=True):
        self.estimator=estimator
        self.importance_func=importance_func        
        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.verbose=verbose
        self.norm=norm
        self.data=data
        (self.X,self.Y) = self.data        
        self.M=len(self.X[0])
        self.n_repeat=n_repeat

        self.K_max=min(K_max+1,len(self.X[0])+1)
        self.rand_index=rand_index
        self.get_prediction_consistency=get_prediction_consistency



    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
        self.test_yhat=[]
        (X,Y) = self.data
        for i in range(self.n_repeat):
            print(i)
            if self.verbose==True:
                print('Iter: ',i)
            x_train, x_test, y_train, y_test,indices_train,indices_test =  internal_resample(X,Y,
                                   random_index=i*self.rand_index,
                                   proportion=self.split_proportion)
            
            ## standardize
            if self.norm == True:
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))
                y_train=(scale(y_train))
                y_test =(scale(y_test))

            

            
            self.fitted = self.estimator.fit(x_train,y_train)
            s=self._impo_score(x_train,y_train)
            this_yhat = self.fitted.predict(x_test)

            acc = self.evaluate_fun(this_yhat,y_test)

            self.scores.append(s)
            self.accuracys.append(acc)   
            if self.get_prediction_consistency ==True:
                this_pred = list(np.repeat('NA',len(X)))
                for a,item in enumerate(indices_test):
                    this_pred[item] = this_yhat[a]    

                self.test_yhat.append(this_pred)
                
    def _impo_score(self,
           x,
           y):
        yhat = self.fitted.predict
            
            
        x=np.array(x, dtype=float) ## shap has error
        scoring=self.evaluate_fun
        packs = self.fitted.__module__.split('.')

        if not self.importance_func:
            if np.isin('linear_model',packs) or np.isin('svm',packs):
                print('use coefs as feature importance ')
                s = np.abs(self.fitted.coef_)
            elif np.isin('ensemble',packs) or np.isin('xgboost',packs) or np.isin('tree',packs):
                print('use feature_importances_ as feature importance ')
                s = self.fitted.feature_importances_
        else:
            impo_pack = self.importance_func.__module__.split('.')
        #### importance function from shap functions

            if np.isin('shap',impo_pack):
                if np.isin('_permutation',impo_pack):
                    
                    
                    explainer =self.importance_func(yhat,x)
#                     return explainer,x_test,yhat
                 
                elif np.isin('_linear',impo_pack):
                    explainer =self.importance_func(self.fitted,x)
                else:
                #if np.isin('_tree',impo_pack):
                    explainer =self.importance_func(self.fitted,check_additivity=False)
                s = explainer(x)[0].values.T.tolist()

            
                
#                 fi = [] ## randomly choose 100 observations  
#                 r = np.random.RandomState()
#                 idx_I = np.sort(r.choice(len(x_test), size=max(100,len(x_test)), replace=False)) # uniform sampling of subset of observations
#                 for idd in idx_I:
#                     exp = explainer.explain_instance(x_test[idd], yhat, num_features=self.M)
#                     mapp = exp.as_map()
#                     fi.append([a[1] for a in sorted(mapp[list(mapp.keys())[0]])])
#                 s=np.array(fi).mean(0)


            elif np.isin('_permutation_importance',impo_pack):
                s = self.importance_func(self.fitted,x, y).importances_mean
        #### use user-defined importance function
        
            else:
                s = self.importance_func(self.fitted,x, y)
        
        return clean_score(s)

    def get_consistency(self,data_name,estimator_name,impotance_func_name=None):
        
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name,range(1,self.K_max,1))
        

        if self.get_prediction_consistency ==True:
            self.prediction_consistency = _pred_consistency_class(self.test_yhat, 
                            data_name,estimator_name)        
        
        
class feature_impoClass():
    """ 
    Parameters
    ----------
    data: (X, Y)
        X: array of shape (N,M)
        Y: array of shape (N,)
        
    estimator: estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function
        or ``scoring`` must be passed.
        
        
    importance_func: str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see:ref:`importance_parameter`);
        - a callable (see:ref:`importance`) that returns a list of values.
        
    evaluate_fun: str, callable, list, tuple or dict, default=mean_squared_error
        Function to evaluate prediction performance.
        
    K_max: int
        The number of top features of interest, default = 30.

    n_repeat: int
        The number of repeats to measure consistency (run in parallel), default=100

    split_proportion: float in (0,1). 
        The proportion of training set in data splitting, default=0.7.
    
    get_prediction_consistency: {True,False}
        Controls whether to calculate prediction consistency, default=True.
    
    norm: {True,False}
        Controls whether to conduct data normalization, default=True. 
    
    rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.
    
    verbose: {True,False}
        Controls the verbosity, default=True.

    Returns
    ----------
    
    prediction_score: array of shape (n_repeat, )
        Prediction error
            
    importance_score: array of shape (n_repeat, M)
        Importance scores of each feature in each repeat 
    
    accuracy: pandas dataframe of shape (n_repeat,3), columns = [data,model,Accuracy]    
        IML model prediction accuracy of each repeat.
            data: name of data set
            model: ML prediction model.  
            Accuracy: prediction accuracy scores of each repeat.
    
    consistency: pandas data frame of shape (n_repeat,6), columns = [data, method, criteria, K, Consistency, Accuracy] 
        IML model interpretation consistency and prediction accuracy 
            data: name of the data set
            method: IML methods. 
            criteria: consistency metrics. 
            K: the number of top features.
            Consistency: average pairwise consistency scores. 
            Accuracy: average prediction accuracy scores. 
 
 
        Can be saved and uploaded to the dashboard. 
    
    prediction_consistency: pandas dataframe of shape (n_repeat,4), columns = [data,model, Entropy, Purity] 
        IML model prediction consistency score
        
            data: name of data set
            model: the ML prediction model.  
            Entropy: prediction entropy, higher entropy indicates lower consistency.
            Purity: consistency score converted from entropy, ranges in [0,1], and higher purity indicates the higher consistency.
    """

    def __init__(self,data,estimator,
                 importance_func=None,
                 evaluate_fun=accuracy_score,
                K_max = 30,
                 n_repeat=100,
                 split_proportion=0.7,
                get_prediction_consistency=True,
                 norm=True,
                 rand_index=None,
                verbose=True):
 
        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.norm=norm
        self.data=data
        (self.X,self.Y) = self.data
        self.M=len(self.X[0])
        self.K_max=min(K_max+1,len(self.X[0])+1)
        
        self.estimator=estimator
        self.importance_func=importance_func
        self.rand_index=rand_index
        self.get_prediction_consistency=get_prediction_consistency
    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
        self.test_yhat = []
        (X,Y) = self.data
        num_class=len(set(Y))
        
        
        
        for i in range(self.n_repeat):
            print(i)
            if self.verbose==True:
                print('Iter: ',i)
            x_train, x_test, y_train, y_test,indices_train,indices_test = internal_resample(
                                   X,Y,
                                   random_index=i*self.rand_index,
                                   proportion=self.split_proportion,stratify=True)
            if self.norm==True:
                
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))
            

            
            self.fitted = self.estimator.fit(x_train,y_train)
            s=self._impo_score(x_train,y_train)
            this_yhat = self.fitted.predict(x_test)
            acc = self.evaluate_fun(this_yhat,y_test)
                
            self.scores.append(s)
            self.accuracys.append(acc)    
            if self.get_prediction_consistency ==True:
                this_pred = list(np.repeat('NA',len(X)))
                for a,item in enumerate(indices_test):
                    this_pred[item] = this_yhat[a]
                self.test_yhat.append(this_pred)
    def _impo_score(self,
           x,y):
        try:
            yhat = self.fitted.predict_proba
        except:
            yhat = self.fitted.predict
            
            
        x=np.array(x, dtype=float)
        scoring=self.evaluate_fun
        packs = self.fitted.__module__.split('.')

        if not self.importance_func:
            if np.isin('linear_model',packs) or np.isin('svm',packs):
                print('use coefs as feature importance ')
                s = np.abs(self.fitted.coef_)
            elif np.isin('ensemble',packs) or np.isin('xgboost',packs) or np.isin('tree',packs):
                print('use feature_importances_ as feature importance ')
                s = self.fitted.feature_importances_
        else:
        #### use user-defined importance function
            impo_pack = self.importance_func.__module__.split('.')

            if np.isin('shap',impo_pack):
                if np.isin('_permutation',impo_pack):
                    
                    
                    explainer =self.importance_func(yhat,x)
                 
                elif np.isin('_linear',impo_pack):
                    explainer =self.importance_func(self.fitted,x)
                else:
                #if np.isin('_tree',impo_pack):
                    explainer =self.importance_func(self.fitted,check_additivity=False)
                try:
                    s = explainer(x)[0].values.T.tolist()

                except:
                    s = explainer(x,check_additivity=False)[0].values.T.tolist()
#             elif np.isin('lime',impo_pack):    
#                 explainer =self.importance_func(x_train,
#                             class_names=list(set(y_train)),
#                            discretize_continuous=True)

                
#                 fi = [] ## randomly choose 100 observations  
#                 r = np.random.RandomState()
#                 idx_I = np.sort(r.choice(len(x_test), size=max(100,len(x_test)), replace=False)) # uniform sampling of subset of observations
#                 for idd in idx_I:
#                     exp = explainer.explain_instance(x_test[idd], yhat, num_features=self.M)
#                     mapp = exp.as_map()
#                     fi.append([a[1] for a in sorted(mapp[list(mapp.keys())[0]])])
#                 s=np.array(fi).mean(0)


            elif np.isin('_permutation_importance',impo_pack):
                s = self.importance_func(self.fitted,x, y).importances_mean
        
            else:
                s = self.importance_func(self.fitted,x, y)
        
        return clean_score(s)

    def get_consistency(self,data_name,estimator_name,impotance_func_name=None):
        
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, range(1,self.K_max,1))

        if self.get_prediction_consistency ==True:
            self.prediction_consistency = _pred_consistency_class(self.test_yhat, 
                            data_name,estimator_name)

class feature_impoReg_MLP():
    """ 
    Parameters
    ----------
    data: (X, Y)
        X: array of shape (N, M)
        Y: array of shape (N,)
        
    estimator: estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function
        or ``scoring`` must be passed.
        
        
    importance_func: str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see:ref:`importance_parameter`);
        - a callable (see:ref:`importance`) that returns a list of values.
        
    evaluate_fun: str, callable, list, tuple or dict, default=mean_squared_error
        Function to evaluate prediction performance.
        
    K_max: int 
        The number of top features of interest, default=30.

    n_repeat: int 
        The number of repeats to measure consistency (run in parallel), default=100.

    split_proportion: float in (0,1). 
        The proportion of training set in data splitting. 
    
    get_prediction_consistency: {True,False}
        Controls whether to calculate prediction consistency, default=True.
    

    norm: {True,False}
        Controls whether to conduct data normalization, default=True.
    
    rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.
    
    verbose: {True,False}
        Controls the verbosity, default=True.

    Returns
    ----------
    
    prediction_score: array of shape (n_repeat, )
        Prediction error
            
    importance_score: array of shape (n_repeat, M)
        Importance scores of each feature in each repeat 
    
    accuracy: pandas dataframe of shape (n_repeat,3), columns = [data,model,Accuracy]    
        IML model prediction accuracy of each repeat.
            data: name of data set
            model: ML prediction model.  
            Accuracy: prediction accuracy scores of each repeat.
    
    consistency: pandas data frame of shape (n_repeat,6), columns = [data, method, criteria,K, Consistency,Accuracy] 
        IML model interpretation consistency and prediction accuracy 
            data: name of the data set
            method: IML methods. 
            criteria: consistency metrics. 
            K: number of top features.
            Consistency: average pairwise consistency scores. 
            Accuracy: average prediction accuracy scores. 
 
 
        Can be saved and uploaded to the dashboard. 
    
    prediction_consistency: pandas dataframe of shape (n_repeat,4), columns = [data,model, Entropy    Purity] 
        IML model prediction consistency score
        
            data: name of data set
            model: ML prediction model.  
            Entropy: prediction entropy, higher entropy indicates lower consistency.
            Purity: consistency score converted from entropy, ranges in [0,1] and higher purity indicates higher consistency.
    """
    
   
    def __init__(self,data,importance_func, 
                 estimator=None,
                 evaluate_fun=accuracy_score,
                 K_max = 30,
                 n_repeat=50,
                 split_proportion=0.7,
                 get_prediction_consistency=True,
                 norm=True,
                 rand_index=None,
                 verbose=True):

        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.verbose=verbose
        self.norm=norm
        self.n_repeat=n_repeat
        self.importance_func=importance_func        
        self.data=data
        (self.X,self.Y) = self.data
        self.M=len(self.X[0])

        self.K_max=min(K_max+1,len(self.X[0])+1)
        self.get_prediction_consistency=get_prediction_consistency
        self.rand_index=rand_index
        if not estimator: ## default regression model 
            self.estimator = self._base_model_regression()
            self.target_layer=-1
        else:
            self.estimator=estimator
        
    def _base_model_regression(self):
        model = Sequential()
        model.add(Dense(self.M, input_dim=self.M, activation='relu'))
        model.add(Dense(self.M, input_dim=self.M, activation='relu'))    
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model   


        
            
    def _impo_score(self,
           x,y):
        
        de_methods = [
                        'zero',
                        'saliency',
                        'grad*input',
                        'intgrad',
                        'elrp',
                        'deeplift',
                        'occlusion',
                        'shapley_sampling']
        
        yhat =  self.estimator.predict
        

        #### use user-defined importance function
    
        if hasattr(self.importance_func.__class__, '__call__') and self.importance_func.__class__!=str:
        ## if importance_func is a function
            impo_pack = self.importance_func.__module__.split('.')
            print(impo_pack)
            if np.isin('permutation_importance',impo_pack):
                my_model = KerasRegressor(build_fn=self._base_model_regression)
                my_model.fit(x,y,verbose=0)
                perm = self.importance_func(my_model).fit(x,y)
                s=perm.feature_importances_
            else:
                model = load_model("mlp_reg_"+str(self.i)+".h5")
                if np.isin('shap',impo_pack):
                    background = x[np.random.choice(x.shape[0], 100, replace=False)]
                    s = self.importance_func(model, background).shap_values(x)
                else: ##user defined function 
                    s = self.importance_func(model,x, y)
  
        else: ## if input is string 
            if np.isin(self.importance_func,de_methods):        
                model = self.estimator
                print('DeepExplain')
                with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
                        input_tensor = model.layers[0].input
                        fModel = Model(inputs=input_tensor, outputs = model.layers[self.target_layer].output)
                        target_tensor = fModel(input_tensor)

                        xs = x
                        ys = np.reshape(y, (len(y),  1))
                        attributions = de.explain(self.importance_func, target_tensor, input_tensor, xs, ys=ys)

                s=attributions.mean(0)
            else:
                impo_pack = (eval(self.importance_func.split('.')[0] + "()")).__module__.split('.')
                if np.isin('deeplift',impo_pack):
                    print('DeepLift')
                    dl_model = kc.convert_model_from_saved_files(
                                                h5_file=self.saved_model_file,
                                                nonlinear_mxts_mode=(eval(self.importance_func)))

                    dl_func = dl_model.get_target_contribs_func(find_scores_layer_idx=0, 
                                                                target_layer_idx=self.target_layer)
                    method_name, score_func=self.importance_func, dl_func
                    print("Computing scores for:",method_name)
                    method_to_task_to_scores = {}
                    scor = np.array(score_func(
                                    task_idx=0,
                                    input_data_list=[x],
                                    input_references_list=[np.zeros_like(x)],
                                    batch_size=100,
                                    progress_update=None))
                    s=scor.mean(0)

        return clean_score(s)


    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
        self.test_yhat=[]
        (X,Y) = self.data
        for i in range(self.n_repeat):
            self.i=i
            if self.verbose==True:
                print('Iter: ',i)
            x_train, x_test, y_train, y_test,indices_train,indices_test =  internal_resample(X,Y,
                                   random_index=i*self.rand_index,
                                   proportion=self.split_proportion)

            if self.norm==True:
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))

                y_train=(scale(y_train))
                y_test =(scale(y_test))
            self.estimator.fit(x_train,y_train,verbose=0)
            ##########
            model_json = self.estimator.to_json()
            with open("mlp_reg_"+str(i)+".json", "w") as json_file:
                    json_file.write(model_json)

            self.estimator.save("mlp_reg_"+str(i)+".h5",overwrite=True)
            ######################
            self.saved_model_file= "mlp_reg_"+str(i)+".h5"
        
            s=self._impo_score(x_train,y_train)

            
    ##### different in MLP!
            acc = self.estimator.evaluate(x_test, y_test, batch_size=10)
            this_yhat = self.estimator.predict(x_test, batch_size=10)
  
            self.scores.append(s)
            self.accuracys.append(acc)
            if self.get_prediction_consistency ==True:
                this_pred = list(np.repeat('NA',len(X)))
                for a,item in enumerate(indices_test):
                    this_pred[item] = this_yhat[a][0]
                                                                  
                self.test_yhat.append(this_pred)
                    

    def get_consistency(self,data_name,estimator_name,impotance_func_name=None):
       
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, range(1,self.K_max,1))

        if self.get_prediction_consistency ==True:
            self.prediction_consistency = _pred_consistency_class(self.test_yhat, 
                            data_name,estimator_name)

class feature_impoClass_MLP():
    """ 
    Parameters
    ----------
    data: (X, Y)
        X: array of shape (N, M)
        Y: array of shape (N,)
        
    estimator: estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function
        or ``scoring`` must be passed.
        
        
    importance_func: str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see:ref:`importance_parameter`);
        - a callable (see:ref:`importance`) that returns a list of values.
        
    evaluate_fun: str, callable, list, tuple or dict, default=mean_squared_error
        Function to evaluate prediction performance.
        
    K_max: int 
        The number of top features of interest.

    n_repeat: int 
        The number of repeats to measure consistency (run in parallel), default=100.

    split_proportion: float in (0,1). 
        The proportion of training set in data splitting, default=0.7.
    
    get_prediction_consistency: {True,False}
        Controls whether to calculate prediction consistency, default=True.
    

    norm: {True,False}
        Controls whether to conduct data normalization, default=True. 
    
    rand_index: RandomState instance
          Make sure the sampling uses the same RandomState instance for all iterations.
    
    verbose: {True,False}
        Controls the verbosity, default=True.

    Returns
    ----------
    
    prediction_score: array of shape (n_repeat, )
        Prediction error
            
    importance_score: array of shape (n_repeat, M)
        Importance scores of each feature in each repeat 
    
    accuracy: pandas dataframe of shape (n_repeat,3), columns = [data,model,Accuracy]    
        IML model prediction accuracy of each repeat.
            data: name of data set
            model: ML prediction model.  
            Accuracy: prediction accuracy scores of each repeat.
    
    consistency: pandas data frame of shape (n_repeat,6), columns = [data, method, criteria,K, Consistency, Accuracy] 
        IML model interpretation consistency and prediction accuracy 
            data: name of the data set
            method: IML methods. 
            criteria: consistency metrics. 
            K: number of top features.
            Consistency: average pairwise consistency scores. 
            Accuracy: average prediction accuracy scores. 
 
 
        Can be saved and uploaded to the dashboard. 
    
    prediction_consistency: pandas dataframe of shape (n_repeat,4), columns = [data,model, Entropy, Purity] 
        IML model prediction consistency score
        
            data: name of data set
            model: ML prediction model.  
            Entropy: prediction entropy, higher entropy indicates lower consistency.
            Purity: consistency score converted from entropy, ranges in [0,1] and higher purity indicates higher consistency.
    """
    
    def __init__(self,data,importance_func, 
                 estimator=None,
                 evaluate_fun=accuracy_score,
                 K_max = 30,
                 n_repeat=100,
                 split_proportion=0.7,
                 get_prediction_consistency=True,
                 norm=True,
                 rand_index=None,
                 verbose=True):

        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.norm=norm
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.importance_func=importance_func        
        self.data=data
        (self.X,self.Y) = self.data
        self.K_max=min(K_max+1,len(self.X[0])+1)
        
        self.get_prediction_consistency=get_prediction_consistency    
        self.M=len(self.X[0])
        self.num_class=len(set(self.Y))
        self.rand_index=rand_index
        self.Y = pd.get_dummies(self.Y)
        if not estimator: ## default classification
            self.estimator = self._base_model_classification()
            self.target_layer=-2

        else:
            self.estimator=estimator
        
        
        print(estimator)
        
    def _base_model_classification(self):
        model = Sequential()
        model.add(Dense(self.M, input_dim=self.M, activation='relu'))
        model.add(Dense(self.M, input_dim=self.M, activation='relu'))
        model.add(Dense(self.num_class))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
        return model   


        
            
    def _impo_score(self,
           x,y):
        s=None
        de_methods = [
                        'zero',
                        'saliency',
                        'grad*input',
                        'intgrad',
                        'elrp',
                        'deeplift',
                        'occlusion',
                        'shapley_sampling']
        
        yhat = self.estimator.predict_proba

        #### use user-defined importance function
    
        if hasattr(self.importance_func.__class__, '__call__') and self.importance_func.__class__!=str:
        ## if importance_func is a function
            impo_pack = self.importance_func.__module__.split('.')
            print(impo_pack)
            if np.isin('permutation_importance',impo_pack):
                ### need _base_model_classification instead of _base_model_classification()
                my_model = KerasClassifier(build_fn=self._base_model_classification)
                my_model.fit(x,y,verbose=0)
                perm = self.importance_func(my_model).fit(x,y)
                s=perm.feature_importances_
            
            

            else:
                ###### Loac MLP model
                model = load_model("mlp_class_"+str(self.i)+".h5")
                if np.isin('shap',impo_pack):
                    background = x[np.random.choice(x.shape[0], 100, replace=False)]
                    s = self.importance_func(model, background).shap_values(x)
                else: 
                    ##user defined function 
                    s = self.importance_func(model,x,y) 
                                   
        else: ## if input is string 
            if np.isin(self.importance_func,de_methods):  
                model = self.estimator
                print('DeepExplain')
                with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
                        input_tensor = model.layers[0].input
                        fModel = Model(inputs=input_tensor, outputs = model.layers[self.target_layer].output)
                        target_tensor = fModel(input_tensor)

                        xs = x
                        ys=np.array(pd.get_dummies(y))
                        attributions = de.explain(self.importance_func, target_tensor, input_tensor,xs,ys=ys)

                s=attributions.mean(0)
            else:
                impo_pack = (eval(self.importance_func.split('.')[0] + "()")).__module__.split('.')
                if np.isin('deeplift',impo_pack):
                    print('DeepLift')

                    dl_model = kc.convert_model_from_saved_files(
                                                h5_file=self.saved_model_file,
                                                nonlinear_mxts_mode=(eval(self.importance_func)))

                    dl_func = dl_model.get_target_contribs_func(find_scores_layer_idx=0, 
                                                                target_layer_idx=self.target_layer)
                    method_name, score_func=self.importance_func, dl_func
                    print("Computing scores for:",method_name)
                    method_to_task_to_scores = {}
                    scor = np.array(score_func(
                                    task_idx=0,
                                    input_data_list=[x],
                                    input_references_list=[np.zeros_like(x)],
                                    batch_size=100,
                                    progress_update=None))
                    s=scor.mean(0)   
#                 except:
#                     print('Invalid feature importance function')
        return clean_score(s)
        

    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
        self.test_yhat=[]
        (X,Y) = self.data

        for i in range(self.n_repeat):
            self.i=i
            if self.verbose==True:
                print('Iter: ',i)
           
            x_train, x_test, y_train, y_test,indices_train,indices_test =  internal_resample(X,Y,
                                   random_index=i*self.rand_index,
                                   proportion=self.split_proportion,stratify=True)


            ## standardize 
            if self.norm==True:                
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))

            ################
            #### Need to use hot-encodede verion of response 
            yy_train,yy_test = (pd.get_dummies(y_train)),(pd.get_dummies(y_test))
            yy_test =np.array(yy_test.reindex(columns = yy_train.columns, fill_value=0))
            yy_train = np.array(yy_train)
            try:
                self.estimator.fit(x_train,y_train,verbose=0)
            except:
                self.estimator.fit(x_train,yy_train,verbose=0)
            if  self.importance_func .__class__!=str and np.isin('permutation_importance',self.importance_func.__module__.split('.')):
                s = self._impo_score(x,y)
            else:
                
                ##########
                model_json = self.estimator.to_json()
                with open("mlp_class_"+str(i)+".json", "w") as json_file:
                        json_file.write(model_json)

                    # serialize weights to HDF5
                self.estimator.save("mlp_class_"+str(i)+".h5")
                ######################
                self.saved_model_file= "mlp_class_"+str(i)+".h5"

                s=self._impo_score(x_train,y_train)

                
        ##### different in MLP!
            try:
                acc = self.estimator.evaluate(x_test, y_test, batch_size=10)
            except:
                acc = self.estimator.evaluate(x_test, yy_test, batch_size=10)

            self.scores.append(s)
            self.accuracys.append(acc)
            if self.get_prediction_consistency ==True:
                this_yhat = self.estimator.predict(x_test, batch_size=10)
                this_yhat = [np.argmax(a) for a in this_yhat]
                this_pred = list(np.repeat('NA',len(X)))
                for a,item in enumerate(indices_test):
                    this_pred[item] = this_yhat[a]
                self.test_yhat.append(this_pred)
                
                
    def get_consistency(self,data_name,estimator_name,impotance_func_name=None):
        
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, range(1,self.K_max,1))

        if self.get_prediction_consistency ==True:
            self.prediction_consistency = _pred_consistency_class(self.test_yhat, 
                            data_name,estimator_name) 
                    
