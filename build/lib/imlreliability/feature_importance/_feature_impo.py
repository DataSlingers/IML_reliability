import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import is_classifier, is_regressor
from sklearn import preprocessing

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow import keras
# from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from .rbo import RankingSimilarity
from .util_feature_impo import (internal_resample,clean_score,get_rank,jaccard_similarity)



def _consistency(estimator, scores, accuracys,data_name,estimator_name,impotance_func_name=None, Ks=range(1,31,1)):
        
        
        
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
                                          'type':estimator_name,
#                                           'nosie':noise_type,
                                          'Accuracy':accuracys
                                         
                                         })
            results = pd.DataFrame(np.vstack((RBO_mean,jaccard_mean)),columns = RBO_mean.columns)
            results['Accuracy'] = round(np.mean(accuracys),3)
            results['Consistency'] = [round(i,3) for i in results['Consistency']]
            return results,accuracy 
        

class feature_impoReg():
    """ 
    Parameters
    ----------
    data: 
        (X,Y)
        X: N*M numpy array
        Y: N*1 numpy array 
        
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
        
        
    self.importance_func : str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see :ref:`importance_parameter`);
        - a callable (see :ref:`importance`) that returns a list of values.
        
            
    split_proportion: float in (0,1). need to specify if noise_type=='split'
    
    sigma: float, level of noise. need to specify if noise_type!='split'
    
    norm: binary. conduct data normalization 
    
    n_repeat: int, default=100
        Number of repeats to measure consistency (run in parallel).
    
    
    
    Attributes
    ----------
    
    prediction_score: mse or accuracy.. 
        a list with length = n_repeat
    
    importance_score: 
        a n_repeat*M matrix
    
    """
    def __init__(self,data,importance_func,
                 estimator=None,
                 sigma=None,
                 evaluate_fun=accuracy_score,
                 n_repeat=50,
                 split_proportion=0.7,
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
        if not estimator: ## default regression model 
            self.estimator = self._base_model_regression()
            self.target_layer=-1
        else:
            self.estimator=estimator



    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
        (X,Y) = self.data
        self.M=len(X[0])

        for i in range(self.n_repeat):
            print(i)
            if self.verbose==True:
                print('Iter: ',i)
           
            x_train, x_test, y_train, y_test = internal_resample(
                                   data=(X,Y),
                                   random_index=i*4,
                                   proportion=self.split_proportion)
            ## standardize
            if norm == True:
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))
                y_train=normalize(scale(y_train))
                y_test =normalize(scale(y_test))

            

            
            self.fitted = self.estimator.fit(x_train,y_train)
            s=self._impo_score(x_train,y_train,x_test)
            acc = self.evaluate_fun(self.fitted.predict(x_test),y_test)

            self.scores.append(s)
            self.accuracys.append(acc)            
    def _impo_score(self,
           x_train,
           y_train,
           x_test):
        yhat = self.fitted.predict
            
            
        x_train,x_test=np.array(x_train, dtype=float),np.array(x_test, dtype=float) ## shap has error
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
                    
                    
                    explainer =self.importance_func(yhat,x_train)
#                     return explainer,x_test,yhat
                 
                elif np.isin('_linear',impo_pack):
                    explainer =self.importance_func(self.fitted,x_train)
                else:
                #if np.isin('_tree',impo_pack):
                    explainer =self.importance_func(self.fitted,check_additivity=False)
                s = explainer(x_test)[0].values.T.tolist()

            
                
                fi = [] ## randomly choose 100 observations  
                r = np.random.RandomState()
                idx_I = np.sort(r.choice(len(x_test), size=max(100,len(x_test)), replace=False)) # uniform sampling of subset of observations
                for idd in idx_I:
                    exp = explainer.explain_instance(x_test[idd], yhat, num_features=self.M)
                    mapp = exp.as_map()
                    fi.append([a[1] for a in sorted(mapp[list(mapp.keys())[0]])])
                s=np.array(fi).mean(0)


            elif np.isin('_permutation_importance',impo_pack):
                s = self.importance_func(self.fitted,x_train, y_train).importances_mean
        
            else:
                s = self.importance_func(self.fitted,x_train, y_train)
        
        return clean_score(s)

    def consistency(self,data_name,estimator_name,impotance_func_name=None, Ks=range(1,31,1)):
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, Ks)
        
        
        
        
        
class feature_impoClass():
    """ 
    Parameters
    ----------
    data: 
        (X,Y)
        X: N*M numpy array
        Y: N*1 numpy array 
        
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
        
        
    importance_func : str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see :ref:`importance_parameter`);
        - a callable (see :ref:`importance`) that returns a list of values.
        
            
    split_proportion: float in (0,1). need to specify if randomn=='split'
    
    sigma: float, level of noise. need to specify if randomn!='split'
    
       norm: binary. conduct data normalization 
  
    n_repeat: int, default=100
        Number of repeats to measure consistency (run in parallel).
    
    
    
    Attributes
    ----------
    
    prediction_score: mse or accuracy.. 
        a list with length = n_repeat
    
    importance_score: 
        a n_repeat*M matrix

    
    """
    def __init__(self,data,estimator,
                 sigma=None,
                 importance_func=None,
                 evaluate_fun=accuracy_score,
                 n_repeat=100,
                 split_proportion=0.7,
                 norm=True,
               rand_index=None,
                verbose=True):
 
        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.norm=norm
        self.data=data
        self.estimator=estimator
        self.importance_func=importance_func


    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
        (X,Y) = self.data
        self.M=len(X[0])
        num_class=len(set(Y))
        print(num_class)

        for i in range(self.n_repeat):
            print(i)
            if self.verbose==True:
                print('Iter: ',i)
           
            x_train, x_test, y_train, y_test = internal_resample(
                                   data=(X,Y),
                                   random_index=i*4,
                                   proportion=self.split_proportion,stratify=True)
            if norm==True:
                
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))
            

            
            self.fitted = self.estimator.fit(x_train,y_train)
            s=self._impo_score(x_train,y_train,x_test)
            acc = self.evaluate_fun(self.fitted.predict(x_test),y_test)

            self.scores.append(s)
            self.accuracys.append(acc)            
    def _impo_score(self,
           x_train,
           y_train,
           x_test):
        try:
            yhat = self.fitted.predict_proba
        except:
            yhat = self.fitted.predict
            
            
        x_train,x_test=np.array(x_train, dtype=float),np.array(x_test, dtype=float) ## shap has error
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
                    
                    
                    explainer =self.importance_func(yhat,x_train)
#                     return explainer,x_test,yhat
                 
                elif np.isin('_linear',impo_pack):
                    explainer =self.importance_func(self.fitted,x_train)
                else:
                #if np.isin('_tree',impo_pack):
                    explainer =self.importance_func(self.fitted,check_additivity=False)
                try:
                    s = explainer(x_test)[0].values.T.tolist()

                except:
                    s = explainer(x_test,check_additivity=False)[0].values.T.tolist()
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
                s = self.importance_func(self.fitted,x_train, y_train).importances_mean
        
            else:
                s = self.importance_func(self.fitted,x_train, y_train)
        
        return clean_score(s)

    def consistency(self,data_name,estimator_name,impotance_func_name=None, Ks=range(1,31,1)):
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, Ks)
        
        
        


class feature_impoReg_MLP():
    """ 
    Parameters
    ----------
    data: 
        (X,Y)
        X: N*M numpy array
        Y: N*1 numpy array 
        
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
        
        
    self.importance_func : str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see :ref:`importance_parameter`);
        - a callable (see :ref:`importance`) that returns a list of values.
        
            
    split_proportion: float in (0,1). need to specify if noise_type=='split'
    
    sigma: float, level of noise. need to specify if noise_type!='split'
    
    
    n_repeat: int, default=100
        Number of repeats to measure consistency (run in parallel).
    
    
    
    Attributes
    ----------
    
    prediction_score: mse or accuracy.. 
        a list with length = n_repeat
    
    importance_score: 
        a n_repeat*M matrix

    
    """
    def __init__(self,data,importance_func, 
                 estimator=None,
                 sigma=None,
                 evaluate_fun=accuracy_score,
                 n_repeat=50,
                 split_proportion=0.7,
                 norm=True,
                rand_index=None,
                verbose=True):
        print(estimator)

        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.verbose=verbose
        self.norm=norm
        self.n_repeat=n_repeat
        self.importance_func=importance_func        
        self.data=data
        (self.X,self.Y) = self.data
        self.M=len(self.X[0])
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
           x_train,
           y_train,
           x_test):
        
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
            if np.isin('permutation_importance',impo_pack):
                perm = self.importance_func(self.estimator).fit(x_train,y_train)
                s=perm.feature_importances_
                print(s)
            else:
                model = load_model("mlp_"+str(self.i)+".h5")
                if np.isin('shap',impo_pack):
                    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
                    s = self.importance_func(model, background).shap_values(x_test)
                else: ##user defined function 
                    s = self.importance_func(model,x_train, y_train)
  
        else: ## if input is string 
            if np.isin(self.importance_func,de_methods):        
                model = load_model(self.saved_model_file,compile=False)
                print('DeepExplain')
                with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
                        input_tensor = model.layers[0].input
                        fModel = Model(inputs=input_tensor, outputs = model.layers[self.target_layer].output)
                        target_tensor = fModel(input_tensor)

                        xs = x_test
                        ys = np.reshape(y_test, (len(y_test),  1))
                        attributions = de.explain(self.importance_func, target_tensor, input_tensor, xs, ys=ys)

                s=attributions.mean(0)
            else:
                impo_pack = (eval(self.importance_func.split('.')[0] + "()")).__module__.split('.')
                if np.isin('deeplift',impo_pack):
                    print('DeepLift')
                    dl_model = kc.convert_model_from_saved_files(
                                                h5_file=self.saved_model_file,
                                                nonlinear_mxts_mode=eval(self.importance_func))

                    dl_func = dl_model.get_target_contribs_func(find_scores_layer_idx=0, 
                                                                target_layer_idx=self.target_layer)
                    method_name, score_func=self.importance_func, dl_func
                    print("Computing scores for:",method_name)
                    method_to_task_to_scores = {}
                    scor = np.array(score_func(
                                    task_idx=0,
                                    input_data_list=[x_test],
                                    input_references_list=[np.zeros_like(x_test)],
                                    batch_size=100,
                                    progress_update=None))
                    s=scor.mean(0)

        return clean_score(s)


    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
  
        
        for i in range(self.n_repeat):
            self.i=i
            if self.verbose==True:
                print('Iter: ',i)
           
            (x_train, x_test, y_train, y_test)= _internal_resample(
                                   data=(self.X,self.Y),
                                   random_index=i*4,
                                   proportion=self.split_proportion)
            ## standardize 
            if norm==True:                
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))

                y_train=normalize(scale(y_train))
                y_test =normalize(scale(y_test))
           
            fitted = self.estimator.fit(x_train,y_train)
            ##########
            model_json = self.estimator.to_json()
            with open("mlp_"+str(i)+".json", "w") as json_file:
                    json_file.write(model_json)

                # serialize weights to HDF5
            self.estimator.save("mlp_"+str(i)+".h5")
            ######################
            self.saved_model_file= "mlp_"+str(i)+".h5"
        
            ss=self._impo_score(x_train,y_train,x_test)

            
    ##### different in MLP!
            acc = self.estimator.evaluate(x_test, y_test, batch_size=10)
            self.scores.append(s)
            self.accuracys.append(acc)

    def consistency(self,data_name,estimator_name,impotance_func_name=None, Ks=range(1,31,1)):
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, Ks)
        


class feature_impoClass_MLP():
    """ 
    Parameters
    ----------
    data: 
        (X,Y)
        X: N*M numpy array
        Y: N*1 numpy array 
        
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
        
        
    self.importance_func : str, callable, list, tuple or dict, default=None
    
        Strategy to evaluate feature importance score.
        If `importance` represents a single score, one can use:
        - a single string (see :ref:`importance_parameter`);
        - a callable (see :ref:`importance`) that returns a list of values.
        
        
    noise_type: str (train/test split or noise addiction)
    
    split_proportion: float in (0,1). need to specify if noise_type=='split'
    
    sigma: float, level of noise. need to specify if noise_type!='split'
    
    
    n_repeat: int, default=100
        Number of repeats to measure consistency (run in parallel).
    
    
    
    Attributes
    ----------
    
    prediction_score: mse or accuracy.. 
        a list with length = n_repeat
    
    importance_score: 
        a n_repeat*M matrix

    
    """
    def __init__(self,data,importance_func, 
                 estimator=None,
                 sigma=None,
                 evaluate_fun=accuracy_score,
                 noise_type='split',
                 n_repeat=50,
                 split_proportion=0.7,
                 norm=True,
                rand_index=None,
                verbose=True):
        print(estimator)

        self.evaluate_fun=evaluate_fun
        self.split_proportion=split_proportion
        self.norm=norm
        self.verbose=verbose
        self.n_repeat=n_repeat
        self.importance_func=importance_func        
        self.data=data
        (self.X,self.Y) = self.data
        self.M=len(self.X[0])
        self.num_class=len(set(self.Y))

        self.Y = pd.get_dummies(self.Y)
        if not estimator: ## default classification
            self.estimator = self._base_model_classification()
            self.target_layer=-2

        else:
            self.estimator=estimator
        
        

        
    def _base_model_classification(self):
        model = Sequential()
        model.add(Dense(self.M, input_dim=self.M, activation='relu'))
        model.add(Dense(self.M, input_dim=self.M, activation='relu'))
        
        model.add(Dense(self.num_class, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
        return model   


        
            
    def _impo_score(self,
           x_train,
           y_train,
           x_test):
        
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
            if np.isin('permutation_importance',impo_pack):
                perm = self.importance_func(self.estimator).fit(x_train,y_train)
                s=perm.feature_importances_
                print(s)
            else:
                model = load_model("mlp_"+str(self.i)+".h5")
                if np.isin('shap',impo_pack):
                    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
                    s = self.importance_func(model, background).shap_values(x_test)
                else: ##user defined function 
                    s = self.importance_func(model,x_train, y_train)
  
        else: ## if input is string 
            if np.isin(self.importance_func,de_methods):        
                model = load_model(self.saved_model_file,compile=False)
                print('DeepExplain')
                with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
                        input_tensor = model.layers[0].input
                        fModel = Model(inputs=input_tensor, outputs = model.layers[self.target_layer].output)
                        target_tensor = fModel(input_tensor)

                        xs = x_test
                        ys = np.reshape(y_test, (len(y_test),  1))
                        attributions = de.explain(self.importance_func, target_tensor, input_tensor, xs, ys=ys)

                s=attributions.mean(0)
            else:
#                 try:
                impo_pack = (eval(self.importance_func.split('.')[0] + "()")).__module__.split('.')
                if np.isin('deeplift',impo_pack):
                    print('DeepLift')
                    dl_model = kc.convert_model_from_saved_files(
                                                h5_file=self.saved_model_file,
                                                nonlinear_mxts_mode=eval(self.importance_func))

                    dl_func = dl_model.get_target_contribs_func(find_scores_layer_idx=0, 
                                                                target_layer_idx=self.target_layer)
                    method_name, score_func=self.importance_func, dl_func
                    print("Computing scores for:",method_name)
                    method_to_task_to_scores = {}
                    scor = np.array(score_func(
                                    task_idx=0,
                                    input_data_list=[x_test],
                                    input_references_list=[np.zeros_like(x_test)],
                                    batch_size=100,
                                    progress_update=None))
                    s=scor.mean(0)
#                 except:
#                     print('Error')
        return clean_score(s)


    def fit(self, *args,**kwargs):
        self.scores= []
        self.accuracys = []
  
        
        for i in range(self.n_repeat):
            self.i=i
            if self.verbose==True:
                print('Iter: ',i)
           
            x_train, x_test, y_train, y_test= _internal_resample(
                                   data=(self.X,self.Y),
                                   random_index=i*4,
                                   proportion=self.split_proportion,stratify=True)

            ## standardize 
            if norm==True:                
                x_train=normalize(scale(x_train))
                x_test =normalize(scale(x_test))

             
            
            fitted = self.estimator.fit(x_train,y_train)
            ##########
            model_json = self.estimator.to_json()
            with open("mlp_"+str(i)+".json", "w") as json_file:
                    json_file.write(model_json)

                # serialize weights to HDF5
            self.estimator.save("mlp_"+str(i)+".h5")
            ######################
            self.saved_model_file= "mlp_"+str(i)+".h5"
        
            ss=self._impo_score(x_train,y_train,x_test)

            
    ##### different in MLP!
            acc = self.estimator.evaluate(x_test, y_test, batch_size=10)
            self.scores.append(s)
            self.accuracys.append(acc)

    def consistency(self,data_name,estimator_name,impotance_func_name=None, Ks=range(1,31,1)):
        self.consistency,self.accuracy =_consistency(self.estimator, self.scores, self.accuracys, data_name,estimator_name,impotance_func_name, Ks)
        
                    
