IML reliability: empirical study 
===
imlreliability provides a unified framework for the reliability test based on repeated data perturbation, as specified in paper... Users are free to implement their own IML methodology, data set or consistency metrics. 
The output can be directly uploaed and compare to our reliability experiments in the [**dashboard**](https://iml-reliability.herokuapp.com).

## Motivation


## IML Tasks 
- Feature importance
    - Regression
    - Classification
- Clustering 
- Dimension Reduction


## Reliability Test


## Criteria 


imlreliability Quickstart
===
## Installation

IML_reliability 0.1 and later require Python 3.7 or Python 3.8. 

Clone this repo and run

    python setup.py install



**The dependencies of this package will be automatically installed into your environment.**

## Usage 

Detailed working examples and tutorials for each task can be found in [tutorial folder](https://github.com/DataSlingers/IML_reliability/tree/main/tutorial) of the repository. Here we give a simple illustration on the usage of the package. To measure interpretation reliability for any of the IML tasks, imlreliability consists of two steps: 

    1. Obtain a set of intepretation by repeatly applying the IML algorithm to perturbed data sets. This step is acheived by initializing the model `model` with:
        - `feature_importance.feature_impoClass()` for feature importance (classification)
        - `feature_importance.feature_impoReg()` for feature importance (regression)    
        - `clustering.clustering()` for clustering 
        - `dimension_reduction.dimension_reduction()` for dimension reduction
    
      and perform the model fitting via `model.fit()`. 
       

```python
    # Pseudo-code 
    ## Load example data 
    from sklearn.preprocessing import scale, normalize
    communities_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data').to_numpy()
    communities_data = np.delete(communities_data, np.arange(5), 1)
# remove predictors with missing values
    communities_data = np.delete(communities_data,
                         np.argwhere((communities_data == '?').sum(0) > 0).reshape(-1), 1)
    communities_data = communities_data.astype(float)
    x = communities_data[:, :-1]
    y = communities_data[:, -1]


    ### scale and normalize data 
    x = normalize(scale(x))
    y = (scale(y))
    ## The input data includes both features and response: data: (X, Y), X: array of shape (N, M), Y: array of shape (N,)
    data_reg=(x,y)

    ## Here we aim to evaluate the interpretation reliability of Ridge regression with cross validation, using the ``feature_impoReg``function. We use ``RidgeCV()`` from ``sklearn`` as our estimator. By setting ``importance_func=None``, the magnitude of coefficients will be used to evaluate feature importance. 

    from sklearn.linear_model import RidgeCV
    estimator=RidgeCV()
    importance_func=None

    ## We initialize the model with the ``mlreliability.feature_importance.feature_impoReg`` function.
    import imlreliability
    model_reg = imlreliability.feature_importance.feature_impoReg(data,estimator, importance_func) # < construct the model
    model_reg.fit() # < train the model
```
    
   
   Details of arguments are listed below. 
   

    Parameter name | Type | Task |Description
    ---------------|------|------|------------
    `data` | numpy array, required |All| Input samples (and response) used to be fed to `data` for reliability test. 
    `estimator` | object, required | All|scikit-learn object of the IML model to run.Either estimator needs to provide a ``score`` function or ``scoring`` must be passed.
    `n_repeat` | int, optional | All| Number of repeats in data perturbation, default as 100.
    `split_proportion` |float in (0,1), optional | All| The proportion of training set in data splitting, default as 0.7.
    `importance_func` |str, callable, list, tuple or dict, optional |Feature importance |Strategy to evaluate feature importance score.  
    `K_max` | int, optional | Feature importance|Number of top features of interest, default as 30.
    `evaluate_fun`|str, callable, list, tuple or dict|Feature importance |Function to evaluate prediction performance. Default as `mean_squared_error` in regression, 
    `get_prediction_consistency` |{True,False}, optional| Feature importance|Controls whether to calculate prediction consistency, default as `True`.
    `K` | int, required | Clustering| Number of clusters. 
    `label` | array of shape (N,1), optional | Clustering| True cluster labels, default as `None`.
    `rank`|int, optional|Dimension reduction|The number of reduced dimensions, default as `2`.
    `perturbation` |{'noise','split'}, optional | Clustering, dimension reduction|Controls the way of perturbation, default as `'noise'`.
    `sigma`|float,optinal|Clustering, dimension reduction|Controls level of noise. need to specify if `perturbation=='noise'`, default as `1`.
    `stratify`|{True,False}, optional|All|Controls whether to conduct stratified sampling, default as `True`.
    `norm` |{True,False}, optional| All|Controls whether to conduct data normalization, default as `True`.
    `rand_index` |RandomState instance, optional | All|Make sure the sampling uses the same RandomState instance for all iterations.
    `verbose` |{True,False}, optional| All| Controls the verbosity, default as `True`.


   2. Calculate interpretation consistency of the obtained interpretations via given reliability metrics. The consistency results is achieved via 
   
        - `model.get_consistency()` for feature importance 

        - `model.get_consistency_clustering()` for clustering 
    
    
        - `model.get_consistency_knn()` for neiborhood consistency in dimension reduction
    
    
        - `model.get_consistency_clustering()` for clustering consistency in dimension reduction
    
The ``.get_consistency`` function results in three pandas dataframe: ``accuracy``: prediction accuracy on test set; ``consistency``: interpretation consistency measured by RBO, Jaccard score, or user-defined metrics; and prediction_consistency measured by prediction entropy and purity if ``get_prediction_consistency ==True``. 
    
```python
    # Pseudo-code 
    model_reg.get_consistency(data_name='communities', estimator_name='Ridge',impotance_func_name='Coef')
    print(model_reg.accuracy)
    print(model_reg.consistency)
    print(model_reg.prediction_consistency)
```
 

The ``consistency`` pandas dataframe can be downloaded and upload to the dashboard. 

Details of arguments are listed below. 

    Parameter name | Type | Task |Description
    ---------------|------|------|------------
    `data_name` | str, required |All|Name of the data set. 
    `method_name` | str, required | All|Name of IML method. 
    `impotance_func_name`| str, optional | Feature importance |Name of funtion to measure feature importance, default as `None`.  
    `Kranges` | callable, optional | Dimension reduction (KNN) | list of numbers of nearest neighbors to measure, default as `None`.
    `cluster_func` | callable, optional | Dimension reduction (clustering) | clustering object applied to the reduced dimension, default as `AgglomerativeClustering()`.
    `cluster_func_name` |str, optional| Dimension reduction (clustering)| Name of clustering function, default as `'HC (ward)'`. 
    `user_metric`|callable, optional|All|User-defined evaluation metric of consistency, default as `None`.
    `user_metric_name`|callable, optional|All|User-defined evaluation metric of consistency, default as `'user_metric'`.



## API Documentations

The detailed API documentation for this package can be found at [https://DataSlingers.github.io/IML_reliability]


    