#!/usr/bin/env python3

import types
from copy import deepcopy
import warnings
import itertools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.stats import wilcoxon, shapiro, anderson, jarque_bera
from scipy.stats import f as scipyf

import pandas as pd

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm


__version__ = '0.1'

# Pretty print dicts
# Adapted from
# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries#comment108715660_3229493
def pretty_dict(d, indent=0, init_message=None):
    if init_message is not None:
        print(init_message)
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key))
            pretty_dict(value, indent+1)
        else:
            print('  ' * (indent+1) + f"{key}: {value}")

def mlcausality(X,
    y,
    lag,
    use_minmaxscaler=True,
    logdiff=True,
    split=None,
    train_size=0.7,
    early_stop_frac = 0.1,
    early_stop_min_samples = 128,
    early_stop_rounds = 25,
    use_standardscaler=False,
    y_bounds_error='warn',
    y_bounds_violation_wilcoxon_drop=True,
    regressor='catboostregressor',
    regressor_params=None,
    regressor_fit_params=None,
    check_model_type_match='raise',
    ftest=False,
    normality_tests=False,
    acorr_tests=False,
    return_kwargs_dict=True,
    return_preds=True,
    return_nanfilled=True,
    return_models=False,
    return_scalers=False,
    return_summary_df=True,
    kwargs_in_summary_df=True,
    pretty_print=True):
    """
    The function to generate predictions for series y based on:
        - the history of y only
        - the history of both X and y
    Note that Granger causality typically involves checking whether variable X --> y or y --> X.
    This function would only check for X --> y, if you want to test for causality in the other
    direction, you would have to call the function again with y and X switched. Also, note that
    y and X here can take numerous features (represented as columns). If y is multivariate, the
    target column is the first column in y; this allows for the inclusion of exogenous time series
    in both the restricted and unrestricted models in columns of y other than the first one.
    If X is multivariate, then the relationship that is tested is X --> y; in other words, we are 
    testing whether the inclusion of the lags of all the features in X improves the prediction of 
    the first column of y.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or None. This has only been tested to work 
    with pandas.Series, pandas.DataFrame or numpy arrays for single or multiple feature data.
    For single feature only, a list or a tuple of length n_samples can also be passed.
    
    y : array-like of shape (n_samples,). This has only been tested to work with 
    pandas.Series, pandas.DataFrame, numpy arrays, lists, or tuples. This is the target
    series on which to perform Granger analysis.
    
    lag : int. The number of lags to test Granger causality for.
    
    use_minmaxscaler : bool. If True, data is scaled to the range [2,3]. This is really if
    any of the time series can be zero or negative and a logdiff (as described below) has 
    to be taken. If your data has zeros or negative values or if you do not know what
    you are doing, keep this as True.
    
    logdiff: bool. Whether to take a log difference of the time series. For most regressors,
    such as tree-based regressors that cannot extrapolate, you want to set this to True, 
    unless your dataset will not require extrapolation to predict the Test set.
    Note that logdiff is applied after the train, val and test splits and that each
    of these datasets will lose an observation as a result of the logdiff operation.
    In consequence, len(test) - lag - 1 predictions will be made for both the 
    restricted and unrestricted models.
    
    split : None, an iterable of 2 iterables, or a generator that generates an 
    iterable with 2 iterables. In the typical case this will be a list of 2 lists
    where the first list contains the index numbers of rows in the training set
    and the second list contains the index numbers of rows in the testing set.
    If split is None, then train_size (described below) must be set.
    
    train_size : float between (0,1) or int. If split is None, and train_size
    is a float, this describes the fraction of the dataset used for training.
    If it is an int, it states how many observations to use for training.
    For instance, of the data has 1000 rows and train_size == 0.7 then the 
    first 700 rows are the training set and the latter 300 rows are the test set. 
    If early stopping is also used (see early_stop_frac below), then the 
    training set is further divided into a training set and a validation set. 
    For instance, if train_size == 0.7, early_stop_frac == 0.1, 
    enough data is available to early stop and a regressor is used that 
    employs early stopping then data that has 1000 rows will have a 
    training set size of 0.9*0.7*1000 = 630, a validation set size 
    of 0.1*0.7*1000 = 70, and a test set size of 0.3*1000 = 300.
    Note that each of these sets will further lose one observation if logdiff
    (described above) is set to True.
    
    early_stop_frac : float between [0,1). The fraction of training data to 
    use for early stopping if there is a sufficient number of observations and 
    regressor (described below) is one of 'catboostregressor', 'xgbregressor', 
    or 'lgbmregressor'. Note that if regressor is set to an instance of
    'catboostregressor', 'xgbregressor', or 'lgbmregressor', or a list composed
    of these strings, or to any other string, early_stop_frac has no effect. By
    enough observations what is meant is that early stopping will happen if
    len(train) - lags - 1 >= early_stop_min_samples
    where len(train) is the length of the training set (after logdiff if applied)
    and early_stop_min_samples is described below. If you do not want to use
    early stopping, set this to 0.0
    
    early_stop_min_samples : int. Early stopping minimum validation dataset size.
    For more information, read early_stop_frac above.
    
    early_stop_rounds : int. The number of rounds to use for early stopping. For
    more information, read the relevant documentation for CatBoost, LightGBM and/or
    XGBoost.
    
    use_standardscaler : bool.
    """
    # Store and parse the dict of passed variables
    if return_kwargs_dict:
        kwargs_dict=locals()
        del kwargs_dict['X']
        del kwargs_dict['y']
        if kwargs_dict['split'] is not None:
            kwargs_dict['split'] = 'notNone'
        if not isinstance(regressor, str):
            if len(regressor) != 2:
                raise ValueError('regressor was not a string or list-like of length 2 in mlcausality')
            else:
                kwargs_dict['regressor'] = [str(type(regressor[0])), str(type(regressor[1]))]
    ### Initial parameter checks; data scaling; and data splits
    early_stop = False
    if X is None or y is None or lag is None:
        raise TypeError('You must supply X, y and lags to mlcausality')
    if not isinstance(lag, int):
        raise TypeError('lag was not passed as an int to mlcausality')
    if isinstance(X, (list,tuple)):
        X = np.atleast_2d(X).reshape(-1,1)
    if isinstance(y, (list,tuple)):
        y = np.atleast_2d(y).reshape(-1,1)
    if isinstance(X, (pd.Series,pd.DataFrame)):
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = np.atleast_2d(X.to_numpy()).reshape(-1,1)
        else:
            X = X.to_numpy()
    if isinstance(y, (pd.Series,pd.DataFrame)):
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = np.atleast_2d(y.to_numpy()).reshape(-1,1)
        else:
            y = y.to_numpy()
    if not isinstance(X, np.ndarray):
        raise TypeError('X could not be cast to np.ndarray in mlcausality')
    if not isinstance(y, np.ndarray):
        raise TypeError('y could not be cast to np.ndarray in mlcausality')
    if len(X.shape) == 1:
        X = np.atleast_2d(X).reshape(-1,1)
    if len(y.shape) == 1:
        y = np.atleast_2d(y).reshape(-1,1)
    if X.shape[0] != y.shape[0]:
        print(X.shape)
        print(y.shape)
        raise ValueError('X and y must have the same length in dimension 0')
    if not isinstance(logdiff, bool):
        raise TypeError('logdiff must be a bool in mlcausality')
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    split_override = False
    if regressor == 'classic':
        regressor = 'linearregression'
        X = np.concatenate([X,X])
        y = np.concatenate([y,y])
        train_size = 0.5
        split_override = True
        use_minmaxscaler = False
        logdiff = False
        use_standardscaler = False
        y_bounds_error = 'ignore'
        y_bounds_violation_wilcoxon_drop = False
        acorr_tests = True
    #data_raw = np.concatenate([y, X],axis=1)
    if use_minmaxscaler:
        minmaxscalers = {}
        minmaxscalers['y'] = MinMaxScaler(feature_range=(2, 3))
        y_transformed = minmaxscalers['y'].fit_transform(y)
        minmaxscalers['X'] = MinMaxScaler(feature_range=(2, 3))
        X_transformed = minmaxscalers['X'].fit_transform(X)
        data_scaled = np.concatenate([y_transformed, X_transformed],axis=1)
    else:
        data_scaled = np.concatenate([y, X],axis=1)
    if not split_override and split is not None:
        if isinstance(split, types.GeneratorType):
            split = list(split)
        if len(split) != 2:
            raise ValueError('If split is provided to mlcausality, it must be of length 2')
        train = data_scaled[split[0], :]
        test = data_scaled[split[1], :]
    elif isinstance(train_size, int) and train_size != 0 and train_size != 1:
        if logdiff and train_size < lag+2:
            raise ValueError('train_size is too small, resulting in no samples in the train set!')
        elif logdiff and train_size > y.shape[0]-lag-2:
            raise ValueError('train_size is too large, resulting in no samples in the test set!')
        elif not logdiff and train_size < lag+1:
            raise ValueError('train_size is too small, resulting in no samples in the train set!')
        elif not logdiff and train_size > y.shape[0]-lag-1:
            raise ValueError('train_size is too large, resulting in no samples in the test set!')
        train = data_scaled[:train_size, :]
        test = data_scaled[train_size:, :]
    elif isinstance(train_size, float):
        if train_size <= 0 or train_size >= 1:
            raise ValueError('train_size is a float that is not between (0,1) in mlcausality')
        elif logdiff and round(train_size*y.shape[0])-lag-2 < 0:
            raise ValueError('train_size is a float that is too small resulting in no samples in train')
        elif logdiff and round((1-train_size)*y.shape[0])-lag-2 < 0:
            raise ValueError('train_size is a float that is too large resulting in no samples in test')
        elif not logdiff and round(train_size*y.shape[0])-lag-1 < 0:
            raise ValueError('train_size is a float that is too small resulting in no samples in train')
        elif not logdiff and round((1-train_size)*y.shape[0])-lag-1 < 0:
            raise ValueError('train_size is a float that is too large resulting in no samples in test')
        else:
            train = data_scaled[:round(train_size*y.shape[0]), :]
            test = data_scaled[round(train_size*y.shape[0]):, :]
    else:
        raise TypeError('train_size must be provided as a float or int to mlcausality. Alternatively, you can provide a split to "split".')
    train_orig_shape0 = deepcopy(train.shape[0])
    ### Regressors
    if regressor_fit_params is None:
        regressor_fit_params_restrict = {}
        regressor_fit_params_unrestrict = {}
    elif isinstance(regressor_fit_params, dict):
        regressor_fit_params_restrict = regressor_fit_params
        regressor_fit_params_unrestrict = regressor_fit_params
    elif isinstance(regressor_fit_params, list):
        if len(regressor_fit_params) != 2 or not isinstance(regressor_fit_params[0], dict) or not isinstance(regressor_fit_params[1], dict):
            raise ValueError('regressor_fit_params must be None, a dict, or a list of 2 dicts')
        else:
            regressor_fit_params_restrict = regressor_fit_params[0]
            regressor_fit_params_unrestrict = regressor_fit_params[1]
    if not isinstance(regressor, str):
        if len(regressor) != 2:
            raise ValueError('regressor was not a string or list-like of length 2 in mlcausality')
        elif check_model_type_match=='raise' and type(regressor[0]) != type(regressor[1]):
            raise TypeError('regressors passed for the restricted and unrestricted models are of different types. This does not really make much sense for the purposes of Granger causality testing because the performance of different types of regressors could be vastly different, which could lead to erroneous conclusions regarding Granger causality. If you know what you are doing, you can re-run with check_model_type_match="warn" or check_model_type_match="ignore"')
        elif check_model_type_match=='warn' and type(regressor[0]) != type(regressor[1]):
            warnings.warn('regressors passed for the restricted and unrestricted models are of different types.')
            model_restrict = regressor[0]
            model_unrestrict = regressor[1]
        else:
            model_restrict = regressor[0]
            model_unrestrict = regressor[1]
    else:
        if regressor_params is not None:
            if not isinstance(regressor_params, (dict,list)):
                raise TypeError('regressor_params have to be one of None, dict, or list of 2 dicts')
            elif isinstance(regressor_params, list):
                if not len(regressor_params) == 2 and not isinstance(regressor_params[0], dict) and not isinstance(regressor_params[1], dict):
                    raise TypeError('regressor_params have to be one of None, dict, or list of 2 dicts')
                else:
                    params_restrict = regressor_params[0]
                    params_unrestrict = regressor_params[1]
            else:
                params_restrict = regressor_params
                params_unrestrict = regressor_params
        else:
            params_restrict = {}
            params_unrestrict = {}
        if regressor.lower() in ['catboostregressor', 'xgbregressor', 'lgbmregressor']:
            if not isinstance(early_stop_frac, float) or early_stop_frac < 0 or early_stop_frac >= 1:
                raise ValueError("early_stop_frac must be a float in [0,1) if regressor.lower() in ['catboostregressor', 'xgbregressor', 'lgbmregressor','gradientboostingregressor','histgradientboostingregressor']")
            if not isinstance(early_stop_min_samples, int):
                raise TypeError('early_stop_min_samples must be an int')
            # if we have less than early_stop_min_samples samples for validation, do not use early stopping. Otherwise, use early stopping
            if logdiff and round(early_stop_frac*train.shape[0])-lag-1-early_stop_min_samples < 0:
                early_stop = False
            elif not logdiff and round(early_stop_frac*train.shape[0])-lag-early_stop_min_samples < 0:
                early_stop = False
            else:
                early_stop = True
            if early_stop:
                val = deepcopy(train[round((1-early_stop_frac)*train.shape[0]):,:])
                train = deepcopy(train[:round((1-early_stop_frac)*train.shape[0]),:])
        if regressor.lower() == 'catboostregressor':
            from catboost import CatBoostRegressor
            if early_stop:
                params_restrict.update({'early_stopping_rounds':early_stop_rounds})
                params_unrestrict.update({'early_stopping_rounds':early_stop_rounds})
            model_restrict = CatBoostRegressor(**params_restrict)
            model_unrestrict = CatBoostRegressor(**params_unrestrict)
        elif regressor.lower() == 'xgbregressor':
            from xgboost import XGBRegressor
            if early_stop:
                params_restrict.update({'early_stopping_rounds':early_stop_rounds})
                params_unrestrict.update({'early_stopping_rounds':early_stop_rounds})
            model_restrict = XGBRegressor(**params_restrict)
            model_unrestrict = XGBRegressor(**params_unrestrict)
        elif regressor.lower() == 'lgbmregressor':
            from lightgbm import LGBMRegressor
            model_restrict = LGBMRegressor(**params_restrict)
            model_unrestrict = LGBMRegressor(**params_unrestrict)
        elif regressor.lower() == 'linearregression':
            ftest = True
            from sklearn.linear_model import LinearRegression
            model_restrict = LinearRegression(**params_restrict)
            model_unrestrict = LinearRegression(**params_unrestrict)
        elif regressor.lower() == 'randomforestregressor':
            from sklearn.ensemble import RandomForestRegressor
            model_restrict = RandomForestRegressor(**params_restrict)
            model_unrestrict = RandomForestRegressor(**params_unrestrict)
        elif regressor.lower() == 'svr':
            from sklearn.svm import SVR
            model_restrict = SVR(**params_restrict)
            model_unrestrict = SVR(**params_unrestrict)
        elif regressor.lower() == 'gradientboostingregressor':
            from sklearn.ensemble import GradientBoostingRegressor
            model_restrict = GradientBoostingRegressor(**params_restrict)
            model_unrestrict = GradientBoostingRegressor(**params_unrestrict)
        elif regressor.lower() == 'histgradientboostingregressor':
            from sklearn.ensemble import HistGradientBoostingRegressor
            model_restrict = HistGradientBoostingRegressor(**params_restrict)
            model_unrestrict = HistGradientBoostingRegressor(**params_unrestrict)
        elif regressor.lower() == 'cuml_svr':
            from cuml.svm import SVR
            model_restrict = SVR(**params_restrict)
            model_unrestrict = SVR(**params_unrestrict)
        elif regressor.lower() == 'cuml_randomforestregressor':
            from cuml.ensemble import RandomForestRegressor
            model_restrict = RandomForestRegressor(**params_restrict)
            model_unrestrict = RandomForestRegressor(**params_unrestrict)
        else:
            raise ValueError('unidentified string regressor passed to mlcausality')
    ### Logdiff
    if logdiff:
        train_integ = np.diff(np.log(train), axis=0)
        test_integ = np.diff(np.log(test), axis=0)
        if early_stop:
            val_integ = np.diff(np.log(val), axis=0)
    elif not logdiff:
        train_integ = train
        test_integ = test
        if early_stop:
            val_integ = val
    ### Standard scaler
    if use_standardscaler:
        standardscalers = {}
        standardscalers['y'] = StandardScaler(copy=False)
        standardscalers['X'] = StandardScaler(copy=False)
        train_integ[:,:y.shape[1]] = standardscalers['y'].fit_transform(train_integ[:,:y.shape[1]])
        train_integ[:,y.shape[1]:] = standardscalers['X'].fit_transform(train_integ[:,y.shape[1]:])
        test_integ[:,:y.shape[1]] = standardscalers['y'].transform(test_integ[:,:y.shape[1]])
        test_integ[:,y.shape[1]:] = standardscalers['X'].transform(test_integ[:,y.shape[1]:])
        if early_stop:
            val_integ[:,:y.shape[1]] = standardscalers['y'].transform(val_integ[:,:y.shape[1]])
            val_integ[:,y.shape[1]:] = standardscalers['X'].transform(val_integ[:,y.shape[1]:])
    ### y bounds error
    if y_bounds_error == 'raise':
        if np.nanmax(train_integ[:,0]) < np.nanmax(test_integ[lag:,0]) or np.nanmin(train_integ[:,0]) > np.nanmin(test_integ[lag:,0]):
            raise ValueError('[y_test_min,y_test_max] is not a subset of [y_train_min,y_train_max]. Since many algorithms, especially tree-based algorithms, cannot extrapolate, this could result in erroneous conclusions regarding Granger causality. If you would still like to perform the Granger causality test anyway, re-run mlcausality with y_bounds_error set to either "warn" or "ignore".')
    elif y_bounds_error == 'warn':
        if np.nanmax(train_integ[:,0]) < np.nanmax(test_integ[lag:,0]) or np.nanmin(train_integ[:,0]) > np.nanmin(test_integ[lag:,0]):
            warnings.warn('[y_test_min,y_test_max] is not a subset of [y_train_min,y_train_max].  Since many algorithms, especially tree-based algorithms, cannot extrapolate, this could result in erroneous conclusions regarding Granger causality.')
    ### y bounds indicies and fractions
    inside_bounds_idx = np.where(np.logical_and(test_integ[lag:,0] >= np.nanmin(train_integ[:,0]), test_integ[lag:,0] <= np.nanmax(train_integ[:,0])))[0]
    outside_bounds_frac = (test_integ[lag:,0].shape[0] - inside_bounds_idx.shape[0])/test_integ[lag:,0].shape[0]
    ### Sliding window views
    train_sw = sliding_window_view(train_integ, [lag+1,data_scaled.shape[1]]) # Lag+1 gives lag features plus the target column
    test_sw = sliding_window_view(test_integ, [lag+1,data_scaled.shape[1]]) # Lag+1 gives lag features plus the target column
    if early_stop:
        val_sw = sliding_window_view(val_integ, [lag+1,data_scaled.shape[1]]) # Lag+1 gives lag features plus the target column
    ### Reshape data
    train_sw_reshape_restrict = train_sw[:,:,:,:y.shape[1]].reshape(train_sw[:,:,:,:y.shape[1]].shape[0],train_sw[:,:,:,:y.shape[1]].shape[1]*train_sw[:,:,:,:y.shape[1]].shape[2]*train_sw[:,:,:,:y.shape[1]].shape[3])
    test_sw_reshape_restrict = test_sw[:,:,:,:y.shape[1]].reshape(test_sw[:,:,:,:y.shape[1]].shape[0],test_sw[:,:,:,:y.shape[1]].shape[1]*test_sw[:,:,:,:y.shape[1]].shape[2]*test_sw[:,:,:,:y.shape[1]].shape[3])
    train_sw_reshape_unrestrict = train_sw.reshape(train_sw.shape[0],train_sw.shape[1]*train_sw.shape[2]*train_sw.shape[3])
    test_sw_reshape_unrestrict = test_sw.reshape(test_sw.shape[0],test_sw.shape[1]*test_sw.shape[2]*test_sw.shape[3])
    if early_stop:
        val_sw_reshape_restrict = val_sw[:,:,:,:y.shape[1]].reshape(val_sw[:,:,:,:y.shape[1]].shape[0],val_sw[:,:,:,:y.shape[1]].shape[1]*val_sw[:,:,:,:y.shape[1]].shape[2]*val_sw[:,:,:,:y.shape[1]].shape[3])
        val_sw_reshape_unrestrict = val_sw.reshape(val_sw.shape[0],val_sw.shape[1]*val_sw.shape[2]*val_sw.shape[3])
    ### Handle early stopping
    if isinstance(regressor, str) and regressor.lower() in ['catboostregressor', 'xgbregressor'] and early_stop:
        regressor_fit_params_restrict.update({'eval_set':[(val_sw_reshape_restrict[:, :-y.shape[1]], val_sw_reshape_restrict[:, -y.shape[1]])]})
        regressor_fit_params_unrestrict.update({'eval_set':[(val_sw_reshape_unrestrict[:, :-data_scaled.shape[1]], val_sw_reshape_unrestrict[:, -data_scaled.shape[1]])]})
    elif isinstance(regressor, str) and regressor.lower() == 'lgbmregressor' and early_stop:
        import lightgbm
        if 'verbose' in params_restrict.keys():
            lgbm_restrict_verbosity = params_restrict['verbose']
        else:
            lgbm_restrict_verbosity = True
        if 'verbose' in params_unrestrict.keys():
            lgbm_unrestrict_verbosity = params_unrestrict['verbose']
        else:
            lgbm_unrestrict_verbosity = True
        lgbm_early_stopping_callback_restrict = lightgbm.early_stopping(early_stop_rounds, first_metric_only=True, verbose=lgbm_restrict_verbosity)
        lgbm_early_stopping_callback_unrestrict = lightgbm.early_stopping(early_stop_rounds, first_metric_only=True, verbose=lgbm_unrestrict_verbosity)
        regressor_fit_params_restrict.update({'callbacks':[lgbm_early_stopping_callback_restrict], 'eval_set':[(deepcopy(val_sw_reshape_restrict[:, :-y.shape[1]]), deepcopy(val_sw_reshape_restrict[:, -y.shape[1]]))]})
        regressor_fit_params_unrestrict.update({'callbacks':[lgbm_early_stopping_callback_unrestrict], 'eval_set':[(deepcopy(val_sw_reshape_unrestrict[:, :-data_scaled.shape[1]]), deepcopy(val_sw_reshape_unrestrict[:, -data_scaled.shape[1]]))]})
    ### Pred y using only past values of y
    model_restrict.fit(deepcopy(train_sw_reshape_restrict[:, :-y.shape[1]]), deepcopy(train_sw_reshape_restrict[:, -y.shape[1]]), **regressor_fit_params_restrict)
    preds_restrict = model_restrict.predict(deepcopy(test_sw_reshape_restrict[:, :-y.shape[1]])).flatten()
    #ytrue_restrict = test_sw_reshape_restrict[:, -1].flatten()
    ### Pred y using past values of y and X    
    model_unrestrict.fit(deepcopy(train_sw_reshape_unrestrict[:, :-data_scaled.shape[1]]), deepcopy(train_sw_reshape_unrestrict[:, -data_scaled.shape[1]]), **regressor_fit_params_unrestrict)
    preds_unrestrict = model_unrestrict.predict(deepcopy(test_sw_reshape_unrestrict[:, :-data_scaled.shape[1]])).flatten()
    #ytrue_unrestrict = test_sw_reshape_unrestrict[:, -data_scaled.shape[1]].flatten()
    ### Transform preds and ytrue if transformations were originally applied
    ytrue = y[-preds_restrict.shape[0]:,[0]]
    if use_standardscaler:
        if y.shape[0] > 1:
            preds_restrict_for_standardscaler = np.concatenate([preds_restrict.reshape(-1, 1),np.zeros_like(y[:preds_restrict.shape[0],1:])], axis=1)
            preds_unrestrict_for_standardscaler = np.concatenate([preds_unrestrict.reshape(-1, 1),np.zeros_like(y[:preds_unrestrict.shape[0],1:])], axis=1)
        else:
            preds_restrict_for_standardscaler = preds_restrict.reshape(-1, 1)
            preds_unrestrict_for_standardscaler = preds_unrestrict.reshape(-1, 1)
        preds_restrict = standardscalers['y'].inverse_transform(preds_restrict_for_standardscaler)[:,0].flatten()
        preds_unrestrict = standardscalers['y'].inverse_transform(preds_unrestrict_for_standardscaler)[:,0].flatten()
        #ytrue_restrict = standardscalers['y'].inverse_transform(ytrue_restrict.reshape(-1, 1)).flatten()
        #ytrue_unrestrict = standardscalers['y'].inverse_transform(ytrue_unrestrict.reshape(-1, 1)).flatten()
    if logdiff:
        preds_restrict = (np.exp(preds_restrict)*(test[lag:-1,0])).reshape(-1, 1)
        preds_unrestrict = (np.exp(preds_unrestrict)*(test[lag:-1,0])).reshape(-1, 1)
        #ytrue_restrict = (np.exp(ytrue_restrict)*(test[lag:-1,0])).reshape(-1, 1)
        #ytrue_unrestrict = (np.exp(ytrue_unrestrict)*(test[lag:-1,0])).reshape(-1, 1)
    else:
        preds_restrict = preds_restrict.reshape(-1, 1)
        preds_unrestrict = preds_unrestrict.reshape(-1, 1)
        #ytrue_restrict = ytrue_restrict.reshape(-1, 1)
        #ytrue_unrestrict = ytrue_unrestrict.reshape(-1, 1)
    if use_minmaxscaler:
        if y.shape[0] > 1:
            preds_restrict_for_minmaxscaler = np.concatenate([preds_restrict.reshape(-1, 1),np.ones_like(y[:preds_restrict.shape[0],1:])*np.mean(minmaxscalers['y'].feature_range)], axis=1)
            preds_unrestrict_for_minmaxscaler = np.concatenate([preds_unrestrict.reshape(-1, 1),np.ones_like(y[:preds_unrestrict.shape[0],1:])*np.mean(minmaxscalers['y'].feature_range)], axis=1)
        else:
            preds_restrict_for_minmaxscaler = preds_restrict.reshape(-1, 1)
            preds_unrestrict_for_minmaxscaler = preds_unrestrict.reshape(-1, 1)
        preds_restrict = minmaxscalers['y'].inverse_transform(preds_restrict_for_minmaxscaler)[:,[0]]
        preds_unrestrict = minmaxscalers['y'].inverse_transform(preds_unrestrict_for_minmaxscaler)[:,[0]]
        #ytrue_restrict = minmaxscalers['y'].inverse_transform(ytrue_restrict)
        #ytrue_unrestrict = minmaxscalers['y'].inverse_transform(ytrue_unrestrict)
    errors_restrict = preds_restrict - ytrue
    errors_unrestrict = preds_unrestrict - ytrue
    if y_bounds_violation_wilcoxon_drop:
        wilcoxon_abserror = wilcoxon(np.abs(errors_restrict[inside_bounds_idx].flatten()), np.abs(errors_unrestrict[inside_bounds_idx].flatten()), alternative='greater', nan_policy='omit')
        wilcoxon_num_preds = errors_restrict[inside_bounds_idx].flatten().shape[0]
    else:
        wilcoxon_abserror = wilcoxon(np.abs(errors_restrict.flatten()), np.abs(errors_unrestrict.flatten()), alternative='greater', nan_policy='omit')
        wilcoxon_num_preds = errors_restrict.flatten().shape[0]
    if ftest:
        normality_tests = True
        errors2_restrict = errors_restrict**2
        errors2_unrestrict = errors_unrestrict**2
        f_dfn = lag*y.shape[1]
        f_dfd = errors2_restrict.shape[0]-(lag*(y.shape[1]+X.shape[1]))-1
        if f_dfd <= 0:
            f_stat = np.nan
            ftest_p_value = np.nan
        else:
            f_stat = ((errors2_restrict.sum() - errors2_unrestrict.sum())/f_dfn)/(errors2_unrestrict.sum()/f_dfd)
            ftest_p_value = scipyf.sf(f_stat, f_dfn, f_dfd)
    if normality_tests:
        shapiro_restrict = shapiro(errors_restrict.flatten())
        shapiro_unrestrict = shapiro(errors_unrestrict.flatten())
        anderson_restrict = anderson(errors_restrict.flatten())
        anderson_unrestrict = anderson(errors_unrestrict.flatten())
        jarque_bera_restrict = jarque_bera(errors_restrict.flatten(), nan_policy='omit')
        jarque_bera_unrestrict = jarque_bera(errors_unrestrict.flatten(), nan_policy='omit')
    if acorr_tests:
        durbin_watson_restricted = durbin_watson(errors_restrict.flatten())
        durbin_watson_unrestricted = durbin_watson(errors_unrestrict.flatten())
        acorr_ljungbox_restricted = acorr_ljungbox(errors_restrict.flatten(), auto_lag=True, model_df=lag*y.shape[1])
        acorr_ljungbox_unrestricted = acorr_ljungbox(errors_unrestrict.flatten(), auto_lag=True, model_df=lag*(y.shape[1]+X.shape[1]))
    if return_nanfilled:
        preds_empty = np.empty([train_orig_shape0+lag+1,]).reshape(-1, 1) # First prediction value will be t=lag+1 to account for the logdiff
        preds_empty[:] = np.nan
        preds_restrict_nanfilled = np.concatenate([preds_empty,preds_restrict])
        preds_unrestrict_nanfilled = np.concatenate([preds_empty,preds_unrestrict])
        ytrue_nanfilled = y[:,[0]]
        #ytrue_restrict_nanfilled = np.concatenate([preds_empty,ytrue_restrict])
        #ytrue_unrestrict_nanfilled = np.concatenate([preds_empty,ytrue_unrestrict])
    return_dict = {'summary':{'lag':lag, 'train_obs':train_integ[:,0].shape[0], 'effective_train_obs':train_integ[lag:,0].shape[0], 'test_obs':test_integ[:,0].shape[0], 'effective_test_obs':test_integ[lag:,0].shape[0]}}
    if early_stop:
        return_dict['summary'].update({'val_obs':val_integ[:,0].shape[0], 'effective_val_obs':val_integ[lag:,0].shape[0]})
    return_dict['summary'].update({'outside_bounds_frac':outside_bounds_frac, 'wilcoxon':{'statistic':wilcoxon_abserror.statistic, 'pvalue':wilcoxon_abserror.pvalue, 'y_bounds_violation_wilcoxon_drop':y_bounds_violation_wilcoxon_drop, 'wilcoxon_num_preds':wilcoxon_num_preds}})
    if ftest:
        return_dict['summary'].update({'ftest':{'statistic':f_stat,'pvalue':ftest_p_value,'dfn':f_dfn,'dfd':f_dfd}})
    if normality_tests:
        return_dict['summary'].update({'normality_tests':{'shapiro':{'restricted':{'statistic':shapiro_restrict.statistic, 'pvalue':shapiro_restrict.pvalue}, 'unrestricted':{'statistic':shapiro_unrestrict.statistic, 'pvalue':shapiro_unrestrict.pvalue}}, 'anderson':{'restricted':{'statistic':anderson_restrict.statistic, 'critical_values':anderson_restrict.critical_values, 'significance_level':anderson_restrict.significance_level, 'fit_result':{'params':anderson_restrict.fit_result.params, 'success':anderson_restrict.fit_result.success, 'message':anderson_restrict.fit_result.message}}, 'unrestricted':{'statistic':anderson_unrestrict.statistic, 'critical_values':anderson_unrestrict.critical_values, 'significance_level':anderson_unrestrict.significance_level, 'fit_result':{'params':anderson_unrestrict.fit_result.params, 'success':anderson_unrestrict.fit_result.success, 'message':anderson_unrestrict.fit_result.message}}}, 'jarque_bera':{'restricted':{'statistic':jarque_bera_restrict.statistic, 'pvalue':jarque_bera_restrict.pvalue}, 'unrestricted':{'statistic':jarque_bera_unrestrict.statistic, 'pvalue':jarque_bera_unrestrict.pvalue}}}})
    if acorr_tests:
        return_dict['summary'].update({'durbin_watson': {'restricted':durbin_watson_restricted, 'unrestricted':durbin_watson_unrestricted}})
        return_dict.update({'ljungbox':{'restricted':acorr_ljungbox_restricted, 'unrestricted':acorr_ljungbox_unrestricted}})
    if return_summary_df:
        return_dict.update({'summary_df': pd.json_normalize(return_dict['summary'])})
    if return_kwargs_dict:
        return_dict.update({'kwargs_dict':kwargs_dict})
    if return_kwargs_dict and kwargs_in_summary_df:
        kwargs_df = pd.json_normalize(return_dict['kwargs_dict'])
        kwargs_df = kwargs_df.loc[[0],[i for i in kwargs_df.columns if i not in ['lag']]]
        return_dict['summary_df'] = return_dict['summary_df'].loc[[0],[i for i in return_dict['summary_df'].columns if i not in ['wilcoxon.y_bounds_violation_wilcoxon_drop']]]
        return_dict['summary_df'] = pd.concat([return_dict['summary_df'], kwargs_df], axis=1)
    if return_preds:
        return_dict.update({'ytrue':ytrue, 'preds':{'restricted':preds_restrict, 'unrestricted':preds_unrestrict}})
    if return_nanfilled:
        return_dict.update({'ytrue_nanfilled':ytrue_nanfilled, 'preds_nanfilled':{'restricted':preds_restrict_nanfilled, 'unrestricted':preds_unrestrict_nanfilled}})
    if return_models:
        return_scalers = True
        return_dict.update({'models':{'restricted': model_restrict, 'unrestricted': model_unrestrict}})
    if return_scalers:
        if use_minmaxscaler:
            return_dict.update({'scalers':{'minmaxscalers':minmaxscalers}})
            if use_standardscaler:
                return_dict['scalers'].update({'standardscalers':standardscalers})
        elif use_standardscaler:
            return_dict.update({'scalers':{'standardscalers':standardscalers}})
            if use_minmaxscaler:
                return_dict['scalers'].update({'minmaxscalers':minmaxscalers})
    if pretty_print:
        pretty_dict(return_dict['summary'], init_message='########## SUMMARY ##########')
    return return_dict


def mlcausality_splits_loop(splits, X=None, y=None, lag=None, loop_position=0, **kwargs):
    if X is not None:
        kwargs.update({'X':X})
    if y is not None:
        kwargs.update({'y':y})
    if lag is not None:
        kwargs.update({'lag':lag})
    kwargs.update({'return_preds':False,'return_nanfilled':False,'return_models':False,'return_scalers':False,
        'return_summary_df':True,'kwargs_in_summary_df':True,'pretty_print':False})
    if isinstance(splits, types.GeneratorType):
        splits = list(splits)
    out_dfs = []
    split_counter = 0
    for train_idx, test_idx in tqdm(splits,desc=' splits loop',position=loop_position,leave=False):
        kwargs.update({'split':[train_idx,test_idx]})
        out_df = mlcausality(**kwargs)['summary_df']
        out_df['split'] = split_counter
        out_dfs.append(out_df)
        split_counter += 1
    all_out_dfs = pd.concat(out_dfs,ignore_index=True)
    return all_out_dfs








def binary_mlcausality(data, lags, permute_list=None, splits=None, **kwargs):
    if 'y' in kwargs:
        del kwargs['y']
    if 'X' in kwargs:
        del kwargs['X']
    if 'lag' in kwargs:
        del kwargs['lag']
    kwargs.update({'return_preds':False,'return_nanfilled':False,'return_models':False,'return_scalers':False,
        'return_summary_df':True,'kwargs_in_summary_df':True,'pretty_print':False})
    if isinstance(data, pd.DataFrame):
        hasnames = True
        names = data.columns.to_list()
        data = data.to_numpy()
    else:
        hasnames = False
    if permute_list is None:
        permute_list = list(itertools.permutations(range(data.shape[1]),2))
    out_dfs = []
    for y_idx, X_idx in tqdm(permute_list,desc=' permute loop',position=0):
        y = data[:,[y_idx]]
        X = data[:,[X_idx]]
        for lag in tqdm(lags,desc=' lags loop',position=1,leave=False):
            if splits is not None:
                out_df = mlcausality_splits_loop(splits, X, y, lag, 2, **kwargs)
                if hasnames:
                    out_df['y'] = names[y_idx]
                    out_df['X'] = names[X_idx]
                else:
                    out_df['y'] = y_idx
                    out_df['X'] = X_idx
            else:
                out_df = mlcausality(X, y, lag, **kwargs)['summary_df']
                if hasnames:
                    out_df['y'] = names[y_idx]
                    out_df['X'] = names[X_idx]
                else:
                    out_df['y'] = y_idx
                    out_df['X'] = X_idx
            out_dfs.append(out_df)
    all_out_dfs = pd.concat(out_dfs,ignore_index=True)
    all_out_dfs = all_out_dfs.loc[:, ['y','X']+[i for i in all_out_dfs.columns if i not in ['y','X']]]
    return all_out_dfs


