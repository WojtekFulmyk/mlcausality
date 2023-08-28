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
from scipy.stats import binomtest

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

__version__ = "1.0"

# Pretty print dicts
# Adapted from
# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries#comment108715660_3229493


def pretty_dict(d, indent=0, init_message=None):
    if init_message is not None:
        print(init_message)
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + str(key))
            pretty_dict(value, indent + 1)
        else:
            print("  " * (indent + 1) + f"{key}: {value}")


# initialize a scaler with params
def init_scaler(scaler="standardscaler", scaler_params=None):
    scaler = scaler.lower()
    if scaler_params is None:
        scaler_params = {}
    if scaler == "maxabsscaler" or scaler == "maxabs":
        scaler_obj = MaxAbsScaler(**scaler_params)
    elif scaler == "minmaxscaler" or scaler == "minmax":
        scaler_obj = MinMaxScaler(**scaler_params)
    elif scaler == "powertransformer" or scaler == "power":
        scaler_obj = PowerTransformer(**scaler_params)
    elif scaler == "quantiletransformer" or scaler == "quantile":
        scaler_obj = QuantileTransformer(**scaler_params)
    elif scaler == "robustscaler" or scaler == "robust":
        scaler_obj = RobustScaler(**scaler_params)
    elif scaler == "standardscaler" or scaler == "standard":
        scaler_obj = StandardScaler(**scaler_params)
    elif scaler == "normalizer":
        scaler_obj = Normalizer(**scaler_params)
    return scaler_obj


def mlcausality(
    X,
    y,
    lag,
    scaler_init_1=None,
    scaler_init_1_params=None,
    scaler_init_2=None,
    scaler_init_2_params=None,
    scaler_prelogdiff_1=None,
    scaler_prelogdiff_1_params=None,
    scaler_prelogdiff_2=None,
    scaler_prelogdiff_2_params=None,
    logdiff=False,
    scaler_postlogdiff_1=None,
    scaler_postlogdiff_1_params=None,
    scaler_postlogdiff_2=None,
    scaler_postlogdiff_2_params=None,
    split=None,
    train_size=0.7,
    early_stop_frac=0.0,
    early_stop_min_samples=1000,
    early_stop_rounds=50,
    scaler_postsplit_1=None,
    scaler_postsplit_1_params=None,
    scaler_postsplit_2=None,
    scaler_postsplit_2_params=None,
    scaler_dm_1=None,
    scaler_dm_1_params=None,
    scaler_dm_2=None,
    scaler_dm_2_params=None,
    y_bounds_error="ignore",
    y_bounds_violation_sign_drop=True,
    regressor="default",
    regressor_params=None,
    regressor_fit_params=None,
    check_model_type_match="raise",
    ftest=False,
    normality_tests=False,
    acorr_tests=False,
    return_restrict_only=False,
    return_inside_bounds_mask=False,
    return_kwargs_dict=True,
    return_preds=True,
    return_errors=False,
    return_nanfilled=True,
    return_models=False,
    return_scalers=False,
    return_summary_df=True,
    kwargs_in_summary_df=True,
    pretty_print=True,
):
    """
    mclcausality is a function to generate predictions for series y
    based on:
        1) the lags of y only (the restricted model); and
        2) the lags of both X and y (the unrestricted model).
    In addition to generating just the predictions, mlcausality is
    capable of running Granger causality analysis whereby the null that
    X does not Granger cause y is tested.

    This function only checks for X --> y, causality in the other
    direction is not tested. Note that y and X here can be multivariate
    with different time-series represented by columns and time
    represented by rows. If y is multivariate, the target time-series
    is the first column in y: this allows for the inclusion of
    exogenous time-series in both the restricted and unrestricted
    models in columns of y other than the first one. If X is
    multivariate, then the relationship that is tested is still
    X --> y; in other words, the null hypothesis is that a model that
    includes the lags of all the time-series in X and y does not
    provide superior predictions for the time-series in the first
    column of y when compared to a model composed of the lags of all
    time-series in y only.

    returns a dict with elements that depend on the parameters with
    which this function is called.

    Example usage:
    import mlcausality
    import numpy as np
    import pandas as pd
    X = np.random.random([1000,5])
    y = np.random.random([1000,4])
    z = mlcausality.mlcausality(X=X,y=y,lag=5)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or None.
    This has only been tested to work with pandas.Series,
    pandas.DataFrame or numpy arrays for single or multiple time-series
    data. If only one time-series is to be included in X, then a list
    or a tuple of length n_samples be passed instead.

    y : array-like of shape (n_samples,) or (n_samples, n_features).
    This has only been tested to work with pandas.Series,
    pandas.DataFrame, numpy arrays, lists, or tuples. This is the
    target time-series on which Granger causality analysis is
    performed. If y has multiple time-series (represented by columns),
    the target time-series is the first column.

    lag : int.
    The number of lags to test Granger causality for.

    scaler_init_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler are NOT
    inversed before a test of Granger causality occurs: once data is
    transformed using this scaler, it stays transformed throughout the
    entire call of this function. The scaling is done using the
    relevant scaler from scikit-learn. Parameters can be set using
    'scaler_init_1_params'.

    scaler_init_1_params : dict or None.
    The parameters for 'scaler_init_1'. The parameters must correspond
    to the relevant scaler's parameters in the scikit-learn package.

    scaler_init_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler are NOT
    inversed before a test of Granger causality occurs: once data is
    transformed using this scaler, it stays transformed throughout the
    entire call of this function. The scaling is done using the
    relevant scaler from scikit-learn. Parameters can be set using
    'scaler_init_2_params'.

    scaler_init_2_params : dict or None.
    The parameters for 'scaler_init_2'. The parameters must correspond
    to the relevant scaler's parameters in the scikit-learn package.

    scaler_prelogdiff_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_prelogdiff_1_params'.

    scaler_prelogdiff_1_params : dict or None.
    The parameters for 'scaler_prelogdiff_1'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    scaler_prelogdiff_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_prelogdiff_2_params'.

    scaler_prelogdiff_2_params : dict or None.
    The parameters for 'scaler_prelogdiff_2'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    logdiff: bool.
    Whether to take a log difference of all the time-series in y and X.
    Note that logdiff is applied before the train, val and test splits
    are taken but that each of these datasets will lose an observation
    as a result of the logdiff operation. In consequence,
    len(test) - lag - 1 predictions will be made by both the restricted
    and unrestricted models. Predictions are subjected to an inversion
    of the logdiff operation and Granger causality analysis occurs on
    the inversed prediction data.

    scaler_postlogdiff_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postlogdiff_1_params'.

    scaler_postlogdiff_1_params : dict or None.
    The parameters for 'scaler_postlogdiff_1'. The parameters must
    correspond to the relevant scaler's parameters in the
    scikit-learn package.

    scaler_postlogdiff_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postlogdiff_2_params'.

    scaler_postlogdiff_2_params : dict or None.
    The parameters for 'scaler_postlogdiff_2'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    split : None, or an iterable of 2 iterables.
    In the typical case this will be a list of 2 lists where the first
    list contains the index numbers of rows in the training set and the
    second list contains the index numbers of rows in the testing set.
    Note that the index numbers in both the train and test splits MUST
    be consecutive and without gaps, otherwise lags will not be taken
    correctly for the time-series in y and X. If split is None, then
    train_size (described below) must be set.

    train_size : float between (0,1) or int.
    If split is None then this train_size describes the fraction of the
    dataset used for training. If it is an int, it states how many
    observations to use for training. For instance, if the data has
    1000 rows and train_size == 0.7 then the first 700 rows are the
    training set and the latter 300 rows are the test set. If early
    stopping is also used (see early_stop_frac below), then the
    training set is further divided into a training set and a
    validation set. For example, if train_size == 0.7,
    early_stop_frac == 0.1, enough data is available to early stop, and
    a regressor is used that employs early stopping, then data that has
    1000 rows will have a training set size of 0.9*0.7*1000 = 630, a
    validation set size of 0.1*0.7*1000 = 70, and a test set size of
    0.3*1000 = 300. Note that each of these sets will further lose one
    observation if logdiff (described above) is set to True. If
    train_size==1 and split==None then the train and test sets are
    identical and equal to the entire dataset.

    early_stop_frac : float between [0.0,1.0).
    The fraction of training data to use for early stopping if there is
    a sufficient number of observations and the regressor (described
    below) is one of 'catboostregressor', 'xgbregressor', or
    'lgbmregressor'. Note that if the regressor is set to a string
    other than 'catboostregressor', 'xgbregressor', or 'lgbmregressor'
    then early_stop_frac has no effect. The "sufficient number of
    observations" criteria is defined as follows: early stopping will
    happen if
    early_stop_frac*len(train) - lags - 1 >= early_stop_min_samples
    where len(train) is the length of the training set (after logdiff
    if logdiff is applied) and early_stop_min_samples is as described
    below. If you do not want to use early stopping, set this to 0.0,
    which is the default.

    early_stop_min_samples : int.
    Early stopping minimum validation dataset size. For more
    information, read early_stop_frac above.

    early_stop_rounds : int.
    The number of rounds to use for early stopping. For more
    information, read the relevant documentation for CatBoost,
    LightGBM and/or XGBoost.

    scaler_postsplit_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postsplit_1_params'.

    scaler_postsplit_1_params : dict or None.
    The parameters for 'scaler_postsplit_1'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    scaler_postsplit_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postsplit_2_params'.

    scaler_postsplit_2_params : dict or None.
    The parameters for 'scaler_postsplit_2'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    scaler_dm_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the design matrix
    composed of lags of X. The scaling is done using the relevant
    scaler from scikit-learn. Parameters can be set using
    'scaler_dm_1_params'.

    scaler_dm_1_params : dict or None.
    The parameters for 'scaler_dm_1'. The parameters must correspond to
    the relevant scaler's parameters in the scikit-learn package.

    scaler_dm_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the design matrix
    composed of lags of X. The scaling is done using the relevant
    scaler from scikit-learn. Parameters can be set using
    'scaler_dm_2_params'.

    scaler_dm_2_params : dict or None.
    The parameters for 'scaler_dm_2'. The parameters must correspond to
    the relevant scaler's parameters in the scikit-learn package.

    y_bounds_error : one of 'warn', 'raise' or 'ignore'.
    If set to 'warn' and min(test) < min(train) or
    max(test) > max(train), then a warning will be printed. If
    y_bounds_error == 'raise', an exception will be raised. If
    y_bounds_error == 'ignore', no exception will be raised or warning
    printed. This parameter is provided because some regressors, such
    as tree-based regressors, cannot extrapolate (or do so very
    poorly). Setting y_bounds_error to 'warn' or 'raise' would
    immediately warn the user or prevent the analysis from actually
    occuring if the test set is not within the bounds of the training
    set.

    y_bounds_violation_sign_drop : bool.
    If True, observations in the test set whose true values are outside
    [min(train), max(train)] are not used when calculating the test
    statistics and p-values of the sign and Wilcoxon tests (note: this
    also requires y_bounds_error to not be set to 'raise'). If False,
    then the sign and Wilcoxon test statistics and p-values are
    calculated using all observations in the test set.

    regressor : string, or list of strings of length 2.
    If a string, it is the regressor used for both the restricted and
    unrestricted models. If a list of strings, the first string in the
    list is the regressor for the restricted model, and the second one
    is the regressor for the unrestricted model. Popular regressors
    include:
        - 'krr' : kernel ridge regressor
        - 'catboostregressor' : CatBoost regressor
        - 'xgbregressor' : XGBoost regressor
        - 'lgbmregressor' : LightGBM regressor
        - 'randomforestregressor' : random forest regressor
        - 'cuml_randomforestregressor' : random forest regressor using
            the cuML library
        - 'linearregression' : linear regressor
        - 'classic' : linear regressor in the classic sense
            (train == test == all data)
        - 'svr' : Epsilon Support Vector Regression
        - 'nusvr' : Nu Support Vector Regression
        - 'cuml_svr' : Epsilon Support Vector Regression using cuML
        - 'knn' : Regression based on k-nearest neighbors
        - 'gaussianprocessregressor' : Gaussian process regressor
        - 'gradientboostingregressor' : Gradient boost regressor
        - 'histgradientboostingregressor' : Histogram-based Gradient
            Boosting Regression Tree
        - 'default' : kernel ridge regressor with the RBF kernel set as
            default (default)
    Note that you must have the correct library installed in order to
    use these regressors with mlcausality. For instance, if you need to
    use the 'xgbregressor', you must have XGBoost installed, while if
    you need to use 'krr', you must have scikit-learn installed. Note
    that your choice of regressor may affect or override the choices
    you make for other parameters. For instance, choosing
    regressor='classic' overrides your choice for 'train_size' because
    in classic Granger causality the test set is equal to the training
    set.

    regressor_params : dict, list of 2 dicts, or None.
    These are the parameters with which the regressor is initialized.
    For instance, if you want to use the 'rbf' kernel with kernel
    ridge, you could use
    regressor_params={'regressor_params':{'kernel':'rbf'}}. A list of
    2 dicts provides a separate set of parameters for the restricted
    and unrestricted models respectively.

    regressor_fit_params : dict, list of 2 dicts, or None.
    These are the parameters used with the regressor's fit method.

    check_model_type_match : one of 'warn', 'raise' or 'ignore'.
    Checks whether the regressors for the restricted and unrestricted
    models are identical. Note that the matching is done on the strings
    in the 'regressor' list, if a list of 2 strings is suppplied. So,
    for instance, if regressor=['krr','kernelridge'] and
    check_model_type_match='raise' then an error will be raised even
    though 'krr' is an alias for 'kernelridge'. This parameter has no
    effect if 'regressor' is a string instead of a list of 2 strings.

    ftest : bool.
    Whether to perform an f-test identical to the one done in classical
    Granger causality. This really only makes sense if
    regressor='classical' or regressor='linear.'

    normality_tests : bool.
    Whether to perform normality tests on the errors of the restricted
    and unrestricted models. Setting this to True will cause the
    Shapiro-Wilk, the Anderson-Darling, and the Jarque-Bera test
    statistics and p-values to be calculated and reported.

    acorr_tests : bool.
    Whether to perform autocorrelation tests on the errors. Setting
    this to True will cause the Durbin-Watson and Ljung-Box test
    statistics and p-values to be calculated and reported.

    return_restrict_only : bool.
    If True then only the restricted model's predictions, errors etc.
    are returned, and the unrestricted model's corresponding values are
    not returned. This is useful for performance purposes if the
    mlcausality() function is used in a loop and the unrestricted
    values do not have to be reported in every loop run. If you do not
    know what you are doing, set this to False.

    return_inside_bounds_mask : bool.
    Whether to return a mask that indicates whether the label value in
    the test set is within the [min,max] range of the training set.
    This could be useful for some models that do not extrapolate well,
    for instance, tree-based models like random forests.

    return_kwargs_dict : bool.
    Whether to return a dict of all kwargs passed to mlcausality().

    return_preds : bool.
    Whether to return the predictions of the models. If
    return_preds=True and return_restrict_only=True, then only the
    predictions of the restricted model will be returned. Note that if
    any of the "init" type of scalers are used then the preds returned
    are for the data transformed by those "init" scalers.

    return_errors : bool.
    Whether to return the errors of the models. If return_errors=True
    and return_restrict_only=True, then only the errors of the
    restricted model will be returned. Note that if any of the "init"
    type of scalers are used then the errors returned are for the data
    transformed by those "init" scalers.

    return_nanfilled : bool.
    Whether to return preds and errors with nan values in the vector
    corresponding to the positions of the input data that were not in
    the test set. This ensures that the predictions vector, for
    example, has the exact same number of observations as the input
    data. If False then the predictions vector could be shorter than
    the total amount of data if the test set contains only a subset of
    the entire dataset.

    return_models : bool.
    If True instances of the fitted models will be returned.

    return_scalers : bool.
    If True fitted instances of scalers (if used) will be returned.

    return_summary_df : bool.
    Whether to return a summary of the return in a pandas.DataFrame
    format.

    kwargs_in_summary_df : bool.
    If this is True and return_summary_df=True then the kwargs passed
    to mlcausality() will be returned in the summary pandas.DataFrame.

    pretty_print : bool.
    Whether a pretty print of the summary should be outputted to stdout
    following a call to mlcausality(). If set to False then
    mlcausality() will run silently unless a warning needs to be
    printed or an exception raised.
    """
    # Store and parse the dict of passed variables
    if return_kwargs_dict:
        kwargs_dict = locals()
        del kwargs_dict["X"]
        del kwargs_dict["y"]
        if kwargs_dict["split"] is not None:
            kwargs_dict["split"] = "notNone"
        if not isinstance(regressor, str):
            if len(regressor) != 2:
                raise ValueError(
                    "regressor was not a string or list-like of length 2 in mlcausality"
                )
            else:
                kwargs_dict["regressor"] = [
                    str(type(regressor[0])),
                    str(type(regressor[1])),
                ]
    # Initial parameter checks; data scaling; and data splits
    early_stop = False
    if (
        (scaler_init_1 is not None and scaler_init_1.lower() == "normalizer")
        or (
            scaler_init_2 is not None and scaler_init_2.lower() == "normalizer"
        )
        or (
            scaler_prelogdiff_1 is not None
            and scaler_prelogdiff_1.lower() == "normalizer"
        )
        or (
            scaler_prelogdiff_2 is not None
            and scaler_prelogdiff_2.lower() == "normalizer"
        )
        or (
            scaler_postlogdiff_1 is not None
            and scaler_postlogdiff_1.lower() == "normalizer"
        )
        or (
            scaler_postlogdiff_2 is not None
            and scaler_postlogdiff_2.lower() == "normalizer"
        )
        or (
            scaler_postsplit_1 is not None
            and scaler_postsplit_1.lower() == "normalizer"
        )
        or (
            scaler_postsplit_2 is not None
            and scaler_postsplit_2.lower() == "normalizer"
        )
    ):
        raise ValueError(
            "'normalizer' can only be used with 'scaler_dm_1' or 'scaler_dm_2'; do not "
            "use it with other scalers"
        )
    if y is None or lag is None:
        raise TypeError("You must supply y and lag to mlcausality")
    if not isinstance(lag, int):
        raise TypeError("lag was not passed as an int to mlcausality")
    if isinstance(y, (list, tuple)):
        y = np.atleast_2d(y).reshape(-1, 1)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = np.atleast_2d(y.to_numpy()).reshape(-1, 1)
        else:
            y = y.to_numpy()
    if not isinstance(y, np.ndarray):
        raise TypeError(
            "y could not be cast to np.ndarray in mlcausality")
    if len(y.shape) == 1:
        y = np.atleast_2d(y).reshape(-1, 1)
    if (
        regressor.lower() == "gaussianprocessregressor"
        or regressor.lower() == "gpr"
    ):
        y = y.astype(np.float128)
    elif (
        regressor.lower() == "kernelridge"
        or regressor.lower() == "kernelridgeregressor"
        or regressor.lower() == "krr"
    ):
        y = y.astype(np.float64)
    else:
        y = y.astype(np.float32)
    if not return_restrict_only:
        if isinstance(X, (list, tuple)):
            X = np.atleast_2d(X).reshape(-1, 1)
        if isinstance(X, (pd.Series, pd.DataFrame)):
            if len(X.shape) == 1 or X.shape[1] == 1:
                X = np.atleast_2d(X.to_numpy()).reshape(-1, 1)
            else:
                X = X.to_numpy()
        if not isinstance(X, np.ndarray):
            raise TypeError(
                "X could not be cast to np.ndarray in mlcausality")
        if len(X.shape) == 1:
            X = np.atleast_2d(X).reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            print(X.shape)
            print(y.shape)
            raise ValueError(
                "X and y must have the same length in dimension 0"
            )
        if (
            regressor.lower() == "gaussianprocessregressor"
            or regressor.lower() == "gpr"
            or regressor.lower() == "kernelridge"
            or regressor.lower() == "kernelridgeregressor"
            or regressor.lower() == "krr"
        ):
            X = X.astype(np.float64)
        else:
            X = X.astype(np.float32)
    if not isinstance(logdiff, bool):
        raise TypeError("logdiff must be a bool in mlcausality")
    if scaler_init_1 is not None:
        scaler_init_1_dict = {}
        scaler_init_1_dict["y"] = init_scaler(
            scaler=scaler_init_1, scaler_params=scaler_init_1_params
        )
        y = scaler_init_1_dict["y"].fit_transform(y)
        if not return_restrict_only:
            scaler_init_1_dict["X"] = init_scaler(
                scaler=scaler_init_1, scaler_params=scaler_init_1_params
            )
            X = scaler_init_1_dict["X"].fit_transform(X)
    if scaler_init_2 is not None:
        scaler_init_2_dict = {}
        scaler_init_2_dict["y"] = init_scaler(
            scaler=scaler_init_2, scaler_params=scaler_init_2_params
        )
        y = scaler_init_2_dict["y"].fit_transform(y)
        if not return_restrict_only:
            scaler_init_2_dict["X"] = init_scaler(
                scaler=scaler_init_2, scaler_params=scaler_init_2_params
            )
            X = scaler_init_2_dict["X"].fit_transform(X)
    # if train_size == 1:
    #    early_stop_frac = 0.0
    #    split_override = True
    # else:
    #    split_override = False
    split_override = False
    if regressor == "default":
        regressor = "krr"
        regressor_params = {"kernel": "rbf"}
    if regressor == "classic":
        if return_restrict_only:
            raise ValueError(
                "If reggressor is classic, return_restrict_only cannot be True"
            )
        regressor = "linearregression"
        train_size = 1
        split_override = True
        scaler_prelogdiff_1 = None
        scaler_prelogdiff_2 = None
        logdiff = False
        scaler_postlogdiff_1 = None
        scaler_postlogdiff_2 = None
        scaler_dm_1 = None
        scaler_dm_2 = None
        y_bounds_error = "ignore"
        y_bounds_violation_sign_drop = False
        acorr_tests = True
    # ytrue
    ytrue = deepcopy(y)
    if scaler_prelogdiff_1 is not None:
        scaler_prelogdiff_1_dict = {}
        scaler_prelogdiff_1_dict["y"] = init_scaler(
            scaler=scaler_prelogdiff_1,
            scaler_params=scaler_prelogdiff_1_params,
        )
        y_transformed = scaler_prelogdiff_1_dict["y"].fit_transform(y)
        if not return_restrict_only:
            scaler_prelogdiff_1_dict["X"] = init_scaler(
                scaler=scaler_prelogdiff_1,
                scaler_params=scaler_prelogdiff_1_params,
            )
            X_transformed = scaler_prelogdiff_1_dict["X"].fit_transform(
                X)
            data_scaled = np.concatenate(
                [y_transformed, X_transformed], axis=1
            )
        else:
            data_scaled = y_transformed
    else:
        if not return_restrict_only:
            data_scaled = np.concatenate([y, X], axis=1)
        else:
            data_scaled = y
    if scaler_prelogdiff_2 is not None:
        scaler_prelogdiff_2_dict = {}
        scaler_prelogdiff_2_dict["y"] = init_scaler(
            scaler=scaler_prelogdiff_2,
            scaler_params=scaler_prelogdiff_2_params,
        )
        y_transformed = scaler_prelogdiff_2_dict["y"].fit_transform(
            data_scaled[:, : y.shape[1]]
        )
        if not return_restrict_only:
            scaler_prelogdiff_2_dict["X"] = init_scaler(
                scaler=scaler_prelogdiff_2,
                scaler_params=scaler_prelogdiff_2_params,
            )
            X_transformed = scaler_prelogdiff_2_dict["X"].fit_transform(
                data_scaled[:, y.shape[1]:]
            )
            data_scaled = np.concatenate(
                [y_transformed, X_transformed], axis=1
            )
        else:
            data_scaled = y_transformed
    # Logdiff
    if logdiff:
        # Store outcome variable with current transformations.
        # This is needed to reverse the logdiff operation later when generating preds
        prelogdiff_outcome = deepcopy(data_scaled[:, [0]])
        data_scaled = np.diff(np.log(data_scaled), axis=0)
    if scaler_postlogdiff_1 is not None:
        scaler_postlogdiff_1_dict = {}
        scaler_postlogdiff_1_dict["y"] = init_scaler(
            scaler=scaler_postlogdiff_1,
            scaler_params=scaler_postlogdiff_1_params,
        )
        y_transformed = scaler_postlogdiff_1_dict["y"].fit_transform(
            data_scaled[:, : y.shape[1]]
        )
        if not return_restrict_only:
            scaler_postlogdiff_1_dict["X"] = init_scaler(
                scaler=scaler_postlogdiff_1,
                scaler_params=scaler_postlogdiff_1_params,
            )
            X_transformed = scaler_postlogdiff_1_dict["X"].fit_transform(
                data_scaled[:, y.shape[1]:]
            )
            data_scaled = np.concatenate(
                [y_transformed, X_transformed], axis=1
            )
        else:
            data_scaled = y_transformed
    if scaler_postlogdiff_2 is not None:
        scaler_postlogdiff_2_dict = {}
        scaler_postlogdiff_2_dict["y"] = init_scaler(
            scaler=scaler_postlogdiff_2,
            scaler_params=scaler_postlogdiff_2_params,
        )
        y_transformed = scaler_postlogdiff_2_dict["y"].fit_transform(
            data_scaled[:, : y.shape[1]]
        )
        if not return_restrict_only:
            scaler_postlogdiff_2_dict["X"] = init_scaler(
                scaler=scaler_postlogdiff_2,
                scaler_params=scaler_postlogdiff_2_params,
            )
            X_transformed = scaler_postlogdiff_2_dict["X"].fit_transform(
                data_scaled[:, y.shape[1]:]
            )
            data_scaled = np.concatenate(
                [y_transformed, X_transformed], axis=1
            )
        else:
            data_scaled = y_transformed
    if not split_override and split is not None:
        if isinstance(split, types.GeneratorType):
            split = list(split)
        if len(split) != 2:
            raise ValueError(
                "If split is provided to mlcausality, it must be of length 2"
            )
        if logdiff:
            split_new = []
            split_new.append([s - 1 for s in split[0] if s - 1 >= 0])
            split_new.append([s - 1 for s in split[1] if s - 1 >= 0])
            split = split_new
        train = data_scaled[split[0], :]
        test = data_scaled[split[1], :]
    elif train_size == 1:
        train = data_scaled.copy()
        test = data_scaled.copy()
    elif isinstance(train_size, int) and train_size != 0 and train_size != 1:
        if logdiff and train_size < lag + 2:
            raise ValueError(
                "train_size is too small, resulting in no samples in the train set!"
            )
        elif logdiff and train_size > y.shape[0] - lag - 2:
            raise ValueError(
                "train_size is too large, resulting in no samples in the test set!"
            )
        elif not logdiff and train_size < lag + 1:
            raise ValueError(
                "train_size is too small, resulting in no samples in the train set!"
            )
        elif not logdiff and train_size > y.shape[0] - lag - 1:
            raise ValueError(
                "train_size is too large, resulting in no samples in the test set!"
            )
        if logdiff:
            train = data_scaled[: train_size - 1, :]
            test = data_scaled[train_size - 1:, :]
        else:
            train = data_scaled[:train_size, :]
            test = data_scaled[train_size:, :]
    elif isinstance(train_size, float):
        if train_size <= 0 or train_size > 1:
            raise ValueError(
                "train_size is a float that is not between (0,1] in mlcausality"
            )
        elif logdiff and round(train_size * y.shape[0]) - lag - 2 < 0:
            raise ValueError(
                "train_size is a float that is too small resulting in no samples in train"
            )
        elif logdiff and round((1 - train_size) * y.shape[0]) - lag - 2 < 0:
            raise ValueError(
                "train_size is a float that is too large resulting in no samples in test"
            )
        elif not logdiff and round(train_size * y.shape[0]) - lag - 1 < 0:
            raise ValueError(
                "train_size is a float that is too small resulting in no samples in train"
            )
        elif (
            not logdiff and round(
                (1 - train_size) * y.shape[0]) - lag - 1 < 0
        ):
            raise ValueError(
                "train_size is a float that is too large resulting in no samples in test"
            )
        else:
            if logdiff:
                train = data_scaled[: round(
                    train_size * y.shape[0]) - 1, :]
                test = data_scaled[round(
                    train_size * y.shape[0]) - 1:, :]
            else:
                train = data_scaled[: round(
                    train_size * y.shape[0]), :]
                test = data_scaled[round(train_size * y.shape[0]):, :]
    else:
        raise TypeError(
            'train_size must be provided as a float or int to mlcausality. Alternatively, '
            'you can provide a split to "split".'
        )
    # Regressors
    if regressor_fit_params is None:
        regressor_fit_params_restrict = {}
        regressor_fit_params_unrestrict = {}
    elif isinstance(regressor_fit_params, dict):
        regressor_fit_params_restrict = regressor_fit_params
        regressor_fit_params_unrestrict = regressor_fit_params
    elif isinstance(regressor_fit_params, list):
        if (
            len(regressor_fit_params) != 2
            or not isinstance(regressor_fit_params[0], dict)
            or not isinstance(regressor_fit_params[1], dict)
        ):
            raise ValueError(
                "regressor_fit_params must be None, a dict, or a list of 2 dicts"
            )
        else:
            regressor_fit_params_restrict = regressor_fit_params[0]
            regressor_fit_params_unrestrict = regressor_fit_params[1]
    if not isinstance(regressor, str):
        if len(regressor) != 2:
            raise ValueError(
                "regressor was not a string or list-like of length 2 in mlcausality"
            )
        elif check_model_type_match == "raise" and type(regressor[0]) != type(
            regressor[1]
        ):
            raise TypeError(
                'regressors passed for the restricted and unrestricted models are of '
                'different types. This does not really make much sense for the purposes '
                'of Granger causality testing because the performance of different types '
                'of regressors could be vastly different, which could lead to erroneous '
                'conclusions regarding Granger causality. If you know what you are doing, '
                'you can re-run with check_model_type_match="warn" or '
                'check_model_type_match="ignore"'
            )
        elif check_model_type_match == "warn" and type(regressor[0]) != type(
            regressor[1]
        ):
            warnings.warn(
                "regressors passed for the restricted and unrestricted models are of "
                "different types."
            )
            model_restrict = regressor[0]
            model_unrestrict = regressor[1]
        else:
            model_restrict = regressor[0]
            model_unrestrict = regressor[1]
    else:
        if regressor_params is not None:
            if not isinstance(regressor_params, (dict, list)):
                raise TypeError(
                    "regressor_params have to be one of None, dict, or list of 2 dicts"
                )
            elif isinstance(regressor_params, list):
                if (
                    not len(regressor_params) == 2
                    and not isinstance(regressor_params[0], dict)
                    and not isinstance(regressor_params[1], dict)
                ):
                    raise TypeError(
                        "regressor_params have to be one of None, dict, or list of 2 dicts"
                    )
                else:
                    params_restrict = regressor_params[0]
                    params_unrestrict = regressor_params[1]
            else:
                params_restrict = regressor_params
                params_unrestrict = regressor_params
        else:
            params_restrict = {}
            params_unrestrict = {}
        if regressor.lower() in [
            "catboostregressor",
            "xgbregressor",
            "lgbmregressor",
        ]:
            if (
                not isinstance(early_stop_frac, float)
                or early_stop_frac < 0
                or early_stop_frac >= 1
            ):
                raise ValueError(
                    "early_stop_frac must be a float in [0,1) if regressor.lower() in "
                    "['catboostregressor', 'xgbregressor', 'lgbmregressor', "
                    "'gradientboostingregressor','histgradientboostingregressor']"
                )
            if not isinstance(early_stop_min_samples, int):
                raise TypeError(
                    "early_stop_min_samples must be an int")
            # if we have less than early_stop_min_samples samples for validation, do not
            # use early stopping. Otherwise, use early stopping
            if (
                logdiff
                and round(early_stop_frac * (train.shape[0] + 1))
                - lag
                - 1
                - early_stop_min_samples
                < 0
            ):
                early_stop = False
            elif (
                not logdiff
                and round(early_stop_frac * train.shape[0])
                - lag
                - early_stop_min_samples
                < 0
            ):
                early_stop = False
            else:
                early_stop = True
            if early_stop:
                if logdiff:
                    val = deepcopy(
                        train[
                            round((1 - early_stop_frac)
                                  * (train.shape[0] + 1))
                            - 1:,
                            :,
                        ]
                    )
                    train = deepcopy(
                        train[
                            : round(
                                (1 - early_stop_frac) *
                                (train.shape[0] + 1)
                            )
                            - 1,
                            :,
                        ]
                    )
                else:
                    val = deepcopy(
                        train[
                            round((1 - early_stop_frac) * train.shape[0]):, :
                        ]
                    )
                    train = deepcopy(
                        train[
                            : round((1 - early_stop_frac) * train.shape[0]), :
                        ]
                    )
        if regressor.lower() == "catboostregressor":
            from catboost import CatBoostRegressor

            if early_stop:
                params_restrict.update(
                    {"early_stopping_rounds": early_stop_rounds}
                )
                params_unrestrict.update(
                    {"early_stopping_rounds": early_stop_rounds}
                )
            model_restrict = CatBoostRegressor(**params_restrict)
            model_unrestrict = CatBoostRegressor(**params_unrestrict)
        elif regressor.lower() == "xgbregressor":
            from xgboost import XGBRegressor

            if early_stop:
                params_restrict.update(
                    {"early_stopping_rounds": early_stop_rounds}
                )
                params_unrestrict.update(
                    {"early_stopping_rounds": early_stop_rounds}
                )
            model_restrict = XGBRegressor(**params_restrict)
            model_unrestrict = XGBRegressor(**params_unrestrict)
        elif regressor.lower() == "lgbmregressor":
            from lightgbm import LGBMRegressor

            model_restrict = LGBMRegressor(**params_restrict)
            model_unrestrict = LGBMRegressor(**params_unrestrict)
        elif regressor.lower() == "linearregression":
            ftest = True
            from sklearn.linear_model import LinearRegression

            model_restrict = LinearRegression(**params_restrict)
            model_unrestrict = LinearRegression(**params_unrestrict)
        elif regressor.lower() == "randomforestregressor":
            from sklearn.ensemble import RandomForestRegressor

            model_restrict = RandomForestRegressor(**params_restrict)
            model_unrestrict = RandomForestRegressor(
                **params_unrestrict)
        elif regressor.lower() == "svr":
            from sklearn.svm import SVR

            model_restrict = SVR(**params_restrict)
            model_unrestrict = SVR(**params_unrestrict)
        elif regressor.lower() == "nusvr":
            from sklearn.svm import NuSVR

            model_restrict = NuSVR(**params_restrict)
            model_unrestrict = NuSVR(**params_unrestrict)
        elif (
            regressor.lower() == "gaussianprocessregressor"
            or regressor.lower() == "gpr"
        ):
            from sklearn.gaussian_process import GaussianProcessRegressor

            model_restrict = GaussianProcessRegressor(
                **params_restrict)
            model_unrestrict = GaussianProcessRegressor(
                **params_unrestrict)
        elif (
            regressor.lower() == "kernelridge"
            or regressor.lower() == "kernelridgeregressor"
            or regressor.lower() == "krr"
        ):
            from sklearn.kernel_ridge import KernelRidge

            model_restrict = KernelRidge(**params_restrict)
            model_unrestrict = KernelRidge(**params_unrestrict)
        elif (
            regressor.lower() == "kneighborsregressor"
            or regressor.lower() == "knn"
        ):
            from sklearn.neighbors import KNeighborsRegressor

            model_restrict = KNeighborsRegressor(**params_restrict)
            model_unrestrict = KNeighborsRegressor(**params_unrestrict)
        elif regressor.lower() == "gradientboostingregressor":
            from sklearn.ensemble import GradientBoostingRegressor

            model_restrict = GradientBoostingRegressor(
                **params_restrict)
            model_unrestrict = GradientBoostingRegressor(
                **params_unrestrict)
        elif regressor.lower() == "histgradientboostingregressor":
            from sklearn.ensemble import HistGradientBoostingRegressor

            model_restrict = HistGradientBoostingRegressor(
                **params_restrict)
            model_unrestrict = HistGradientBoostingRegressor(
                **params_unrestrict
            )
        elif regressor.lower() == "cuml_svr":
            from cuml.svm import SVR

            model_restrict = SVR(**params_restrict)
            model_unrestrict = SVR(**params_unrestrict)
        elif regressor.lower() == "cuml_randomforestregressor":
            from cuml.ensemble import RandomForestRegressor

            model_restrict = RandomForestRegressor(**params_restrict)
            model_unrestrict = RandomForestRegressor(
                **params_unrestrict)
        else:
            raise ValueError(
                "unidentified string regressor passed to mlcausality"
            )
    train_integ = train
    test_integ = test
    if early_stop:
        val_integ = val
    if logdiff:
        test_integ = test_integ[1:, :]
        if early_stop:
            val_integ = val_integ[1:, :]
    # scaler_postsplit_1
    if scaler_postsplit_1 is not None:
        scaler_postsplit_1_dict = {}
        scaler_postsplit_1_dict["y"] = init_scaler(
            scaler=scaler_postsplit_1, scaler_params=scaler_postsplit_1_params
        )
        train_integ[:, : y.shape[1]] = scaler_postsplit_1_dict[
            "y"
        ].fit_transform(train_integ[:, : y.shape[1]])
        test_integ[:, : y.shape[1]] = scaler_postsplit_1_dict["y"].transform(
            test_integ[:, : y.shape[1]]
        )
        if early_stop:
            val_integ[:, : y.shape[1]] = scaler_postsplit_1_dict[
                "y"
            ].transform(val_integ[:, : y.shape[1]])
        if not return_restrict_only:
            scaler_postsplit_1_dict["X"] = init_scaler(
                scaler=scaler_postsplit_1,
                scaler_params=scaler_postsplit_1_params,
            )
            train_integ[:, y.shape[1]:] = scaler_postsplit_1_dict[
                "X"
            ].fit_transform(train_integ[:, y.shape[1]:])
            test_integ[:, y.shape[1]:] = scaler_postsplit_1_dict[
                "X"
            ].transform(test_integ[:, y.shape[1]:])
            if early_stop:
                val_integ[:, y.shape[1]:] = scaler_postsplit_1_dict[
                    "X"
                ].transform(val_integ[:, y.shape[1]:])
    # scaler_postsplit_2
    if scaler_postsplit_2 is not None:
        scaler_postsplit_2_dict = {}
        scaler_postsplit_2_dict["y"] = init_scaler(
            scaler=scaler_postsplit_2, scaler_params=scaler_postsplit_2_params
        )
        train_integ[:, : y.shape[1]] = scaler_postsplit_2_dict[
            "y"
        ].fit_transform(train_integ[:, : y.shape[1]])
        test_integ[:, : y.shape[1]] = scaler_postsplit_2_dict["y"].transform(
            test_integ[:, : y.shape[1]]
        )
        if early_stop:
            val_integ[:, : y.shape[1]] = scaler_postsplit_2_dict[
                "y"
            ].transform(val_integ[:, : y.shape[1]])
        if not return_restrict_only:
            scaler_postsplit_2_dict["X"] = init_scaler(
                scaler=scaler_postsplit_2,
                scaler_params=scaler_postsplit_2_params,
            )
            train_integ[:, y.shape[1]:] = scaler_postsplit_2_dict[
                "X"
            ].fit_transform(train_integ[:, y.shape[1]:])
            test_integ[:, y.shape[1]:] = scaler_postsplit_2_dict[
                "X"
            ].transform(test_integ[:, y.shape[1]:])
            if early_stop:
                val_integ[:, y.shape[1]:] = scaler_postsplit_2_dict[
                    "X"
                ].transform(val_integ[:, y.shape[1]:])
    # y bounds error
    if y_bounds_error == "raise":
        if np.nanmax(train_integ[lag:, 0]) < np.nanmax(
            test_integ[lag:, 0]
        ) or np.nanmin(train_integ[lag:, 0]) > np.nanmin(test_integ[lag:, 0]):
            raise ValueError(
                '[y_test_min,y_test_max] is not a subset of [y_train_min,y_train_max]. '
                'Since many algorithms, especially tree-based algorithms, cannot '
                'extrapolate, this could result in erroneous conclusions regarding '
                'Granger causality. If you would still like to perform the Granger '
                'causality test anyway, re-run mlcausality with y_bounds_error set to '
                'either "warn" or "ignore".'
            )
    elif y_bounds_error == "warn":
        if np.nanmax(train_integ[lag:, 0]) < np.nanmax(
            test_integ[lag:, 0]
        ) or np.nanmin(train_integ[lag:, 0]) > np.nanmin(test_integ[lag:, 0]):
            warnings.warn(
                "[y_test_min,y_test_max] is not a subset of [y_train_min,y_train_max]. "
                "Since many algorithms, especially tree-based algorithms, cannot "
                "extrapolate, this could result in erroneous conclusions regarding "
                "Granger causality."
            )
    # y bounds indicies and fractions
    inside_bounds_mask_init = np.logical_and(
        test_integ[lag:, 0] >= np.nanmin(train_integ[lag:, 0]),
        test_integ[lag:, 0] <= np.nanmax(train_integ[lag:, 0]),
    )
    inside_bounds_idx = np.where(inside_bounds_mask_init)[0].flatten()
    outside_bounds_frac = (
        test_integ[lag:, 0].shape[0] - inside_bounds_idx.shape[0]
    ) / test_integ[lag:, 0].shape[0]
    if return_inside_bounds_mask:
        inside_bounds_mask = inside_bounds_mask_init.astype(
            float).flatten()
        inside_bounds_mask[inside_bounds_mask == 0] = np.nan
    # Sliding window views
    # Lag+1 gives lag features plus the target column
    train_sw = sliding_window_view(
        train_integ, [lag + 1, data_scaled.shape[1]]
    )
    # Lag+1 gives lag features plus the target column
    test_sw = sliding_window_view(
        test_integ, [lag + 1, data_scaled.shape[1]])
    if early_stop:
        # Lag+1 gives lag features plus the target column
        val_sw = sliding_window_view(
            val_integ, [lag + 1, data_scaled.shape[1]]
        )
    # Reshape data
    train_sw_reshape_restrict = train_sw[:, :, :, : y.shape[1]].reshape(
        train_sw[:, :, :, : y.shape[1]].shape[0],
        train_sw[:, :, :, : y.shape[1]].shape[1]
        * train_sw[:, :, :, : y.shape[1]].shape[2]
        * train_sw[:, :, :, : y.shape[1]].shape[3],
    )
    test_sw_reshape_restrict = test_sw[:, :, :, : y.shape[1]].reshape(
        test_sw[:, :, :, : y.shape[1]].shape[0],
        test_sw[:, :, :, : y.shape[1]].shape[1]
        * test_sw[:, :, :, : y.shape[1]].shape[2]
        * test_sw[:, :, :, : y.shape[1]].shape[3],
    )
    if early_stop:
        val_sw_reshape_restrict = val_sw[:, :, :, : y.shape[1]].reshape(
            val_sw[:, :, :, : y.shape[1]].shape[0],
            val_sw[:, :, :, : y.shape[1]].shape[1]
            * val_sw[:, :, :, : y.shape[1]].shape[2]
            * val_sw[:, :, :, : y.shape[1]].shape[3],
        )
    if not return_restrict_only:
        train_sw_reshape_unrestrict = train_sw.reshape(
            train_sw.shape[0],
            train_sw.shape[1] * train_sw.shape[2] * train_sw.shape[3],
        )
        test_sw_reshape_unrestrict = test_sw.reshape(
            test_sw.shape[0],
            test_sw.shape[1] * test_sw.shape[2] * test_sw.shape[3],
        )
        if early_stop:
            val_sw_reshape_unrestrict = val_sw.reshape(
                val_sw.shape[0],
                val_sw.shape[1] * val_sw.shape[2] * val_sw.shape[3],
            )
    # Design matrix scalers: restricted model
    if scaler_dm_1 is not None:
        scaler_dm_1_dict = {}
        scaler_dm_1_dict["restricted"] = init_scaler(
            scaler=scaler_dm_1, scaler_params=scaler_dm_1_params
        )
        X_train_dm_restricted = scaler_dm_1_dict["restricted"].fit_transform(
            train_sw_reshape_restrict[:, : -y.shape[1]]
        )
        X_test_dm_restricted = scaler_dm_1_dict["restricted"].transform(
            test_sw_reshape_restrict[:, : -y.shape[1]]
        )
        if early_stop:
            X_val_dm_restricted = scaler_dm_1_dict["restricted"].transform(
                val_sw_reshape_restrict[:, : -y.shape[1]]
            )
    else:
        X_train_dm_restricted = train_sw_reshape_restrict[:,
                                                          : -y.shape[1]]
        X_test_dm_restricted = test_sw_reshape_restrict[:,
                                                        : -y.shape[1]]
        if early_stop:
            X_val_dm_restricted = val_sw_reshape_restrict[:,
                                                          : -y.shape[1]]
    if scaler_dm_2 is not None:
        scaler_dm_2_dict = {}
        scaler_dm_2_dict["restricted"] = init_scaler(
            scaler=scaler_dm_2, scaler_params=scaler_dm_2_params
        )
        X_train_dm_restricted = scaler_dm_2_dict["restricted"].fit_transform(
            X_train_dm_restricted
        )
        X_test_dm_restricted = scaler_dm_2_dict["restricted"].transform(
            X_test_dm_restricted
        )
        if early_stop:
            X_val_dm_restricted = scaler_dm_2_dict["restricted"].transform(
                X_val_dm_restricted
            )
    if not return_restrict_only:
        # Design matrix scalers: unrestricted model
        if scaler_dm_1 is not None:
            scaler_dm_1_dict = {}
            scaler_dm_1_dict["unrestricted"] = init_scaler(
                scaler=scaler_dm_1, scaler_params=scaler_dm_1_params
            )
            X_train_dm_unrestricted = scaler_dm_1_dict[
                "unrestricted"
            ].fit_transform(
                train_sw_reshape_unrestrict[:, : -data_scaled.shape[1]]
            )
            X_test_dm_unrestricted = scaler_dm_1_dict[
                "unrestricted"
            ].transform(test_sw_reshape_unrestrict[:, : -data_scaled.shape[1]])
            if early_stop:
                X_val_dm_unrestricted = scaler_dm_1_dict[
                    "unrestricted"
                ].transform(
                    val_sw_reshape_unrestrict[:,
                                              : -data_scaled.shape[1]]
                )
        else:
            X_train_dm_unrestricted = train_sw_reshape_unrestrict[
                :, : -data_scaled.shape[1]
            ]
            X_test_dm_unrestricted = test_sw_reshape_unrestrict[
                :, : -data_scaled.shape[1]
            ]
            if early_stop:
                X_val_dm_unrestricted = val_sw_reshape_unrestrict[
                    :, : -data_scaled.shape[1]
                ]
        if scaler_dm_2 is not None:
            scaler_dm_2_dict = {}
            scaler_dm_2_dict["unrestricted"] = init_scaler(
                scaler=scaler_dm_2, scaler_params=scaler_dm_2_params
            )
            X_train_dm_unrestricted = scaler_dm_2_dict[
                "unrestricted"
            ].fit_transform(X_train_dm_unrestricted)
            X_test_dm_unrestricted = scaler_dm_2_dict[
                "unrestricted"
            ].transform(X_test_dm_unrestricted)
            if early_stop:
                X_val_dm_unrestricted = scaler_dm_2_dict[
                    "unrestricted"
                ].transform(X_val_dm_unrestricted)
    # Handle early stopping
    if (
        isinstance(regressor, str)
        and regressor.lower() in ["catboostregressor", "xgbregressor"]
        and early_stop
    ):
        regressor_fit_params_restrict.update(
            {
                "eval_set": [
                    (
                        X_val_dm_restricted,
                        val_sw_reshape_restrict[:, -y.shape[1]],
                    )
                ]
            }
        )
        if not return_restrict_only:
            regressor_fit_params_unrestrict.update(
                {
                    "eval_set": [
                        (
                            X_val_dm_unrestricted,
                            val_sw_reshape_unrestrict[
                                :, -data_scaled.shape[1]
                            ],
                        )
                    ]
                }
            )
    elif (
        isinstance(regressor, str)
        and regressor.lower() == "lgbmregressor"
        and early_stop
    ):
        import lightgbm

        if "verbose" in params_restrict.keys():
            lgbm_restrict_verbosity = params_restrict["verbose"]
        else:
            lgbm_restrict_verbosity = True
        lgbm_early_stopping_callback_restrict = lightgbm.early_stopping(
            early_stop_rounds,
            first_metric_only=True,
            verbose=lgbm_restrict_verbosity,
        )
        regressor_fit_params_restrict.update(
            {
                "callbacks": [lgbm_early_stopping_callback_restrict],
                "eval_set": [
                    (
                        X_val_dm_restricted,
                        val_sw_reshape_restrict[:, -y.shape[1]],
                    )
                ],
            }
        )
        if not return_restrict_only:
            if "verbose" in params_unrestrict.keys():
                lgbm_unrestrict_verbosity = params_unrestrict["verbose"]
            else:
                lgbm_unrestrict_verbosity = True
            lgbm_early_stopping_callback_unrestrict = lightgbm.early_stopping(
                early_stop_rounds,
                first_metric_only=True,
                verbose=lgbm_unrestrict_verbosity,
            )
            regressor_fit_params_unrestrict.update(
                {
                    "callbacks": [lgbm_early_stopping_callback_unrestrict],
                    "eval_set": [
                        (
                            X_val_dm_unrestricted,
                            val_sw_reshape_unrestrict[
                                :, -data_scaled.shape[1]
                            ],
                        )
                    ],
                }
            )
    # Fit restricted model
    model_restrict.fit(
        X_train_dm_restricted,
        train_sw_reshape_restrict[:, -y.shape[1]],
        **regressor_fit_params_restrict,
    )
    preds_restrict = model_restrict.predict(
        X_test_dm_restricted).flatten()
    if not return_restrict_only:
        # Fit unrestricted model
        model_unrestrict.fit(
            X_train_dm_unrestricted,
            train_sw_reshape_unrestrict[:, -data_scaled.shape[1]],
            **regressor_fit_params_unrestrict,
        )
        preds_unrestrict = model_unrestrict.predict(
            X_test_dm_unrestricted
        ).flatten()
    # Transform preds and ytrue if transformations were originally applied
    if not split_override and split is not None:
        if logdiff:
            split_unadj = [i + 1 for i in split[1]]
            split_unadj = split_unadj[1:]
        else:
            split_unadj = split[1]
        ytrue = ytrue[split_unadj, [0]]
        ytrue = ytrue[lag:]
    else:
        ytrue = ytrue[-preds_restrict.shape[0]:, [0]]
    if scaler_postsplit_2 is not None:
        if y.shape[1] > 1:
            preds_restrict_for_scaler_postsplit_2 = np.concatenate(
                [
                    preds_restrict.reshape(-1, 1),
                    np.zeros_like(y[: preds_restrict.shape[0], 1:]),
                ],
                axis=1,
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postsplit_2 = np.concatenate(
                    [
                        preds_unrestrict.reshape(-1, 1),
                        np.zeros_like(
                            y[: preds_unrestrict.shape[0], 1:]),
                    ],
                    axis=1,
                )
        else:
            preds_restrict_for_scaler_postsplit_2 = preds_restrict.reshape(
                -1, 1
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postsplit_2 = (
                    preds_unrestrict.reshape(-1, 1)
                )
        preds_restrict = (
            scaler_postsplit_2_dict["y"]
            .inverse_transform(preds_restrict_for_scaler_postsplit_2)[:, 0]
            .flatten()
        )
        if not return_restrict_only:
            preds_unrestrict = (
                scaler_postsplit_2_dict["y"]
                .inverse_transform(preds_unrestrict_for_scaler_postsplit_2)[
                    :, 0
                ]
                .flatten()
            )
    if scaler_postsplit_1 is not None:
        if y.shape[1] > 1:
            preds_restrict_for_scaler_postsplit_1 = np.concatenate(
                [
                    preds_restrict.reshape(-1, 1),
                    np.zeros_like(y[: preds_restrict.shape[0], 1:]),
                ],
                axis=1,
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postsplit_1 = np.concatenate(
                    [
                        preds_unrestrict.reshape(-1, 1),
                        np.zeros_like(
                            y[: preds_unrestrict.shape[0], 1:]),
                    ],
                    axis=1,
                )
        else:
            preds_restrict_for_scaler_postsplit_1 = preds_restrict.reshape(
                -1, 1
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postsplit_1 = (
                    preds_unrestrict.reshape(-1, 1)
                )
        preds_restrict = (
            scaler_postsplit_1_dict["y"]
            .inverse_transform(preds_restrict_for_scaler_postsplit_1)[:, 0]
            .flatten()
        )
        if not return_restrict_only:
            preds_unrestrict = (
                scaler_postsplit_1_dict["y"]
                .inverse_transform(preds_unrestrict_for_scaler_postsplit_1)[
                    :, 0
                ]
                .flatten()
            )
    if scaler_postlogdiff_2 is not None:
        if y.shape[1] > 1:
            preds_restrict_for_scaler_postlogdiff_2 = np.concatenate(
                [
                    preds_restrict.reshape(-1, 1),
                    np.zeros_like(y[: preds_restrict.shape[0], 1:]),
                ],
                axis=1,
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postlogdiff_2 = np.concatenate(
                    [
                        preds_unrestrict.reshape(-1, 1),
                        np.zeros_like(
                            y[: preds_unrestrict.shape[0], 1:]),
                    ],
                    axis=1,
                )
        else:
            preds_restrict_for_scaler_postlogdiff_2 = preds_restrict.reshape(
                -1, 1
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postlogdiff_2 = (
                    preds_unrestrict.reshape(-1, 1)
                )
        preds_restrict = (
            scaler_postlogdiff_2_dict["y"]
            .inverse_transform(preds_restrict_for_scaler_postlogdiff_2)[:, 0]
            .flatten()
        )
        if not return_restrict_only:
            preds_unrestrict = (
                scaler_postlogdiff_2_dict["y"]
                .inverse_transform(preds_unrestrict_for_scaler_postlogdiff_2)[
                    :, 0
                ]
                .flatten()
            )
    if scaler_postlogdiff_1 is not None:
        if y.shape[1] > 1:
            preds_restrict_for_scaler_postlogdiff_1 = np.concatenate(
                [
                    preds_restrict.reshape(-1, 1),
                    np.zeros_like(y[: preds_restrict.shape[0], 1:]),
                ],
                axis=1,
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postlogdiff_1 = np.concatenate(
                    [
                        preds_unrestrict.reshape(-1, 1),
                        np.zeros_like(
                            y[: preds_unrestrict.shape[0], 1:]),
                    ],
                    axis=1,
                )
        else:
            preds_restrict_for_scaler_postlogdiff_1 = preds_restrict.reshape(
                -1, 1
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_postlogdiff_1 = (
                    preds_unrestrict.reshape(-1, 1)
                )
        preds_restrict = (
            scaler_postlogdiff_1_dict["y"]
            .inverse_transform(preds_restrict_for_scaler_postlogdiff_1)[:, 0]
            .flatten()
        )
        if not return_restrict_only:
            preds_unrestrict = (
                scaler_postlogdiff_1_dict["y"]
                .inverse_transform(preds_unrestrict_for_scaler_postlogdiff_1)[
                    :, 0
                ]
                .flatten()
            )
    if logdiff:
        if not split_override and split is not None:
            prelogdiff_mult = prelogdiff_outcome[split[1]]
            prelogdiff_mult = prelogdiff_mult[lag + 1:].flatten()
        else:
            prelogdiff_mult = prelogdiff_outcome[
                -preds_restrict.shape[0] - 1: -1, 0
            ].flatten()
        preds_restrict = (np.exp(preds_restrict) *
                          prelogdiff_mult).flatten()
        if not return_restrict_only:
            preds_unrestrict = (
                np.exp(preds_unrestrict) * prelogdiff_mult
            ).flatten()
    if scaler_prelogdiff_2 is not None:
        if y.shape[1] > 1:
            preds_restrict_for_scaler_prelogdiff_2 = np.concatenate(
                [
                    preds_restrict.reshape(-1, 1),
                    np.zeros_like(y[: preds_restrict.shape[0], 1:]),
                ],
                axis=1,
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_prelogdiff_2 = np.concatenate(
                    [
                        preds_unrestrict.reshape(-1, 1),
                        np.zeros_like(
                            y[: preds_unrestrict.shape[0], 1:]),
                    ],
                    axis=1,
                )
        else:
            preds_restrict_for_scaler_prelogdiff_2 = preds_restrict.reshape(
                -1, 1
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_prelogdiff_2 = (
                    preds_unrestrict.reshape(-1, 1)
                )
        preds_restrict = (
            scaler_prelogdiff_2_dict["y"]
            .inverse_transform(preds_restrict_for_scaler_prelogdiff_2)[:, 0]
            .flatten()
        )
        if not return_restrict_only:
            preds_unrestrict = (
                scaler_prelogdiff_2_dict["y"]
                .inverse_transform(preds_unrestrict_for_scaler_prelogdiff_2)[
                    :, 0
                ]
                .flatten()
            )
    if scaler_prelogdiff_1 is not None:
        if y.shape[1] > 1:
            preds_restrict_for_scaler_prelogdiff_1 = np.concatenate(
                [
                    preds_restrict.reshape(-1, 1),
                    np.zeros_like(y[: preds_restrict.shape[0], 1:]),
                ],
                axis=1,
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_prelogdiff_1 = np.concatenate(
                    [
                        preds_unrestrict.reshape(-1, 1),
                        np.zeros_like(
                            y[: preds_unrestrict.shape[0], 1:]),
                    ],
                    axis=1,
                )
        else:
            preds_restrict_for_scaler_prelogdiff_1 = preds_restrict.reshape(
                -1, 1
            )
            if not return_restrict_only:
                preds_unrestrict_for_scaler_prelogdiff_1 = (
                    preds_unrestrict.reshape(-1, 1)
                )
        preds_restrict = (
            scaler_prelogdiff_1_dict["y"]
            .inverse_transform(preds_restrict_for_scaler_prelogdiff_1)[:, 0]
            .flatten()
        )
        if not return_restrict_only:
            preds_unrestrict = (
                scaler_prelogdiff_1_dict["y"]
                .inverse_transform(preds_unrestrict_for_scaler_prelogdiff_1)[
                    :, 0
                ]
                .flatten()
            )
    # Calculate errors
    errors_restrict = preds_restrict - ytrue.flatten()
    if not return_restrict_only:
        errors_unrestrict = preds_unrestrict - ytrue.flatten()
        if y_bounds_violation_sign_drop:
            error_delta = np.abs(
                errors_restrict[inside_bounds_idx].flatten()
            ) - np.abs(errors_unrestrict[inside_bounds_idx].flatten())
            error_delta_num_positive = (error_delta > 0).sum()
            error_delta_len = error_delta[~np.isnan(
                error_delta)].shape[0]
            sign_test_result = binomtest(
                error_delta_num_positive,
                error_delta_len,
                alternative="greater",
            )
            wilcoxon_abserror = wilcoxon(
                np.abs(errors_restrict[inside_bounds_idx].flatten()),
                np.abs(errors_unrestrict[inside_bounds_idx].flatten()),
                alternative="greater",
                nan_policy="omit",
                zero_method="wilcox",
            )
            wilcoxon_num_preds = (
                errors_restrict[inside_bounds_idx].flatten().shape[0]
            )
        else:
            error_delta = np.abs(errors_restrict.flatten()) - np.abs(
                errors_unrestrict.flatten()
            )
            error_delta_num_positive = (error_delta > 0).sum()
            error_delta_len = error_delta[~np.isnan(
                error_delta)].shape[0]
            sign_test_result = binomtest(
                error_delta_num_positive,
                error_delta_len,
                alternative="greater",
            )
            wilcoxon_abserror = wilcoxon(
                np.abs(errors_restrict.flatten()),
                np.abs(errors_unrestrict.flatten()),
                alternative="greater",
                nan_policy="omit",
                zero_method="wilcox",
            )
            wilcoxon_num_preds = errors_restrict.flatten().shape[0]
        if ftest:
            normality_tests = True
            errors2_restrict = errors_restrict**2
            errors2_unrestrict = errors_unrestrict**2
            f_dfn = lag * y.shape[1]
            f_dfd = (
                errors2_restrict.shape[0]
                - (lag * (y.shape[1] + X.shape[1]))
                - 1
            )
            if f_dfd <= 0:
                f_stat = np.nan
                ftest_p_value = np.nan
            else:
                f_stat = (
                    (errors2_restrict.sum() -
                     errors2_unrestrict.sum()) / f_dfn
                ) / (errors2_unrestrict.sum() / f_dfd)
                ftest_p_value = scipyf.sf(f_stat, f_dfn, f_dfd)
    if normality_tests:
        shapiro_restrict = shapiro(errors_restrict.flatten())
        anderson_restrict = anderson(errors_restrict.flatten())
        jarque_bera_restrict = jarque_bera(
            errors_restrict.flatten(), nan_policy="omit"
        )
        if not return_restrict_only:
            shapiro_unrestrict = shapiro(errors_unrestrict.flatten())
            anderson_unrestrict = anderson(errors_unrestrict.flatten())
            jarque_bera_unrestrict = jarque_bera(
                errors_unrestrict.flatten(), nan_policy="omit"
            )
    if acorr_tests:
        durbin_watson_restricted = durbin_watson(
            errors_restrict.flatten())
        acorr_ljungbox_restricted = acorr_ljungbox(
            errors_restrict.flatten(), auto_lag=True, model_df=lag * y.shape[1]
        )
        if not return_restrict_only:
            durbin_watson_unrestricted = durbin_watson(
                errors_unrestrict.flatten()
            )
            acorr_ljungbox_unrestricted = acorr_ljungbox(
                errors_unrestrict.flatten(),
                auto_lag=True,
                model_df=lag * (y.shape[1] + X.shape[1]),
            )
    if return_nanfilled:
        preds_empty = np.empty(
            [
                y.shape[0] - preds_restrict.shape[0],
            ]
        )
        preds_empty[:] = np.nan
        preds_restrict_nanfilled = np.concatenate(
            [preds_empty, preds_restrict]
        )
        if not return_restrict_only:
            preds_unrestrict_nanfilled = np.concatenate(
                [preds_empty, preds_unrestrict]
            )
        ytrue_nanfilled = y[:, [0]]
    return_dict = {
        "summary": {
            "lag": lag,
            "train_obs": train_integ[:, 0].shape[0],
            "effective_train_obs": train_integ[lag:, 0].shape[0],
            "test_obs": test_integ[:, 0].shape[0],
            "effective_test_obs": test_integ[lag:, 0].shape[0],
        }
    }
    if early_stop:
        return_dict["summary"].update(
            {
                "val_obs": val_integ[:, 0].shape[0],
                "effective_val_obs": val_integ[lag:, 0].shape[0],
            }
        )
    return_dict["summary"].update(
        {"outside_bounds_frac": outside_bounds_frac}
    ),
    if not return_restrict_only:
        return_dict["summary"].update(
            {
                "sign_test": {
                    "statistic": sign_test_result.statistic,
                    "pvalue": sign_test_result.pvalue,
                    "y_bounds_violation_sign_drop": y_bounds_violation_sign_drop,
                    "sign_test_num_preds": wilcoxon_num_preds,
                },
                "wilcoxon": {
                    "statistic": wilcoxon_abserror.statistic,
                    "pvalue": wilcoxon_abserror.pvalue,
                    "y_bounds_violation_sign_drop": y_bounds_violation_sign_drop,
                    "wilcoxon_num_preds": wilcoxon_num_preds,
                },
            }
        )
        if ftest:
            return_dict["summary"].update(
                {
                    "ftest": {
                        "statistic": f_stat,
                        "pvalue": ftest_p_value,
                        "dfn": f_dfn,
                        "dfd": f_dfd,
                    }
                }
            )
    if normality_tests:
        if not return_restrict_only:
            return_dict["summary"].update(
                {
                    "normality_tests": {
                        "shapiro": {
                            "restricted": {
                                "statistic": shapiro_restrict.statistic,
                                "pvalue": shapiro_restrict.pvalue,
                            },
                            "unrestricted": {
                                "statistic": shapiro_unrestrict.statistic,
                                "pvalue": shapiro_unrestrict.pvalue,
                            },
                        },
                        "anderson": {
                            "restricted": {
                                "statistic": anderson_restrict.statistic,
                                "critical_values": anderson_restrict.critical_values,
                                "significance_level": anderson_restrict.significance_level,
                                "fit_result": {
                                    "params": {
                                        "loc": anderson_restrict.fit_result.params.loc,
                                        "scale": anderson_restrict.fit_result.params.scale,
                                    },
                                    "success": anderson_restrict.fit_result.success,
                                    "message": str(
                                        anderson_restrict.fit_result.message
                                    ),
                                },
                            },
                            "unrestricted": {
                                "statistic": anderson_unrestrict.statistic,
                                "critical_values": anderson_unrestrict.critical_values,
                                "significance_level": anderson_unrestrict.significance_level,
                                "fit_result": {
                                    "params": {
                                        "loc": anderson_unrestrict.fit_result.params.loc,
                                        "scale": anderson_unrestrict.fit_result.params.scale,
                                    },
                                    "success": anderson_unrestrict.fit_result.success,
                                    "message": str(
                                        anderson_unrestrict.fit_result.message
                                    ),
                                },
                            },
                        },
                        "jarque_bera": {
                            "restricted": {
                                "statistic": jarque_bera_restrict.statistic,
                                "pvalue": jarque_bera_restrict.pvalue,
                            },
                            "unrestricted": {
                                "statistic": jarque_bera_unrestrict.statistic,
                                "pvalue": jarque_bera_unrestrict.pvalue,
                            },
                        },
                    }
                }
            )
        else:
            return_dict["summary"].update(
                {
                    "normality_tests": {
                        "shapiro": {
                            "restricted": {
                                "statistic": shapiro_restrict.statistic,
                                "pvalue": shapiro_restrict.pvalue,
                            }
                        },
                        "anderson": {
                            "restricted": {
                                "statistic": anderson_restrict.statistic,
                                "critical_values": anderson_restrict.critical_values,
                                "significance_level": anderson_restrict.significance_level,
                                "fit_result": {
                                    "params": {
                                        "loc": anderson_restrict.fit_result.params.loc,
                                        "scale": anderson_restrict.fit_result.params.scale,
                                    },
                                    "success": anderson_restrict.fit_result.success,
                                    "message": str(
                                        anderson_restrict.fit_result.message
                                    ),
                                },
                            }
                        },
                        "jarque_bera": {
                            "restricted": {
                                "statistic": jarque_bera_restrict.statistic,
                                "pvalue": jarque_bera_restrict.pvalue,
                            }
                        },
                    }
                }
            )
    if acorr_tests:
        if not return_restrict_only:
            return_dict["summary"].update(
                {
                    "durbin_watson": {
                        "restricted": durbin_watson_restricted,
                        "unrestricted": durbin_watson_unrestricted,
                    }
                }
            )
            return_dict.update(
                {
                    "ljungbox": {
                        "restricted": acorr_ljungbox_restricted,
                        "unrestricted": acorr_ljungbox_unrestricted,
                    }
                }
            )
        else:
            return_dict["summary"].update(
                {"durbin_watson": {"restricted": durbin_watson_restricted}}
            )
            return_dict.update(
                {"ljungbox": {"restricted": acorr_ljungbox_restricted}}
            )
    if return_summary_df:
        return_dict.update(
            {"summary_df": pd.json_normalize(return_dict["summary"])}
        )
    if return_kwargs_dict:
        return_dict.update({"kwargs_dict": kwargs_dict})
    if return_kwargs_dict and kwargs_in_summary_df:
        kwargs_df = pd.json_normalize(return_dict["kwargs_dict"])
        kwargs_df = kwargs_df.loc[
            [0], [i for i in kwargs_df.columns if i not in ["lag"]]
        ]
        return_dict["summary_df"] = return_dict["summary_df"].loc[
            [0],
            [
                i
                for i in return_dict["summary_df"].columns
                if i not in ["wilcoxon.y_bounds_violation_sign_drop"]
            ],
        ]
        return_dict["summary_df"] = pd.concat(
            [return_dict["summary_df"], kwargs_df], axis=1
        )
    if not return_restrict_only:
        if return_preds:
            return_dict.update(
                {
                    "ytrue": ytrue,
                    "preds": {
                        "restricted": preds_restrict,
                        "unrestricted": preds_unrestrict,
                    },
                }
            )
        if return_nanfilled:
            return_dict.update(
                {
                    "ytrue_nanfilled": ytrue_nanfilled,
                    "preds_nanfilled": {
                        "restricted": preds_restrict_nanfilled,
                        "unrestricted": preds_unrestrict_nanfilled,
                    },
                }
            )
        if return_models:
            return_scalers = True
            return_dict.update(
                {
                    "models": {
                        "restricted": model_restrict,
                        "unrestricted": model_unrestrict,
                    }
                }
            )
        if return_errors:
            return_dict.update(
                {
                    "errors": {
                        "restricted": errors_restrict,
                        "unrestricted": errors_unrestrict,
                    }
                }
            )
    else:
        if return_preds:
            return_dict.update(
                {"ytrue": ytrue, "preds": {"restricted": preds_restrict}}
            )
        if return_nanfilled:
            return_dict.update(
                {
                    "ytrue_nanfilled": ytrue_nanfilled,
                    "preds_nanfilled": {
                        "restricted": preds_restrict_nanfilled
                    },
                }
            )
        if return_models:
            return_scalers = True
            return_dict.update(
                {"models": {"restricted": model_restrict}})
        if return_errors:
            return_dict.update(
                {"errors": {"restricted": errors_restrict}})
    if return_inside_bounds_mask:
        return_dict.update({"inside_bounds_mask": inside_bounds_mask})
    if return_scalers:
        return_dict.update({"scalers": {}})
        if scaler_init_1 is not None:
            return_dict["scalers"].update(
                {"scaler_init_1": scaler_init_1_dict}
            )
        if scaler_init_2 is not None:
            return_dict["scalers"].update(
                {"scaler_init_2": scaler_init_2_dict}
            )
        if scaler_prelogdiff_1 is not None:
            return_dict["scalers"].update(
                {"scaler_prelogdiff_1": scaler_prelogdiff_1_dict}
            )
        if scaler_prelogdiff_2 is not None:
            return_dict["scalers"].update(
                {"scaler_prelogdiff_2": scaler_prelogdiff_2_dict}
            )
        if scaler_postlogdiff_1 is not None:
            return_dict["scalers"].update(
                {"scaler_postlogdiff_1": scaler_postlogdiff_1_dict}
            )
        if scaler_postlogdiff_2 is not None:
            return_dict["scalers"].update(
                {"scaler_postlogdiff_2": scaler_postlogdiff_2_dict}
            )
        if scaler_dm_1 is not None:
            return_dict["scalers"].update(
                {"scaler_dm_1": scaler_dm_1_dict})
        if scaler_dm_2 is not None:
            return_dict["scalers"].update(
                {"scaler_dm_2": scaler_dm_2_dict})
    if pretty_print:
        pretty_dict(
            return_dict["summary"],
            init_message="########## SUMMARY ##########",
        )
    return return_dict


def mlcausality_splits_loop(splits, X=None, y=None, lag=None, **kwargs):
    """
    This is a utility function that runs mlcausality() for a list of
    splits.

    Example usage:
    import mlcausality
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    X = np.random.random([1000,5])
    y = np.random.random([1000,4])
    tscv = TimeSeriesSplit()
    splits = list(tscv.split(X))
    z = mlcausality.mlcausality_splits_loop(splits=splits,
                                            X=X,y=y,lag=5)

    returns : a pandas.DataFrame with each row corresponding to a
    particular train-test split

    Parameters
    ----------
    splits : list of lists, or list of tuples, where the first element
    in the tuple are the index numbers for the train set and the second
    element in the tuple are the index numbers for the test set.
    For instance, if tscv = TimeSeriesSplit(), then splits can be set
    to list(tscv.split(X)). Note that all train and test set splits
    MUST be composed of consecutive numbered indicies without gaps;
    otherwise the lags will not be generated correctly.

    X : array-like of shape (n_samples, n_features) or None.
    This has only been tested to work with pandas.Series,
    pandas.DataFrame or numpy arrays for single or multiple time-series
    data. For single feature only, a list or a tuple of length
    n_samples can also be passed.

    y : array-like of shape (n_samples,) or (n_samples, n_features).
    This has only been tested to work with pandas.Series,
    pandas.DataFrame, numpy arrays, lists, or tuples. This is the
    target series on which to perform Granger causality analysis. If y
    has multiple columns, the target time series will be the first
    column.

    lag : int. The number of lags to test Granger causality for.

    **kwargs : any other keyword arguments one might want to pass to
    mlcausality(), such as "regressor", or "regressor_fit_params", etc.
    """
    if X is not None:
        kwargs.update({"X": X})
    if y is not None:
        kwargs.update({"y": y})
    if lag is not None:
        kwargs.update({"lag": lag})
    kwargs.update(
        {
            "return_preds": False,
            "return_nanfilled": False,
            "return_models": False,
            "return_scalers": False,
            "return_summary_df": True,
            "kwargs_in_summary_df": True,
            "pretty_print": False,
        }
    )
    if isinstance(splits, types.GeneratorType):
        splits = list(splits)
    out_dfs = []
    split_counter = 0
    for train_idx, test_idx in splits:
        kwargs.update({"split": [train_idx, test_idx]})
        out_df = mlcausality(**kwargs)["summary_df"]
        out_df["split"] = split_counter
        out_dfs.append(out_df)
        split_counter += 1
    all_out_dfs = pd.concat(out_dfs, ignore_index=True)
    return all_out_dfs


def bivariate_mlcausality(
    data,
    lags,
    permute_list=None,
    y_bounds_violation_sign_drop=True,
    ftest=False,
    return_pvalue_matrix_only=False,
    pvalue_matrix_type="sign_test",
    **kwargs,
):
    """
    This function takes several time-series in a single 'data'
    parameter as an input and checks for Granger causal relationships
    between all bivariate combinations of those time-series.
    Internally, all relationships are are tested using mlcausality().

    Returns : pandas.DataFrame unless return_pvalue_matrix_only==True,
    in which case a numpy array with pvalues is returned instead.

    Example usage:
    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([1000,5])
    z = mlcausality.bivariate_mlcausality(data=data,lags=[5,10],
        regressor='krr',
        regressor_params={'kernel':'rbf'})

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features).
    Contains all the time-series for which to perform bivariate
    Granger causality analysis.

    lags : list of ints.
    The number of lags to test Granger causality for. Multiple lag
    orders can be tested by including more than one int in the list.

    permute_list : list or None.
    To calculate bivariate connections for only a subset of the
    time-series include the column indicies of interest in this
    parameter.

    y_bounds_violation_sign_drop : bool.
    Whether to drop rows where the outcome variables in the test set
    are outside the boundaries of the variables in the training set
    when calculating the sign test and/or the Wilcoxon signed rank
    test.

    return_pvalue_matrix_only : bool.
    If True instead of outputing a pandas.Dataframe a numpy array
    similar to an adjacency matrix except with pvalues for the test is
    returned. Note that, in order to have the same format as an
    adjacency matrix where the row variable Granger causes the column
    variable, it is most logical to set 'lags' to a list that only
    contains one lag value. The code will work if 'lags' is a list of
    more than one lag order but the user would then have to account for
    the order of the entries in the resulting matrix.
    return_pvalue_matrix_only is provided in order to make this
    function run faster and to output only the information that is most
    relevant. If performance is not really important to you or you do
    not know what you are doing then set
    return_pvalue_matrix_only=False (the default).

    pvalue_matrix_type : either 'sign_test' or 'wilcoxon'.
    Indicates which pvalues should be included in the pvalue matrix if
    return_pvalue_matrix_only=True. By default the pvalues from the
    sign test are returned.

    ftest : bool.
    Whether to calculate the F-test (only useful if the regressor is
    'linear' or 'classic')

    **kwargs : any other keyword arguments one might want to pass to
    mlcausality(), such as regressor, or regressor_fit_params, etc.
    Some mlcausality() parameters may be unavailable or have no effect
    if called from this function.
    """
    if return_pvalue_matrix_only:
        lags = [lags[0]]
        permute_list = None
    if "y" in kwargs:
        del kwargs["y"]
    if "X" in kwargs:
        del kwargs["X"]
    if "lag" in kwargs:
        del kwargs["lag"]
    kwargs.update(
        {
            "acorr_tests": False,
            "normality_tests": False,
            "return_restrict_only": True,
            "return_inside_bounds_mask": False,
            "return_kwargs_dict": False,
            "return_preds": False,
            "return_errors": True,
            "return_nanfilled": False,
            "return_models": False,
            "return_scalers": False,
            "return_summary_df": False,
            "kwargs_in_summary_df": False,
            "pretty_print": False,
        }
    )
    if y_bounds_violation_sign_drop:
        kwargs_restricted = deepcopy(kwargs)
        kwargs_restricted.update({"return_inside_bounds_mask": True})
    else:
        kwargs_restricted = kwargs
    if isinstance(data, pd.DataFrame):
        hasnames = True
        names = data.columns.to_list()
        data = data.to_numpy()
    else:
        hasnames = False
    if permute_list is None:
        permute_list = list(
            itertools.permutations(range(data.shape[1]), 2))
    if return_pvalue_matrix_only:
        out_df = np.ones([data.shape[1], data.shape[1]])
    results_list = []
    y_unique_list = sorted(set([i[1] for i in permute_list]))
    for y_idx in y_unique_list:
        X_idx_list = [i[0] for i in permute_list if i[1] == y_idx]
        # restricted models
        restricted = {}
        for lag in lags:
            restricted[lag] = mlcausality(
                X=None, y=data[:, [y_idx]], lag=lag, **kwargs_restricted
            )
        for X_idx in X_idx_list:
            data_unrestrict = data[:, [y_idx, X_idx]]
            for lag in lags:
                unrestricted = mlcausality(
                    X=None, y=data_unrestrict, lag=lag, **kwargs
                )
                errors_unrestrict = unrestricted["errors"]["restricted"]
                errors_restrict = restricted[lag]["errors"]["restricted"]
                if ftest and not return_pvalue_matrix_only:
                    errors2_restrict = errors_restrict**2
                    errors2_unrestrict = errors_unrestrict**2
                    f_dfn = lag
                    f_dfd = (
                        errors2_restrict.shape[0] -
                            (lag * data.shape[1]) - 1
                    )
                    if f_dfd <= 0:
                        f_stat = np.nan
                        ftest_p_value = np.nan
                    else:
                        f_stat = (
                            (errors2_restrict.sum() -
                             errors2_unrestrict.sum())
                            / f_dfn
                        ) / (errors2_unrestrict.sum() / f_dfd)
                        ftest_p_value = scipyf.sf(f_stat, f_dfn, f_dfd)
                if y_bounds_violation_sign_drop:
                    errors_unrestrict = (
                        errors_unrestrict
                        * restricted[lag]["inside_bounds_mask"]
                    )
                    errors_restrict = (
                        errors_restrict *
                        restricted[lag]["inside_bounds_mask"]
                    )
                error_delta = np.abs(errors_restrict.flatten()) - np.abs(
                    errors_unrestrict.flatten()
                )
                error_delta_num_positive = (error_delta > 0).sum()
                error_delta_len = error_delta[~np.isnan(
                    error_delta)].shape[0]
                if (
                    return_pvalue_matrix_only
                    and (
                        pvalue_matrix_type == "sign_test"
                        or pvalue_matrix_type == "sign"
                    )
                ) or (not return_pvalue_matrix_only):
                    sign_test_result = binomtest(
                        error_delta_num_positive,
                        error_delta_len,
                        alternative="greater",
                    )
                if (
                    return_pvalue_matrix_only
                    and pvalue_matrix_type == "wilcoxon"
                ) or (not return_pvalue_matrix_only):
                    wilcoxon_abserror = wilcoxon(
                        np.abs(errors_restrict.flatten()),
                        np.abs(errors_unrestrict.flatten()),
                        alternative="greater",
                        nan_policy="omit",
                        zero_method="wilcox",
                    )
                if return_pvalue_matrix_only:
                    if pvalue_matrix_type == "wilcoxon":
                        out_df[X_idx, y_idx] = wilcoxon_abserror.pvalue
                    elif (
                        pvalue_matrix_type == "sign_test"
                        or pvalue_matrix_type == "sign"
                    ):
                        out_df[X_idx, y_idx] = sign_test_result.pvalue
                else:
                    wilcoxon_num_preds = np.count_nonzero(
                        ~np.isnan(errors_restrict.flatten())
                    )
                    if ftest:
                        if hasnames:
                            results_list.append(
                                [
                                    names[X_idx],
                                    names[y_idx],
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                    f_stat,
                                    ftest_p_value,
                                ]
                            )
                        else:
                            results_list.append(
                                [
                                    X_idx,
                                    y_idx,
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                    f_stat,
                                    ftest_p_value,
                                ]
                            )
                    else:
                        if hasnames:
                            results_list.append(
                                [
                                    names[X_idx],
                                    names[y_idx],
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                ]
                            )
                        else:
                            results_list.append(
                                [
                                    X_idx,
                                    y_idx,
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                ]
                            )
    if not return_pvalue_matrix_only:
        if ftest:
            out_df = pd.DataFrame(
                results_list,
                columns=[
                    "X",
                    "y",
                    "lag",
                    "wilcoxon.statistic",
                    "wilcoxon.pvalue",
                    "wilcoxon.num_preds",
                    "sign_test.statistic",
                    "sign_test.pvalue",
                    "ftest.statistic",
                    "ftest.pvalue",
                ],
            )
        else:
            out_df = pd.DataFrame(
                results_list,
                columns=[
                    "X",
                    "y",
                    "lag",
                    "wilcoxon.statistic",
                    "wilcoxon.pvalue",
                    "wilcoxon.num_preds",
                    "sign_test.statistic",
                    "sign_test.pvalue",
                ],
            )
    return out_df


def loco_mlcausality(
    data,
    lags,
    permute_list=None,
    y_bounds_violation_sign_drop=True,
    ftest=False,
    return_pvalue_matrix_only=False,
    pvalue_matrix_type="sign_test",
    **kwargs,
):
    """
    This function takes several time-series in a single 'data'
    parameter as an input and checks for Granger causal relationships
    by Leaving One Column Out (loco) for the restricted model.
    Internally, all relationships are tested using mlcausality().

    Returns : pandas.DataFrame if return_pvalue_matrix_only=False else
    a numpy array similar to an adjacency matrix except with pvalues
    for the test.

    Example usage:
    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    z =  mlcausality.loco_mlcausality(data, lags=[5,10],
         scaler_init_1='quantile', regressor='krr',
         regressor_params={'kernel':'rbf'})

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features).
    Contains all the time-series for which to calculate bivariate
    Granger causality relationships.

    lags : list of ints.
    The number of lags to test Granger causality for. Multiple lag
    orders can be tested by including more than one int in the list.

    permute_list : list or None.
    To calculate bivariate connections for only a subset of the
    time-series include the column indicies to use in this parameter.

    y_bounds_violation_sign_drop : bool.
    Whether to rows where the outcome variables in the test set are
    outside the boundaries of the variables in the training set.

    ftest : bool.
    Whether to calculate the F-test (only useful if the regressor is
    'linear' or 'classic')

    return_pvalue_matrix_only : bool.
    If True instead of outputing a pandas.Dataframe a numpy array
    similar to an adjacency matrix except with pvalues for the test is
    returned. Note that, in order to have the same format as an
    adjacency matrix where the row variable Granger causes the column
    variable, it is most logical to set 'lags' to a list that only
    contains one lag value. The code will work if 'lags' is a list of
    more than one lag order but the user would then have to account for
    the order of the entries in the resulting matrix.
    return_pvalue_matrix_only is provided in order to make this
    function run faster and to output only the information that is most
    important. If performance is not really important to you or you do
    not know what you are doing then set
    return_pvalue_matrix_only=False (the default).

    pvalue_matrix_type : either 'sign_test' or 'wilcoxon'.
    Indicates which pvalues should be included in the pvalue matrix if
    return_pvalue_matrix_only=True. By default the pvalues from the
    sign test are returned.

    **kwargs : any other keyword arguments one might want to pass to
    mlcausality(), such as regressor, or regressor_fit_params, etc.
    Some parameter values in mlcausality() may be unavailable or have
    no effect if called from this function.
    """
    if return_pvalue_matrix_only:
        lags = [lags[0]]
        permute_list = None
    if "y" in kwargs:
        del kwargs["y"]
    if "X" in kwargs:
        del kwargs["X"]
    if "lag" in kwargs:
        del kwargs["lag"]
    kwargs.update(
        {
            "acorr_tests": False,
            "normality_tests": False,
            "return_restrict_only": True,
            "return_inside_bounds_mask": False,
            "return_kwargs_dict": False,
            "return_preds": False,
            "return_errors": True,
            "return_nanfilled": False,
            "return_models": False,
            "return_scalers": False,
            "return_summary_df": False,
            "kwargs_in_summary_df": False,
            "pretty_print": False,
        }
    )
    if y_bounds_violation_sign_drop:
        kwargs_unrestricted = deepcopy(kwargs)
        kwargs_unrestricted.update({"return_inside_bounds_mask": True})
    else:
        kwargs_unrestricted = kwargs
    if isinstance(data, pd.DataFrame):
        hasnames = True
        names = data.columns.to_list()
        data = data.to_numpy()
    else:
        hasnames = False
    if permute_list is None:
        permute_list = list(
            itertools.permutations(range(data.shape[1]), 2))
    if return_pvalue_matrix_only:
        out_df = np.ones([data.shape[1], data.shape[1]])
    else:
        if isinstance(data, pd.DataFrame):
            hasnames = True
            names = data.columns.to_list()
            data = data.to_numpy()
        else:
            hasnames = False
        results_list = []
    y_unique_list = sorted(set([i[1] for i in permute_list]))
    for y_idx in y_unique_list:
        X_idx_list = [i[0] for i in permute_list if i[1] == y_idx]
        # unrestricted models
        unrestricted = {}
        for lag in lags:
            unrestricted[lag] = mlcausality(
                X=None,
                y=data[
                    :,
                    [y_idx]
                    + [i for i in range(data.shape[1])
                       if i not in [y_idx]],
                ],
                lag=lag,
                **kwargs_unrestricted,
            )
        for X_idx in X_idx_list:
            data_restrict = data[
                :,
                [y_idx]
                + [i for i in range(data.shape[1])
                   if i not in [y_idx, X_idx]],
            ]
            for lag in lags:
                restricted = mlcausality(
                    X=None, y=data_restrict, lag=lag, **kwargs
                )
                errors_unrestrict = unrestricted[lag]["errors"]["restricted"]
                errors_restrict = restricted["errors"]["restricted"]
                if ftest:
                    errors2_restrict = errors_restrict**2
                    errors2_unrestrict = errors_unrestrict**2
                    f_dfn = lag
                    f_dfd = (
                        errors2_restrict.shape[0] -
                            (lag * data.shape[1]) - 1
                    )
                    if f_dfd <= 0:
                        f_stat = np.nan
                        ftest_p_value = np.nan
                    else:
                        f_stat = (
                            (errors2_restrict.sum() -
                             errors2_unrestrict.sum())
                            / f_dfn
                        ) / (errors2_unrestrict.sum() / f_dfd)
                        ftest_p_value = scipyf.sf(f_stat, f_dfn, f_dfd)
                if y_bounds_violation_sign_drop:
                    errors_unrestrict = (
                        errors_unrestrict
                        * unrestricted[lag]["inside_bounds_mask"]
                    )
                    errors_restrict = (
                        errors_restrict
                        * unrestricted[lag]["inside_bounds_mask"]
                    )
                error_delta = np.abs(errors_restrict.flatten()) - np.abs(
                    errors_unrestrict.flatten()
                )
                error_delta_num_positive = (error_delta > 0).sum()
                error_delta_len = error_delta[~np.isnan(
                    error_delta)].shape[0]
                if (
                    return_pvalue_matrix_only
                    and (
                        pvalue_matrix_type == "sign_test"
                        or pvalue_matrix_type == "sign"
                    )
                ) or (not return_pvalue_matrix_only):
                    sign_test_result = binomtest(
                        error_delta_num_positive,
                        error_delta_len,
                        alternative="greater",
                    )
                if (
                    return_pvalue_matrix_only
                    and pvalue_matrix_type == "wilcoxon"
                ) or (not return_pvalue_matrix_only):
                    wilcoxon_abserror = wilcoxon(
                        np.abs(errors_restrict.flatten()),
                        np.abs(errors_unrestrict.flatten()),
                        alternative="greater",
                        nan_policy="omit",
                        zero_method="wilcox",
                    )
                wilcoxon_num_preds = np.count_nonzero(
                    ~np.isnan(errors_restrict.flatten())
                )
                if return_pvalue_matrix_only:
                    if pvalue_matrix_type == "wilcoxon":
                        out_df[X_idx, y_idx] = wilcoxon_abserror.pvalue
                    elif (
                        pvalue_matrix_type == "sign_test"
                        or pvalue_matrix_type == "sign"
                    ):
                        out_df[X_idx, y_idx] = sign_test_result.pvalue
                else:
                    if ftest:
                        if hasnames:
                            results_list.append(
                                [
                                    names[X_idx],
                                    names[y_idx],
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                    f_stat,
                                    ftest_p_value,
                                ]
                            )
                        else:
                            results_list.append(
                                [
                                    X_idx,
                                    y_idx,
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                    f_stat,
                                    ftest_p_value,
                                ]
                            )
                    else:
                        if hasnames:
                            results_list.append(
                                [
                                    names[X_idx],
                                    names[y_idx],
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                ]
                            )
                        else:
                            results_list.append(
                                [
                                    X_idx,
                                    y_idx,
                                    lag,
                                    wilcoxon_abserror.statistic,
                                    wilcoxon_abserror.pvalue,
                                    wilcoxon_num_preds,
                                    sign_test_result.statistic,
                                    sign_test_result.pvalue,
                                ]
                            )
                    if ftest:
                        out_df = pd.DataFrame(
                            results_list,
                            columns=[
                                "X",
                                "y",
                                "lag",
                                "wilcoxon.statistic",
                                "wilcoxon.pvalue",
                                "wilcoxon.num_preds",
                                "sign_test.statistic",
                                "sign_test.pvalue",
                                "ftest.statistic",
                                "ftest.pvalue",
                            ],
                        )
                    else:
                        out_df = pd.DataFrame(
                            results_list,
                            columns=[
                                "X",
                                "y",
                                "lag",
                                "wilcoxon.statistic",
                                "wilcoxon.pvalue",
                                "wilcoxon.num_preds",
                                "sign_test.statistic",
                                "sign_test.pvalue",
                            ],
                        )
    return out_df


def multireg_mlcausality(
    data,
    lag,
    scaler_init_1=None,
    scaler_init_1_params=None,
    scaler_init_2=None,
    scaler_init_2_params=None,
    scaler_prelogdiff_1=None,
    scaler_prelogdiff_1_params=None,
    scaler_prelogdiff_2=None,
    scaler_prelogdiff_2_params=None,
    logdiff=False,
    scaler_postlogdiff_1=None,
    scaler_postlogdiff_1_params=None,
    scaler_postlogdiff_2=None,
    scaler_postlogdiff_2_params=None,
    split=None,
    train_size=0.7,
    early_stop_frac=0.0,
    early_stop_min_samples=1000,
    early_stop_rounds=50,
    scaler_postsplit_1=None,
    scaler_postsplit_1_params=None,
    scaler_postsplit_2=None,
    scaler_postsplit_2_params=None,
    scaler_dm_1=None,
    scaler_dm_1_params=None,
    scaler_dm_2=None,
    scaler_dm_2_params=None,
    regressor="default",
    regressor_params=None,
    regressor_fit_params=None,
    return_inside_bounds_mask=True,
    return_kwargs_dict=False,
    return_preds=False,
    return_errors=True,
    return_model=False,
    return_scalers=False,
    return_summary_df=False,
    kwargs_in_summary_df=False,
):
    """
    Multiregression version of the mlcausality() is a function. At the
    moment, this function only works with 2 regressors:
        1) krr (kernel ridge regression)
        2) catboostregressor

    Depending on which regressor is chosen, this function behaves
    differently:
        1) If regressor=='krr' then this function is effectively the
           multiregression version of mlcausality() and returns
           predictions from a multiregression model for all time-series
           in "data". The predictions are identical to those obtained
           using mlcausality() except the function runs much faster
           because of the multiregression format.
        2) If regressor=='catboostregressor' then the obective is set
           to 'MultiRMSEWithMissingValues' by default which is not an
           equivalent objective to 'RMSE' in the single target CatBoost
           regression model. See the CatBoost documentation for
           details.

    returns a dict with elements that depend on the parameters with
    which this function is called.

    Example usage:
    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([1000,5])
    z = mlcausality.multireg_mlcausality(data,lag=5)

    Parameters
    ----------
    data : array-like of shape (n_samples,) or (n_samples, n_features).
    This has only been tested to work with pandas.Series,
    pandas.DataFrame, numpy arrays, lists, or tuples.

    lag : int.
    The number of lags to test Granger causality for.

    scaler_init_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler are NOT
    inversed before a test of Granger causality occurs: once data is
    transformed using this scaler, it stays transformed throughout the
    entire call of this function. The scaling is done using the
    relevant scaler from scikit-learn. Parameters can be set using
    'scaler_init_1_params'.

    scaler_init_1_params : dict or None.
    The parameters for 'scaler_init_1'. The parameters must correspond
    to the relevant scaler's parameters in the scikit-learn package.

    scaler_init_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler are NOT
    inversed before a test of Granger causality occurs: once data is
    transformed using this scaler, it stays transformed throughout the
    entire call of this function. The scaling is done using the
    relevant scaler from scikit-learn. Parameters can be set using
    'scaler_init_2_params'.

    scaler_init_2_params : dict or None.
    The parameters for 'scaler_init_2'. The parameters must correspond
    to the relevant scaler's parameters in the scikit-learn package.

    scaler_prelogdiff_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_prelogdiff_1_params'.

    scaler_prelogdiff_1_params : dict or None.
    The parameters for 'scaler_prelogdiff_1'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    scaler_prelogdiff_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_prelogdiff_2_params'.

    scaler_prelogdiff_2_params : dict or None.
    The parameters for 'scaler_prelogdiff_2'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    logdiff: bool.
    Whether to take a log difference of all the time-series in y and X.
    Note that logdiff is applied before the train, val and test splits
    are taken but that each of these datasets will lose an observation
    as a result of the logdiff operation. In consequence,
    len(test) - lag - 1 predictions will be made by both the restricted
    and unrestricted models. Predictions are subjected to an inversion
    of the logdiff operation and Granger causality analysis occurs on
    the inversed prediction data.

    scaler_postlogdiff_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postlogdiff_1_params'.

    scaler_postlogdiff_1_params : dict or None.
    The parameters for 'scaler_postlogdiff_1'. The parameters must
    correspond to the relevant scaler's parameters in the
    scikit-learn package.

    scaler_postlogdiff_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postlogdiff_2_params'.

    scaler_postlogdiff_2_params : dict or None.
    The parameters for 'scaler_postlogdiff_2'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    split : None, or an iterable of 2 iterables.
    In the typical case this will be a list of 2 lists where the first
    list contains the index numbers of rows in the training set and the
    second list contains the index numbers of rows in the testing set.
    Note that the index numbers in both the train and test splits MUST
    be consecutive and without gaps, otherwise lags will not be taken
    correctly for the time-series in y and X. If split is None, then
    train_size (described below) must be set.

    train_size : float between (0,1) or int.
    If split is None then this train_size describes the fraction of the
    dataset used for training. If it is an int, it states how many
    observations to use for training. For instance, if the data has
    1000 rows and train_size == 0.7 then the first 700 rows are the
    training set and the latter 300 rows are the test set. If early
    stopping is also used (see early_stop_frac below), then the
    training set is further divided into a training set and a
    validation set. For example, if train_size == 0.7,
    early_stop_frac == 0.1, enough data is available to early stop, and
    a regressor is used that employs early stopping, then data that has
    1000 rows will have a training set size of 0.9*0.7*1000 = 630, a
    validation set size of 0.1*0.7*1000 = 70, and a test set size of
    0.3*1000 = 300. Note that each of these sets will further lose one
    observation if logdiff (described above) is set to True. If
    train_size==1 and split==None then the train and test sets are
    identical and equal to the entire dataset.

    early_stop_frac : float between [0.0,1.0).
    The fraction of training data to use for early stopping if there is
    a sufficient number of observations and the regressor (described
    below) is one of 'catboostregressor', 'xgbregressor', or
    'lgbmregressor'. Note that if the regressor is set to a string
    other than 'catboostregressor', 'xgbregressor', or 'lgbmregressor'
    then early_stop_frac has no effect. The "sufficient number of
    observations" criteria is defined as follows: early stopping will
    happen if
    early_stop_frac*len(train) - lags - 1 >= early_stop_min_samples
    where len(train) is the length of the training set (after logdiff
    if logdiff is applied) and early_stop_min_samples is as described
    below. If you do not want to use early stopping, set this to 0.0,
    which is the default.

    early_stop_min_samples : int.
    Early stopping minimum validation dataset size. For more
    information, read early_stop_frac above.

    early_stop_rounds : int.
    The number of rounds to use for early stopping. For more
    information, read the relevant documentation for CatBoost,
    LightGBM and/or XGBoost.

    scaler_postsplit_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postsplit_1_params'.

    scaler_postsplit_1_params : dict or None.
    The parameters for 'scaler_postsplit_1'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    scaler_postsplit_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the input data
    (y and X). Transformations performed by this scaler ARE inversed
    before a test of Granger causality occurs: the transformed data is
    used to train and generate predictions for the restricted and
    unrestricted models, however, the predictions are then inversed to
    account for this scaler and Granger causality analysis occurs on
    the inversed data. The scaling is done using the relevant scaler
    from scikit-learn. Parameters can be set using
    'scaler_postsplit_2_params'.

    scaler_postsplit_2_params : dict or None.
    The parameters for 'scaler_postsplit_2'. The parameters must
    correspond to the relevant scaler's parameters in the scikit-learn
    package.

    scaler_dm_1 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the design matrix
    composed of lags of X. The scaling is done using the relevant
    scaler from scikit-learn. Parameters can be set using
    'scaler_dm_1_params'.

    scaler_dm_1_params : dict or None.
    The parameters for 'scaler_dm_1'. The parameters must correspond to
    the relevant scaler's parameters in the scikit-learn package.

    scaler_dm_2 : None or one of 'maxabsscaler' ('maxabs'),
    'minmaxscaler' ('minmax'), 'powertransformer' ('power'),
    'quantiletransformer' ('quantile'), 'robustscaler' ('robust') or
    'normalizer'.
    Applies a scaler or preprocessing transformer to the design matrix
    composed of lags of X. The scaling is done using the relevant
    scaler from scikit-learn. Parameters can be set using
    'scaler_dm_2_params'.

    scaler_dm_2_params : dict or None.
    The parameters for 'scaler_dm_2'. The parameters must correspond to
    the relevant scaler's parameters in the scikit-learn package.

    regressor : string. One of:
        - 'krr' : kernel ridge regressor
        - 'catboostregressor' : CatBoost regressor
        - 'default' : kernel ridge regressor with the RBF kernel set as
            default (default)
    Note that you must have the correct library installed in order to
    use these regressors with mlcausality.

    regressor_params : dict, list of 2 dicts, or None.
    These are the parameters with which the regressor is initialized.
    For instance, if you want to use the 'rbf' kernel with kernel
    ridge, you could use
    regressor_params={'regressor_params':{'kernel':'rbf'}}. A list of 2
    dicts provides a separate set of parameters for the restricted and
    unrestricted models respectively.

    regressor_fit_params : dict, list of 2 dicts, or None.
    These are the parameters used with the regressor's fit method.

    return_inside_bounds_mask : bool.
    Whether to return a mask that indicates whether the label value in
    the test set is within the [min,max] range of the training set.
    This could be useful for some models that do not extrapolate well,
    for instance, tree-based models like random forests.

    return_kwargs_dict : bool.
    Whether to return a dict of all kwargs passed to mlcausality().

    return_preds : bool.
    Whether to return the predictions of the models. If
    return_preds=True and return_restrict_only=True, then only the
    predictions of the restricted model will be returned. Note that if
    any of the "init" type of scalers are used then the preds returned
    are for the data transformed by those "init" scalers.

    return_errors : bool.
    Whether to return the errors of the models. If return_errors=True
    and return_restrict_only=True, then only the errors of the
    restricted model will be returned. Note that if any of the "init"
    type of scalers are used then the errors returned are for the data
    transformed by those "init" scalers.

    return_nanfilled : bool.
    Whether to return preds and errors with nan values in the vector
    corresponding to the positions of the input data that were not in
    the test set. This ensures that the predictions vector, for
    example, has the exact same number of observations as the input
    data. If False then the predictions vector could be shorter than
    the total amount of data if the test set contains only a subset of
    the entire dataset.

    return_models : bool.
    If True instances of the fitted models will be returned.

    return_scalers : bool.
    If True fitted instances of scalers (if used) will be returned.

    return_summary_df : bool.
    Whether to return a summary of the return in a pandas.DataFrame
    format.

    kwargs_in_summary_df : bool.
    If this is True and return_summary_df=True then the kwargs passed
    to mlcausality() will be returned in the summary pandas.DataFrame
    """
    # Store and parse the dict of passed variables
    if return_kwargs_dict:
        kwargs_dict = locals()
        del kwargs_dict["data"]
        if kwargs_dict["split"] is not None:
            kwargs_dict["split"] = "notNone"
    # Initial parameter checks; data scaling; and data splits
    early_stop = False
    if (
        (scaler_init_1 is not None and scaler_init_1.lower() == "normalizer")
        or (
            scaler_init_2 is not None and scaler_init_2.lower() == "normalizer"
        )
        or (
            scaler_prelogdiff_1 is not None
            and scaler_prelogdiff_1.lower() == "normalizer"
        )
        or (
            scaler_prelogdiff_2 is not None
            and scaler_prelogdiff_2.lower() == "normalizer"
        )
        or (
            scaler_postlogdiff_1 is not None
            and scaler_postlogdiff_1.lower() == "normalizer"
        )
        or (
            scaler_postlogdiff_2 is not None
            and scaler_postlogdiff_2.lower() == "normalizer"
        )
        or (
            scaler_postsplit_1 is not None
            and scaler_postsplit_1.lower() == "normalizer"
        )
        or (
            scaler_postsplit_2 is not None
            and scaler_postsplit_2.lower() == "normalizer"
        )
    ):
        raise ValueError(
            "'normalizer' can only be used with 'scaler_dm_1' or 'scaler_dm_2'; do not "
            "use it with other scalers"
        )
    if data is None or lag is None:
        raise TypeError(
            "You must supply data and lag to multireg_mlcausality")
    if not isinstance(lag, int):
        raise TypeError(
            "lag was not passed as an int to multireg_mlcausality")
    if isinstance(data, (list, tuple)):
        data = np.atleast_2d(data).reshape(-1, 1)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if len(data.shape) == 1 or data.shape[1] == 1:
            data = np.atleast_2d(data.to_numpy()).reshape(-1, 1)
        else:
            data = data.to_numpy()
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "data could not be cast to np.ndarray in multireg_mlcausality"
        )
    if len(data.shape) == 1:
        data = np.atleast_2d(data).reshape(-1, 1)
    if not isinstance(logdiff, bool):
        raise TypeError(
            "logdiff must be a bool in multireg_mlcausality")
    if regressor.lower() == "catboostregressor":
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float64)
    if scaler_init_1 is not None:
        scaler_init_1_dict = {}
        scaler_init_1_dict["data"] = init_scaler(
            scaler=scaler_init_1, scaler_params=scaler_init_1_params
        )
        data = scaler_init_1_dict["data"].fit_transform(data)
    if scaler_init_2 is not None:
        scaler_init_2_dict = {}
        scaler_init_2_dict["data"] = init_scaler(
            scaler=scaler_init_2, scaler_params=scaler_init_2_params
        )
        data = scaler_init_2_dict["data"].fit_transform(data)
    # if train_size == 1:
    #    early_stop_frac = 0.0
    #    split_override = True
    # else:
    #    split_override = False
    split_override = False
    if regressor == "default":
        regressor = "krr"
        regressor_params = {"kernel": "rbf"}
    # ytrue
    ytrue = deepcopy(data)
    if scaler_prelogdiff_1 is not None:
        scaler_prelogdiff_1_dict = {}
        scaler_prelogdiff_1_dict["data"] = init_scaler(
            scaler=scaler_prelogdiff_1,
            scaler_params=scaler_prelogdiff_1_params,
        )
        data_scaled = scaler_prelogdiff_1_dict["data"].fit_transform(
            data)
    else:
        data_scaled = data
    if scaler_prelogdiff_2 is not None:
        scaler_prelogdiff_2_dict = {}
        scaler_prelogdiff_2_dict["data"] = init_scaler(
            scaler=scaler_prelogdiff_2,
            scaler_params=scaler_prelogdiff_2_params,
        )
        data_scaled = scaler_prelogdiff_2_dict["data"].fit_transform(
            data_scaled
        )
    # Logdiff
    if logdiff:
        # Store outcome variable with current transformations.
        # This is needed to reverse the logdiff operation later when generating preds
        prelogdiff_data_scaled = deepcopy(data_scaled)
        data_scaled = np.diff(np.log(data_scaled), axis=0)
    if scaler_postlogdiff_1 is not None:
        scaler_postlogdiff_1_dict = {}
        scaler_postlogdiff_1_dict["data"] = init_scaler(
            scaler=scaler_postlogdiff_1,
            scaler_params=scaler_postlogdiff_1_params,
        )
        data_scaled = scaler_postlogdiff_1_dict["data"].fit_transform(
            data_scaled
        )
    if scaler_postlogdiff_2 is not None:
        scaler_postlogdiff_2_dict = {}
        scaler_postlogdiff_2_dict["data"] = init_scaler(
            scaler=scaler_postlogdiff_2,
            scaler_params=scaler_postlogdiff_2_params,
        )
        data_scaled = scaler_postlogdiff_2_dict["data"].fit_transform(
            data_scaled
        )
    if not split_override and split is not None:
        if isinstance(split, types.GeneratorType):
            split = list(split)
        if len(split) != 2:
            raise ValueError(
                "If split is provided to multireg_mlcausality, it must be of length 2"
            )
        if logdiff:
            split_new = []
            split_new.append([s - 1 for s in split[0] if s - 1 >= 0])
            split_new.append([s - 1 for s in split[1] if s - 1 >= 0])
            split = split_new
        train = data_scaled[split[0], :]
        test = data_scaled[split[1], :]
    elif train_size == 1:
        train = data_scaled.copy()
        test = data_scaled.copy()
    elif isinstance(train_size, int) and train_size != 0 and train_size != 1:
        if logdiff and train_size < lag + 2:
            raise ValueError(
                "train_size is too small, resulting in no samples in the train set!"
            )
        elif logdiff and train_size > data.shape[0] - lag - 2:
            raise ValueError(
                "train_size is too large, resulting in no samples in the test set!"
            )
        elif not logdiff and train_size < lag + 1:
            raise ValueError(
                "train_size is too small, resulting in no samples in the train set!"
            )
        elif not logdiff and train_size > data.shape[0] - lag - 1:
            raise ValueError(
                "train_size is too large, resulting in no samples in the test set!"
            )
        if logdiff:
            train = data_scaled[: train_size - 1, :]
            test = data_scaled[train_size - 1:, :]
        else:
            train = data_scaled[:train_size, :]
            test = data_scaled[train_size:, :]
    elif isinstance(train_size, float):
        if train_size <= 0 or train_size > 1:
            raise ValueError(
                "train_size is a float that is not between (0,1] in multireg_mlcausality"
            )
        elif logdiff and round(train_size * data.shape[0]) - lag - 2 < 0:
            raise ValueError(
                "train_size is a float that is too small resulting in no samples in train"
            )
        elif logdiff and round((1 - train_size) * data.shape[0]) - lag - 2 < 0:
            raise ValueError(
                "train_size is a float that is too large resulting in no samples in test"
            )
        elif not logdiff and round(train_size * data.shape[0]) - lag - 1 < 0:
            raise ValueError(
                "train_size is a float that is too small resulting in no samples in train"
            )
        elif (
            not logdiff
            and round((1 - train_size) * data.shape[0]) - lag - 1 < 0
        ):
            raise ValueError(
                "train_size is a float that is too large resulting in no samples in test"
            )
        else:
            if logdiff:
                train = data_scaled[: round(
                    train_size * data.shape[0]) - 1, :]
                test = data_scaled[round(
                    train_size * data.shape[0]) - 1:, :]
            else:
                train = data_scaled[: round(
                    train_size * data.shape[0]), :]
                test = data_scaled[round(
                    train_size * data.shape[0]):, :]
    else:
        raise TypeError(
            'train_size must be provided as a float or int to multireg_mlcausality. '
            'Alternatively, you can provide a split to "split".'
        )
    # Regressors
    if regressor_fit_params is None:
        regressor_fit_params = {}
    if regressor_params is not None:
        if not isinstance(regressor_params, (dict)):
            raise TypeError(
                "regressor_params have to be one of None, dict")
        else:
            pass
    else:
        regressor_params = {}
    if regressor.lower() == "catboostregressor":
        if "objective" not in regressor_params.keys():
            regressor_params.update(
                {"objective": "MultiRMSEWithMissingValues"}
            )
        if "verbose" not in regressor_params.keys():
            regressor_params.update({"verbose": False})
        if (
            not isinstance(early_stop_frac, float)
            or early_stop_frac < 0
            or early_stop_frac >= 1
        ):
            raise ValueError(
                "early_stop_frac must be a float in [0,1)")
        if not isinstance(early_stop_min_samples, int):
            raise TypeError("early_stop_min_samples must be an int")
        # if we have less than early_stop_min_samples samples for validation, do not use
        # early stopping. Otherwise, use early stopping
        if (
            logdiff
            and round(early_stop_frac * (train.shape[0] + 1))
            - lag
            - 1
            - early_stop_min_samples
            < 0
        ):
            early_stop = False
        elif (
            not logdiff
            and round(early_stop_frac * train.shape[0])
            - lag
            - early_stop_min_samples
            < 0
        ):
            early_stop = False
        else:
            early_stop = True
        if early_stop:
            if logdiff:
                val = deepcopy(
                    train[
                        round((1 - early_stop_frac)
                              * (train.shape[0] + 1))
                        - 1:,
                        :,
                    ]
                )
                train = deepcopy(
                    train[
                        : round((1 - early_stop_frac) * (train.shape[0] + 1))
                        - 1,
                        :,
                    ]
                )
            else:
                val = deepcopy(
                    train[round((1 - early_stop_frac)
                                * train.shape[0]):, :]
                )
                train = deepcopy(
                    train[: round((1 - early_stop_frac)
                                  * train.shape[0]), :]
                )
        from catboost import CatBoostRegressor

        if early_stop:
            regressor_params.update(
                {"early_stopping_rounds": early_stop_rounds}
            )
        model = CatBoostRegressor(**regressor_params)
    elif (
        regressor.lower() == "kernelridge"
        or regressor.lower() == "kernelridgeregressor"
        or regressor.lower() == "krr"
    ):
        from sklearn.kernel_ridge import KernelRidge

        model = KernelRidge(**regressor_params)
    train_integ = train
    test_integ = test
    if early_stop:
        val_integ = val
    if logdiff:
        test_integ = test_integ[1:, :]
        if early_stop:
            val_integ = val_integ[1:, :]
    if scaler_postsplit_1 is not None:
        scaler_postsplit_1_dict = {}
        scaler_postsplit_1_dict["data"] = init_scaler(
            scaler=scaler_postsplit_1, scaler_params=scaler_postsplit_1_params
        )
        train_integ = scaler_postsplit_1_dict["data"].fit_transform(
            train_integ
        )
        test_integ = scaler_postsplit_1_dict["data"].transform(
            test_integ)
        if early_stop:
            val_integ = scaler_postsplit_1_dict["data"].transform(
                val_integ)
    if scaler_postsplit_2 is not None:
        scaler_postsplit_2_dict = {}
        scaler_postsplit_2_dict["data"] = init_scaler(
            scaler=scaler_postsplit_2, scaler_params=scaler_postsplit_2_params
        )
        train_integ = scaler_postsplit_2_dict["data"].fit_transform(
            train_integ
        )
        test_integ = scaler_postsplit_2_dict["data"].transform(
            test_integ)
        if early_stop:
            val_integ = scaler_postsplit_2_dict["data"].transform(
                val_integ)
    # y bounds indicies
    if return_inside_bounds_mask:
        inside_bounds_mask = np.logical_and(
            test_integ[lag:]
            >= np.tile(
                np.nanmin(train_integ[lag:], axis=0).reshape(-1, 1),
                test_integ[lag:].shape[0],
            ).T,
            test_integ[lag:]
            <= np.tile(
                np.nanmax(train_integ[lag:], axis=0).reshape(-1, 1),
                test_integ[lag:].shape[0],
            ).T,
        ).astype(float)
        inside_bounds_mask[inside_bounds_mask == 0] = np.nan
    # Sliding window views
    # Lag+1 gives lag features plus the target column
    train_sw = sliding_window_view(
        train_integ, [lag + 1, data_scaled.shape[1]]
    )
    # Lag+1 gives lag features plus the target column
    test_sw = sliding_window_view(
        test_integ, [lag + 1, data_scaled.shape[1]])
    if early_stop:
        # Lag+1 gives lag features plus the target column
        val_sw = sliding_window_view(
            val_integ, [lag + 1, data_scaled.shape[1]]
        )
    # Reshape data
    train_sw_reshape = train_sw.reshape(
        train_sw.shape[0],
        train_sw.shape[1] * train_sw.shape[2] * train_sw.shape[3],
    )
    test_sw_reshape = test_sw.reshape(
        test_sw.shape[0],
        test_sw.shape[1] * test_sw.shape[2] * test_sw.shape[3],
    )
    if early_stop:
        val_sw_reshape = val_sw.reshape(
            val_sw.shape[0],
            val_sw.shape[1] * val_sw.shape[2] * val_sw.shape[3],
        )
    # Design matrix scalers: restricted model
    if scaler_dm_1 is not None:
        scaler_dm_1_dict = {}
        scaler_dm_1_dict["data"] = init_scaler(
            scaler=scaler_dm_1, scaler_params=scaler_dm_1_params
        )
        X_train_dm = scaler_dm_1_dict["data"].fit_transform(
            train_sw_reshape[:, : -data_scaled.shape[1]]
        )
        X_test_dm = scaler_dm_1_dict["data"].transform(
            test_sw_reshape[:, : -data_scaled.shape[1]]
        )
        if early_stop:
            X_val_dm = scaler_dm_1_dict["data"].transform(
                val_sw_reshape[:, : -data_scaled.shape[1]]
            )
    else:
        X_train_dm = train_sw_reshape[:, : -data_scaled.shape[1]]
        X_test_dm = test_sw_reshape[:, : -data_scaled.shape[1]]
        if early_stop:
            X_val_dm = val_sw_reshape[:, : -data_scaled.shape[1]]
    if scaler_dm_2 is not None:
        scaler_dm_2_dict = {}
        scaler_dm_2_dict["data"] = init_scaler(
            scaler=scaler_dm_2, scaler_params=scaler_dm_2_params
        )
        X_train_dm = scaler_dm_2_dict["data"].fit_transform(X_train_dm)
        X_test_dm = scaler_dm_2_dict["data"].transform(X_test_dm)
        if early_stop:
            X_val_dm = scaler_dm_1_dict["data"].transform(X_val_dm)
    # Fit model
    # Handle early stopping
    if early_stop:
        regressor_fit_params.update(
            {
                "eval_set": [
                    (X_val_dm,
                     val_sw_reshape[:, -data_scaled.shape[1]:])
                ]
            }
        )
    model.fit(
        X_train_dm,
        train_sw_reshape[:, -data_scaled.shape[1]:],
        **regressor_fit_params,
    )
    preds = model.predict(X_test_dm)
    if regressor.lower() == "catboostregressor" and len(preds.shape) == 1:
        preds = preds.reshape(-1, 1)
    if not split_override and split is not None:
        if logdiff:
            split_unadj = [i + 1 for i in split[1]]
            split_unadj = split_unadj[1:]
        else:
            split_unadj = split[1]
        ytrue = ytrue[split_unadj]
        ytrue = ytrue[lag:]
    else:
        ytrue = ytrue[-preds.shape[0]:]
    # Transform preds if transformations were originally applied
    if scaler_postsplit_2 is not None:
        preds = scaler_postsplit_2_dict["data"].inverse_transform(
            preds)
    if scaler_postsplit_1 is not None:
        preds = scaler_postsplit_1_dict["data"].inverse_transform(
            preds)
    if scaler_postlogdiff_2 is not None:
        preds = scaler_postlogdiff_2_dict["data"].inverse_transform(
            preds)
    if scaler_postlogdiff_1 is not None:
        preds = scaler_postlogdiff_1_dict["data"].inverse_transform(
            preds)
    if logdiff:
        if not split_override and split is not None:
            prelogdiff_mult = prelogdiff_data_scaled[split[1]]
            prelogdiff_mult = prelogdiff_mult[lag + 1:]
        else:
            prelogdiff_mult = prelogdiff_data_scaled[-preds.shape[0] - 1: -1]
        preds = np.exp(preds) * prelogdiff_mult
    if scaler_prelogdiff_2 is not None:
        preds = scaler_prelogdiff_2_dict["data"].inverse_transform(
            preds)
    if scaler_prelogdiff_1 is not None:
        preds = scaler_prelogdiff_1_dict["data"].inverse_transform(
            preds)
    return_dict = {
        "summary": {
            "lag": lag,
            "train_obs": train_integ[:, 0].shape[0],
            "effective_train_obs": train_integ[lag:, 0].shape[0],
            "test_obs": test_integ[:, 0].shape[0],
            "effective_test_obs": test_integ[lag:, 0].shape[0],
        }
    }
    if return_summary_df:
        return_dict.update(
            {"summary_df": pd.json_normalize(return_dict["summary"])}
        )
    if return_kwargs_dict:
        return_dict.update({"kwargs_dict": kwargs_dict})
    if return_kwargs_dict and kwargs_in_summary_df:
        kwargs_df = pd.json_normalize(return_dict["kwargs_dict"])
        kwargs_df = kwargs_df.loc[
            [0], [i for i in kwargs_df.columns if i not in ["lag"]]
        ]
        return_dict["summary_df"] = return_dict["summary_df"].loc[
            [0],
            [
                i
                for i in return_dict["summary_df"].columns
                if i not in ["wilcoxon.y_bounds_violation_sign_drop"]
            ],
        ]
        return_dict["summary_df"] = pd.concat(
            [return_dict["summary_df"], kwargs_df], axis=1
        )
    if return_preds:
        return_dict.update({"ytrue": ytrue, "preds": preds})
    if return_errors:
        errors = preds - ytrue
        return_dict.update({"errors": errors})
    if return_inside_bounds_mask:
        return_dict.update({"inside_bounds_mask": inside_bounds_mask})
    if return_model:
        return_scalers = True
        return_dict.update({"model": model})
    if return_scalers:
        return_dict.update({"scalers": {}})
        if scaler_init_1 is not None:
            return_dict["scalers"].update(
                {"scaler_init_1": scaler_init_1_dict}
            )
        if scaler_init_2 is not None:
            return_dict["scalers"].update(
                {"scaler_init_2": scaler_init_2_dict}
            )
        if scaler_prelogdiff_1 is not None:
            return_dict["scalers"].update(
                {"scaler_prelogdiff_1": scaler_prelogdiff_1_dict}
            )
        if scaler_prelogdiff_2 is not None:
            return_dict["scalers"].update(
                {"scaler_prelogdiff_2": scaler_prelogdiff_2_dict}
            )
        if scaler_postlogdiff_1 is not None:
            return_dict["scalers"].update(
                {"scaler_postlogdiff_1": scaler_postlogdiff_1_dict}
            )
        if scaler_postlogdiff_2 is not None:
            return_dict["scalers"].update(
                {"scaler_postlogdiff_2": scaler_postlogdiff_2_dict}
            )
        if scaler_dm_1 is not None:
            return_dict["scalers"].update(
                {"scaler_dm_1": scaler_dm_1_dict})
        if scaler_dm_2 is not None:
            return_dict["scalers"].update(
                {"scaler_dm_2": scaler_dm_2_dict})
    return return_dict


def multiloco_mlcausality(
    data,
    lags,
    permute_list=None,
    y_bounds_violation_sign_drop=True,
    return_pvalue_matrix_only=False,
    pvalue_matrix_type="sign_test",
    **kwargs,
):
    """
    This function takes several time-series in a single 'data'
    parameter as an input and checks for Granger causal relationships
    by multiregression Leaving One Column Out (loco) for the restricted
    model. Internally, all relationships are tested using
    multireg_mlcausality().

    Returns : pandas.DataFrame if return_pvalue_matrix_only=False else
    a numpy array similar to an adjacency matrix except with pvalues
    for the test.

    Example usage:
    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    z = mlcausality.multiloco_mlcausality(data, lags=[5,10],
        scaler_init_1='quantile', regressor='krr',
        regressor_params={'kernel':'rbf'})

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features).
    Contains all the time-series for which to calculate bivariate
    Granger causality relationships.

    lags : list of ints.
    The number of lags to test Granger causality for. Multiple lag
    orders can be tested by including more than one int in the list.

    permute_list : list or None.
    To calculate bivariate connections for only a subset of the
    time-series include the column indicies to use in this parameter.

    y_bounds_violation_sign_drop : bool.
    Whether to rows where the outcome variables in the test set are
    outside the boundaries of the variables in the training set.

    ftest : bool.
    Whether to calculate the F-test (only useful if the regressor is
    'linear' or 'classic')

    return_pvalue_matrix_only : bool.
    If True instead of outputing a pandas.Dataframe a numpy array
    similar to an adjacency matrix except with pvalues for the test is
    returned. Note that, in order to have the same format as an
    adjacency matrix where the row variable Granger causes the column
    variable, it is most logical to set 'lags' to a list that only
    contains one lag value. The code will work if 'lags' is a list of
    more than one lag order but the user would then have to account for
    the order of the entries in the resulting matrix.
    return_pvalue_matrix_only is provided in order to make this
    function run faster and to output only the information that is most
    important. If performance is not really important to you or you do
    not know what you are doing then set
    return_pvalue_matrix_only=False (the default).

    pvalue_matrix_type : either 'sign_test' or 'wilcoxon'.
    Indicates which pvalues should be included in the pvalue matrix if
    return_pvalue_matrix_only=True. By default the pvalues from the
    sign test are returned.

    **kwargs : any other keyword arguments one might want to pass to
    mlcausality(), such as regressor, or regressor_fit_params, etc.
    Note that some mlcausality() parameter values may be unavailable or
    have no effect if called from this function.
    """
    if return_pvalue_matrix_only:
        lags = [lags[0]]
        permute_list = None
    if "y" in kwargs:
        del kwargs["y"]
    if "X" in kwargs:
        del kwargs["X"]
    if "lag" in kwargs:
        del kwargs["lag"]
    kwargs.update(
        {
            "return_kwargs_dict": False,
            "return_preds": False,
            "return_errors": True,
            "return_inside_bounds_mask": False,
            "return_model": False,
            "return_scalers": False,
            "return_summary_df": False,
            "kwargs_in_summary_df": False,
        }
    )
    if y_bounds_violation_sign_drop:
        kwargs_unrestricted = deepcopy(kwargs)
        kwargs_unrestricted.update({"return_inside_bounds_mask": True})
    else:
        kwargs_unrestricted = kwargs
    if permute_list is None:
        permute_list = list(range(data.shape[1]))
    if return_pvalue_matrix_only:
        out_df = np.ones([data.shape[1], data.shape[1]])
    else:
        if isinstance(data, pd.DataFrame):
            hasnames = True
            names = data.columns.to_list()
            data = data.to_numpy()
        else:
            hasnames = False
        results_list = []
    # unrestricted models
    unrestricted = {}
    for lag in lags:
        unrestricted[lag] = multireg_mlcausality(
            data, lag, **kwargs_unrestricted
        )
    for skip_idx in permute_list:
        data_restrict = data[
            :, [i for i in range(data.shape[1]) if i not in [skip_idx]]
        ]
        for lag in lags:
            restricted = multireg_mlcausality(
                data_restrict, lag, **kwargs)
            errors_unrestrict = unrestricted[lag]["errors"][
                :, [i for i in permute_list if i not in [skip_idx]]
            ]
            errors_restrict = restricted["errors"]
            if y_bounds_violation_sign_drop:
                errors_unrestrict = (
                    errors_unrestrict
                    * unrestricted[lag]["inside_bounds_mask"][
                        :, [i for i in permute_list if i not in [skip_idx]]
                    ]
                )
                errors_restrict = (
                    errors_restrict
                    * unrestricted[lag]["inside_bounds_mask"][
                        :, [i for i in permute_list if i not in [skip_idx]]
                    ]
                )
            for error_idx, y_idx in enumerate(
                [i for i in permute_list if i not in [skip_idx]]
            ):
                if (
                    return_pvalue_matrix_only
                    and pvalue_matrix_type == "wilcoxon"
                ) or (not return_pvalue_matrix_only):
                    wilcoxon_abserror = wilcoxon(
                        np.abs(
                            errors_restrict[:, error_idx].flatten()),
                        np.abs(
                            errors_unrestrict[:, error_idx].flatten()),
                        alternative="greater",
                        nan_policy="omit",
                        zero_method="wilcox",
                    )
                error_delta = np.abs(
                    errors_restrict[:, error_idx].flatten()
                ) - np.abs(errors_unrestrict[:, error_idx].flatten())
                error_delta_num_positive = (error_delta > 0).sum()
                error_delta_len = error_delta[~np.isnan(
                    error_delta)].shape[0]
                if (
                    return_pvalue_matrix_only
                    and (
                        pvalue_matrix_type == "sign_test"
                        or pvalue_matrix_type == "sign"
                    )
                ) or (not return_pvalue_matrix_only):
                    sign_test_result = binomtest(
                        error_delta_num_positive,
                        error_delta_len,
                        alternative="greater",
                    )
                if not return_pvalue_matrix_only:
                    wilcoxon_num_preds = np.count_nonzero(
                        ~np.isnan(
                            errors_restrict[:, error_idx].flatten())
                    )
                if return_pvalue_matrix_only:
                    if pvalue_matrix_type == "wilcoxon":
                        out_df[skip_idx, y_idx] = wilcoxon_abserror.pvalue
                    elif (
                        pvalue_matrix_type == "sign_test"
                        or pvalue_matrix_type == "sign"
                    ):
                        out_df[skip_idx, y_idx] = sign_test_result.pvalue
                else:
                    if hasnames:
                        results_list.append(
                            [
                                names[skip_idx],
                                names[y_idx],
                                lag,
                                wilcoxon_abserror.statistic,
                                wilcoxon_abserror.pvalue,
                                wilcoxon_num_preds,
                                sign_test_result.statistic,
                                sign_test_result.pvalue,
                            ]
                        )
                    else:
                        results_list.append(
                            [
                                skip_idx,
                                y_idx,
                                lag,
                                wilcoxon_abserror.statistic,
                                wilcoxon_abserror.pvalue,
                                wilcoxon_num_preds,
                                sign_test_result.statistic,
                                sign_test_result.pvalue,
                            ]
                        )
    if not return_pvalue_matrix_only:
        out_df = pd.DataFrame(
            results_list,
            columns=[
                "X",
                "y",
                "lag",
                "wilcoxon.statistic",
                "wilcoxon.pvalue",
                "wilcoxon.num_preds",
                "sign_test.statistic",
                "sign_test.pvalue",
            ],
        )
        return out_df.sort_values(["y", "X", "lag"]).reset_index(drop=True)
    else:
        return out_df
