<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[ ![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<!-- 
<br />
<div align="center">
  <a href="https://github.com/WojtekFulmyk/mlcausality">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h3 align="center">mlcausality</h3>

  <p align="center">
    Nonlinear Granger Causality Using Machine Learning Techniques
    <br />
    <a href="https://github.com/WojtekFulmyk/mlcausality/issues">Report Bug</a>
    ·
    <a href="https://github.com/WojtekFulmyk/mlcausality/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#overview-of-available-functions">Overview of available functions</a></li>
        <li><a href="#basic-usage">Basic usage</a></li>
        <li><a href="#setting-parameters">Setting parameters</a></li>
        <li><a href="#available-regressors">Available regressors</a></li>
        <li><a href="#data-preprocessing">Data preprocessing</a></li>
        <li><a href="#data-splits">Data splits</a></li>
        <li><a href="#other-important-parameters">Other important parameters</a></li>
        <li><a href="#additional-help-and-documentation">Additional help and documentation</a></li>
      </ul>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- 
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
-->

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

__mlcausality__ is a Python library for linear and nonlinear Granger causality analysis. Given time-series `X` and `y` if the lags of both `X` and <code>y</code> provide a better prediction for the current value of `y` than the lags of `y` alone then `X` is said to Granger cause `y`. Note that Granger causality is a misnomer: no actual causality is implied, Granger causality is entirely grounded in prediction.

The __mlcausality__ package provides a new way for establishing such Granger causal links using machine learning techniques. Thanks to the usage of the 
<a href="https://en.wikipedia.org/wiki/Sign_test">sign test</a> and the <a href="https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test">Wilcoxon signed rank test</a> under the hood, __mlcausality__ is extremely flexible and can be used with a multitude of machine learning regressors. By default, <a href="https://scikit-learn.org/stable/modules/kernel_ridge.html">kernel ridge regression</a> is used, but the __mlcausality__ package can use other base regressors such as:

 <ul>
  <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html">support vector regressor (SVR)</a></li>
  <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">random forest regressor</a></li>
  <li><a href="https://xgboost.readthedocs.io/en/stable/">XGBoost regressor</a></li>
  <li><a href="https://lightgbm.readthedocs.io/en/stable/">LightGBM regressor</a></li>
  <li><a href="https://catboost.ai/">CatBoost regressor</a></li>
  <li><a href="https://en.wikipedia.org/wiki/Linear_regression">linear regressor</a></li>
</ul>

and more!

When used correctly, __mlcausality__ has exhibited leading performance both in terms of accuracy and execution speed. With __mlcausality__ batteries are always included, and preprocessing is typically not necessary. Have a look at the <a href="#usage">Usage</a> guide below to get an overview of the powerful capabilites that __mlcausality__ provides.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!--
### Built With


* [![NumPy][NumPy]][NumPy-url]
* [![SciPy][SciPy]][SciPy-url]
* [![pandas][pandas]][pandas-url]
* [![statsmodels][statsmodels]][statsmodels-url]
* [![scikit-learn][scikit-learn]][scikit-learn-url]
* [![XGBoost][XGBoost]][XGBoost-url]
* [![LightGBM][LightGBM]][LightGBM-url]
* [![CatBoost][CatBoost]][CatBoost-url]
* [![cuML][cuML]][cuML-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- GETTING STARTED -->
## Getting Started

The following presents the easiest way to get __mlcausality__ up and running on your local computer.

### Prerequisites

Installation requires Python and the pip package installer.

In order to function correctly __mlcausality__ requires the following Python packages:
* [NumPy](NumPy-url)
* [SciPy](SciPy-url)
* [pandas](pandas-url)
* [statsmodels](statsmodels-url)
* [scikit-learn](scikit-learn-url)

__mlcausality__ also has the following optional prerequisites which you should install if you plan to use the relevant regressors:
* [XGBoost](XGBoost-url)
* [LightGBM](LightGBM-url)
* [CatBoost](CatBoost-url)
* [cuML](cuML-url)

### Installation

There are several installation options depending on the number of dependencies that need to be installed. Note that, for all options below, missing and optional dependencies will be installed using pip, except for cuML which does not have a pip installation path; if you wish to use the cuML library you have to install it separately.

From the list of options below, choose the one that suits your needs best:

1. To make a minimal install with just the core prerequisites, run:
    ```sh
    pip install mlcausality@git+https://github.com/WojtekFulmyk/mlcausality.git
    ```

2. To install core prerequisites plus XGBoost, run:
    ```sh
    pip install mlcausality[xgboost]@git+https://github.com/WojtekFulmyk/mlcausality.git
    ```

3. To install core prerequisites plus LightGBM, run:
    ```sh
    pip install mlcausality[lightgbm]@git+https://github.com/WojtekFulmyk/mlcausality.git
    ```

4. To install core prerequisites plus CatBoost, run:
    ```sh
    pip install mlcausality[catboost]@git+https://github.com/WojtekFulmyk/mlcausality.git
    ```

5. To install core prerequisites plus XGBoost, LightGBM and Catboost, run:
    ```sh
    pip install mlcausality[all]@git+https://github.com/WojtekFulmyk/mlcausality.git
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Overview of available functions
The __mlcausality__ package provides the following functions:

* `mlcausality` : Tests whether `X` Granger causes `y`. This is a low-level function that only tests whether `X` Granger-causes `y` but does not test whether `y` Granger-causes `X`
* `mlcausality_splits_loop` : Runs `mlcausality` but for different train-test split combinations supplied by the user.
* `bivariate_mlcausality` : Runs `mlcausality` to test for all bivariate Granger causal relationships amongst the time-series passed to the parameter `data`.
* `loco_mlcausality` : Runs `mlcausality` to test for all Granger causal relationships amongst the time-series passed to the parameter `data` by successively leaving one column out (loco). This function tests for Granger causality in the presence of exogenous time-series whereas `bivariate_mlcausality` only tests for bivariate combinations.
* `multireg_mlcausality` : A multiregression analogue of `mlcausality`. Most users will probably never have to run `multireg_mlcausality` directly; rather, it is expected that `multiloco_mlcausality` will be run instead. Currently `multireg_mlcausality` only supports the kernel ridge regressor and the CatBoost regressor.
* `multiloco_mlcausality` : A multiregression analogue of `loco_mlcausality`. This function uses `multireg_mlcausality` under the hood and is therefore currently supported for kernel ridge regressor and the CatBoost regressor. __If you would like to recover Granger causal connections for an entire network efficiently using kernel ridge regression this is the function you want to use__.

### Basic usage

Suppose you have just 2 time-series of equal length, `X` and `y`, and you would like to find out whether `X` Granger-causes `y` only. Then you can run:

    import mlcausality
    import numpy as np
    import pandas as pd
    X = np.random.random([500,1])
    y = np.random.random([500,1])
    z = mlcausality.mlcausality(X=X,y=y,lag=5)
    #print(z)

Note that both `X` and `y` can be multivariate, meaning that they can take multiple time-series. If `X` is multivariate then the Granger-causality test is run with respect to the lags of all time-series in `X`. If `y` is multivariate then the target time-series is the first one and all additional time-series in `y` are exogenous time-series whose lags are kept in both the restricted and unrestricted models when conducting the Granger causality test.

Now suppose that, instead of just being interested in whether one time series Granger causes another, you would like to instead find all Granger-causal relationships amongst several time-series. In that case, you can run:

    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    z = mlcausality.multiloco_mlcausality(data, lags=[5,10])
    print(z)

The above code will check for all Granger-causal connections amongst all time-series in Data by successively leaving one column out. Note that the above code uses the `multiloco_mlcausality` multi-regression function which will yield identical results to the `loco_mlcausality` function only if the regressor is kernel ridge (the default) but will do so significantly faster than `loco_mlcausality`.

The syntax of the __mlcausality__ package is internally consistent. If you would like to use `loco_mlcausality` instead of `multiloco_mlcausality` for the code block above just substitute `multiloco_mlcausality` with `loco_mlcausality` to obtain an equivalent but slower solution. Moreover, if instead of finding Granger-causal relationships by leaving one column out you instead wanted to just test for Granger-causal relationships in a bivariate fashion, you can instead substitute  `loco_mlcausality` for `bivariate_mlcausality`.

### Setting parameters
The functions `mlcausality` and `multireg_mlcausality` largely share the same parameter spaces so in most cases calls to these two functions can be made with the same parameters.

The functions `bivariate_mlcausality`, `loco_mlcausality` and `multiloco_mlcausality` largely share the same parameter spaces so in most cases calls to these two functions can be made with the same parameters. Moreover, `bivariate_mlcausality` and `loco_mlcausality` admit the parameters that `mlcausality` accepts and `multiloco_mlcausality` admits the parameters that `multireg_mlcausality` accepts. So, for instance, if one wishes to call `loco_mlcausality`, which uses `mlcausality` internally, with a specific set of parameters that one would like to pass to the inner `mlcausality` function, then one simply needs to pass those parameters to the `loco_mlcausality` function:

    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    z = mlcausality.loco_mlcausality(data, lags=[5,10], regressor='catboostregressor')
    print(z)

The above code recovers the whole network using CatBoost instead of kernel ridge with the RBF kernel (the default). Note that the parameter `regressor` is not defined for the `loco_mlcausality` function but it is defined for `mlcausality`, thus the parameter `regressor` is passed through to `mlcausality`.

### Available regressors
`mlcausality` admits the following regressors:
* 'krr' : kernel ridge regressor
* 'catboostregressor' : CatBoost regressor
* 'xgbregressor' : XGBoost regressor
* 'lgbmregressor' : LightGBM regressor
* 'randomforestregressor' : random forest regressor
* 'cuml_randomforestregressor' : random forest regressor using the cuML library
* 'linearregression' : linear regressor
* 'classic' : linear regressor in the classic sense (train == test == all data)
* 'svr' : Epsilon Support Vector Regression
* 'nusvr' : Nu Support Vector Regression
* 'cuml_svr' : Epsilon Support Vector Regression using the cuML library
* 'knn' : Regression based on k-nearest neighbors
* 'gaussianprocessregressor' : Gaussian process regressor
* 'gradientboostingregressor' : Gradient boost regressor
* 'histgradientboostingregressor' : Histogram-based Gradient Boosting Regression Tree
* 'default' : kernel ridge regressor with the RBF kernel set as default (default)

`multireg_mlcausality` admits the following regressors:
* 'krr' : kernel ridge regressor
* 'catboostregressor' : CatBoost regressor
* 'default' : kernel ridge regressor with the RBF kernel set as default (default)

Note that the CatBoost regressor option in `multireg_mlcausality` uses a different objective (`MultiRMSEWithMissingValues`) than those available for the CatBoost regressor in `mlcausality` (most notably `RMSE`) hence `multireg_mlcausality` with `regressor='catboostregressor'` will not be identical to `mlcausality` with `regressor='catboostregressor'`.

Note that not all regressors were fully tested for all parameter values. Less used regressors may not work for unscaled or non-normalized time-series.

Finally, regressors can be called with regressor-specific parameters using the `regressor_params` option, and they can be fitted with regressor-specific fit parameters using the `regressor_fit_params` option. For instance, the following recovers a network using CatBoost with 99 trees (instead of the CatBoost default of 1000) with no verbosity:

    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    z = mlcausality.loco_mlcausality(data, lags=[5,10], regressor='catboostregressor', regressor_params={'iterations':99}, regressor_fit_params={'verbose':False})
    print(z)

### Data preprocessing

__mlcausality__ comes with batteries included and you will typically not have to engage in substantial preprocessing before using the package. All functions support the usage of scalers or transformers from the __scikit-learn__ package at various stages of the Granger causality testing process:

* parameters `scaler_init_1` and `scaler_init_2` apply scalars to the input data and those transformations persist throughout the analysis. Predictions generated by the restricted and unrestricted models are not going to be inverse-transformed and the Granger causality analysis will be performed on the not inverse-transformed predictions and errors.
* parameters `scaler_prelogdiff_1`, `scaler_postlogdiff_2` `scaler_postlogdiff_1`, `scaler_prelogdiff_2`, `scaler_postsplit_1` and `scaler_postsplit_2` apply scalars to the input data and those transformations do not persist throughout the analysis. Predictions generated by the restricted and unrestricted models are going to be inverse-transformed and the Granger causality analysis will be performed on the inverse-transformed predictions and errors. The names of the parameters indicate the stage at which the transformation occurs with respect to taking a logdiff (see below) or splitting the dataset into a train and test set.
* parameter `logdiff` transforms, in a reversible way, the data by taking a log difference of all time-series. Predictions generated by the restricted and unrestricted models are going to be inverse-transformed and the Granger causality analysis will be performed on the inverse-transformed predictions and errors.
* parameters `scaler_dm_1` and `scaler_dm_2` apply scalars to the design matricies of the train and test data.

For additional clarity, note that the order in which transformations are applied is as follows:
init --> prelogdiff --> logdiff --> postlogdiff --> (data split occurs into test and train) --> postsplit --> dm

The following scalers and transformers are currently supported:
* ['maxabsscaler'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
* ['minmaxscaler'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
* ['powertransformer'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
* ['quantiletransformer'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
* ['robustscaler'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
* ['standardscaler'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
* ['normalizer'](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer) : Only available for `scaler_dm_1` and `scaler_dm_2`.

Finally, note that parameters for the above scalers are available in parameters named `*_params` where `*` stands for the name of the scaler. So `scaler_dm_1_params` are the params for `scaler_dm_1` etc.

The following usage example recovers a network by first applying a 'minmaxscaler' in the `[2,3]` range on the input data and then taking a log difference. Note that the MinMaxScaler is needed here because the input data is negative which would make taking a log difference impossible:

    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    z = mlcausality.multiloco_mlcausality(data,
        lags=[5,10],
        scaler_prelogdiff_1='minmaxscaler',
        scaler_prelogdiff_1_params={'feature_range':(2,3)},
        logdiff=True)
    print(z)

### Data splits

Data can be split into a test and train set using the `train_size` or `split` or `splits` parameters, as appropriate.

If `splits` is None and `train_size` is a float then `train_size` indicates the fraction of the data on which training occurs with the rest of the data ending up in the test set. Moreover, if `splits` is None and `train_size` is an integer greater than 1 then `train_size` indicates the number of observations to include in the training set. Finally, if `train_size` is equal to 1 then the train set and the test set are identical and equal to all the available data.

Other than controlling the data split using `train_size` one can instead provide a list of 2 lists to the `split` or `splits` parameter as appropriate. The first of the 2 lists would provide the indicies for the training set, while the second fo the 2 lists would provide the indicies for the test set. __All index lists must contain consecutive indicies with no gaps or holes otherwise lags will not be constructed correctly__.

Note that both train and test will lose lag number of observations. Moreover, if `logdiff` is True, an additional observation is lost from train and test because of the differencing operation.

The following provides an example of how to correctly use the `split` operator:

    import mlcausality
    import numpy as np
    import pandas as pd
    data = np.random.random([500,5])
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit()
    splits = list(tscv.split(data))
    split=splits[0]
    z = mlcausality.multiloco_mlcausality(data, lags=[5,10], split=split)
    print(z)

### Other important parameters
`y_bounds_violation_sign_drop` is an important Boolean parameter with implications for testing Granger causality using the sign test and the Wilcoxon signed rank test. If True, observations in the test set whose true values are outside [min(train), max(train)] are not used when calculating the test statistics and p-values of the sign and Wilcoxon tests (note: this also requires `y_bounds_error` to not be set to 'raise' in the `mlcausality` function). If False, then the sign and Wilcoxon test statistics and p-values are calculated using all observations in the test set. The default True because some models, especially tree-based models, extrapolate very poorly to the range of target values outside of what was seen in train.


### Additional help and documentation
Less commonly used features are documented using `help()`:

    import mlcausality
    help(mlcausality.loco_mlcausality)

<!-- 
_For more examples, please refer to the [Documentation](https://example.com)_
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
<!--
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/WojtekFulmyk/mlcausality/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
<!-- 
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/WojtekFulmyk/mlcausality](https://github.com/WojtekFulmyk/mlcausality)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- ACKNOWLEDGMENTS -->
<!-- 
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[contributors-url]: https://github.com/WojtekFulmyk/mlcausality/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[forks-url]: https://github.com/WojtekFulmyk/mlcausality/network/members
[stars-shield]: https://img.shields.io/github/stars/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[stars-url]: https://github.com/WojtekFulmyk/mlcausality/stargazers
[issues-shield]: https://img.shields.io/github/issues/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[issues-url]: https://github.com/WojtekFulmyk/mlcausality/issues
[license-shield]: https://img.shields.io/github/license/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[license-url]: https://github.com/WojtekFulmyk/mlcausality/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

[NumPy-url]: https://numpy.org
[SciPy-url]: https://scipy.org
[pandas-url]: https://pandas.pydata.org
[statsmodels-url]: https://www.statsmodels.org
[scikit-learn-url]: https://scikit-learn.org
[XGBoost-url]: https://xgboost.readthedocs.io
[LightGBM-url]: https://lightgbm.readthedocs.io
[CatBoost-url]: https://catboost.ai/
[cuML-url]: https://github.com/rapidsai/cuml


[NumPy-sheild]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[SciPy-sheild]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white
[Pandas-sheild]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

