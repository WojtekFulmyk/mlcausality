from setuptools import setup

from mlcausality import __version__

setup(
    name='mlcausality',
    version=__version__,
    
    url='https://github.com/WojtekFulmyk/mlcausality',
    author='Wojciech Fulmyk',
    author_email='wfulmyk@proton.me',
    
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'statsmodels',
        'scikit-learn',
        'tqdm'
    ],
    
    extras_require={
        'catboost': [
            'catboost',
        ],
        'xgboost' : [
            'xgboost',
        ],
        'lightgbm': [
            'lightgbm',
        ],
        'all': [
            'catboost', 'xgboost', 'lightgbm',
        ],
    },
    
    py_modules=['mlcausality'],
)
