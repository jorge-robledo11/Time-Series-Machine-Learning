# Ingeniería de variables
# ==============================================================================
import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.timeseries.forecasting import WindowFeatures, ExpandingWindowFeatures, LagFeatures
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.preprocessing import MatchCategories
from feature_engine.discretisation import EqualFrequencyDiscretiser

# Preprocesadores
# ==============================================================================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


class CustomColumnsScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables, scaler='standard'):
        self.variables = variables
        scalers = {
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler(),
            'standard': StandardScaler()
        }
        self.scaler = scalers.get(scaler.lower(), StandardScaler())
        
    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X[self.variables])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.variables] = self.scaler.transform(X[self.variables])
        return X_scaled
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Pipelines
# ==============================================================================
def pipeline_linear_models(data: pd.DataFrame, tmp: list, target: str, n_windows=3) -> pd.DataFrame:
    
    # Generating the list with the desired output
    windows_mean = [f'window_{w}_mean' for w in range(1, n_windows+1)] + ['expanding_mean']

    # List comprehension to combine target variable with each suffix
    target_windows_mean = [f'{target}_{w}' for w in windows_mean]
    
    # Datetime features
    dtf = DatetimeFeatures(
        variables='index',
        features_to_extract=tmp
    )

    # Window features
    winf = WindowFeatures(
        variables=[target],
        window=list(range(1, n_windows+1)),
        missing_values='ignore',
    )

    # Expanding Window features
    expnd_winf = ExpandingWindowFeatures(
        variables=[target],
        functions=['mean'],
        missing_values='ignore'
    )

    # Periodic features
    cyclicf = CyclicalFeatures(
        variables=['month']
    )

    # Casting to categoricals
    categoricals = MatchCategories(
        ignore_format=True,
        missing_values='ignore',
        variables=tmp
    )

    # Encoding
    ohe = OneHotEncoder(
        variables=tmp
    )

    # Drop missing data
    imputer = DropMissingData()
    
    # Scale some features
    scaler = CustomColumnsScaler(
        variables=target_windows_mean
    )

    # Pipeline
    pipe = Pipeline([
        ('datetime_features', dtf),
        ('winf', winf),
        ('exp_winf', expnd_winf),
        ('periodic', cyclicf),
        ('cat_features', categoricals),
        ('encoding', ohe),
        ('dropna', imputer),
        ('scaler', scaler)
    ])

    # Ajustar los datos a la Pipeline
    data_transformada = pipe.fit_transform(data)

    return data_transformada



def pipeline_no_linear_models(data: pd.DataFrame, tmp: list, target: str, n_windows=3) -> pd.DataFrame:
    
    # Generating the list with the desired output
    windows_mean = [f'window_{w}_mean' for w in range(1, n_windows+1)] + ['expanding_mean']

    # List comprehension to combine target variable with each suffix
    target_windows_mean = [f'{target}_{w}' for w in windows_mean]
    
    # Datetime features
    dtf = DatetimeFeatures(
        variables='index',
        features_to_extract=tmp
    )

    # Window features
    winf = WindowFeatures(
        variables=[target],
        window=list(range(1, n_windows+1)),
        missing_values='ignore',
    )

    # Expanding Window features
    expnd_winf = ExpandingWindowFeatures(
        variables=[target],
        functions=['mean'],
        missing_values='ignore'
    )

    # Periodic features
    cyclicf = CyclicalFeatures(
        variables=tmp
    )
    
    # Discretiser
    discretiser = EqualFrequencyDiscretiser(
        variables=target_windows_mean
    )
    
    # Drop missing data
    imputer = DropMissingData()
    
    # Casting to categoricals
    categoricals = MatchCategories(
        ignore_format=True,
        missing_values='ignore',
        variables=tmp + target_windows_mean
    )

    # Encoding
    encoder = OrdinalEncoder(
        ignore_format=True,
        missing_values='ignore',
        variables=tmp + target_windows_mean
    )
    
    # Pipeline
    pipe = Pipeline([
        ('datetime_features', dtf),
        ('winf', winf),
        ('exp_winf', expnd_winf),
        ('periodic', cyclicf),
        ('dropna', imputer),
        ('discretiser', discretiser),
        ('cat_features', categoricals)
    ])

    # Ajustar los datos a la Pipeline
    data_transformada = pipe.fit_transform(data, data[target])
    data_transformada = encoder.fit_transform(data_transformada, data_transformada[target]) # Relación monotónica

    return data_transformada