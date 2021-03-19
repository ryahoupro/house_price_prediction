# libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np



class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Categorical data missing values imputer
    """
    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, Y :pd.Series = None) -> 'CategoricalImputer' :
        """Fit statement to accomodate the pipe"""
        return self

    def transform(self, X: pd.DataFrame) -> 'pd.DataFrame':
        """Apply the transformations to the dataframe"""
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Numerical data missing value imputer
    """
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """returns dictionnary with key variable name and value imputer (mode)"""
        self.dict_imputer_ = {}
        for variable in self.variables:
            self.dict_imputer_[variable] = X[variable].mode()[0]
        return  self

    def transform(self, X):
        """fills X missing values with the mode. Returns X"""
        X = X.copy()

        for variable in self.variables:
            X[variable].fillna(self.dict_imputer_[variable], inplace=True)

        return X


class TemporalVariablesCalculator(BaseEstimator, TransformerMixin):
    """ Temporal values calculator"""

    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        """ step needed for the sklearn pipeline"""
        return self

    def transform(self, X, y=None):
        """creates new variables"""
        X = X.copy()

        for variable in self.variables:
            X[variable] = X[variable] - X[self.reference_variable]

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """ rare label categorical encoder"""

    def __init__(self, variables, tolerance=0.05):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.tolerance = tolerance

    def fit(self, X, y=None):
        """fills dict_encoder"""
        self.dict_encoder = {}

        for variable in self.variables:
            tmp = pd.Series(X[variable].value_counts()/ np.float(len(X)))

            self.dict_encoder[variable] = list(tmp[ tmp >= self.tolerance ].index)

        return self

    def transform(self, X):
        """ tags rare labels with rare"""
        X = X.copy()

        for variable in self.variables:
            X[variable] = np.where(X[variable].isin(self.dict_encoder[variable]), X[variable], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """encode categorical variables"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        """creates dictionnary self.encoder_dict that transforms string data to numerical data"""

        tmp = pd.concat([X,y], axis=1)
        tmp.columns = list(X.columns) + ["target"]

        self.encoder_dict = {}

        for variable in self.variables:
            t = tmp.groupby(variable)["target"].mean().sort_values(ascending=True).index
            self.encoder_dict[variable] = {k: i for i,k in enumerate(t,0)}



        return self

    def transform(self, X):
        """ encodes labels"""
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].map(self.encoder_dict[var])

        return  X


class LogTransformer(BaseEstimator, TransformerMixin):
    """logarithm transformer for numerical variables"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """to accomodate the pipeline"""
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            if any(X[var] <= 0):
                raise ValueError(
                    f"Variables contain zero or negative values, "
                    f"can't apply log for vars: {var}")
            else:
                X[var] = np.log(X[var])

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X














