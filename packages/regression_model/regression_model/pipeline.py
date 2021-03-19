# libraries
from sklearn.pipeline import Pipeline
from regression_model.processors import preprocessors as pp
from regression_model.config import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso


PIPELINE_NAME = 'lasso_regresion'

my_pipeline = Pipeline(
    [
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CAT_VARS_WITH_NA),
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUM_VARS_WITH_NA),
        ),
        (
            "temporal_variable",
            pp.TemporalVariablesCalculator(
                variables=config.TEMPORAL_VARS, reference_variable=config.DROP_FEATURES
            ),
        ),
        (
            "rare_label_encoder",
            pp.RareLabelCategoricalEncoder(tolerance=0.01, variables=config.CAT_VARS),
        ),
        (
            "categorical_encoder",
            pp.CategoricalEncoder(variables=config.CAT_VARS),
        ),
        ("log_transformer", pp.LogTransformer(variables=config.NUM_LOG)),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
        ),
        ("scaler", MinMaxScaler()),
        ("Linear_model", Lasso(alpha=0.005, random_state=0)),
    ]
)