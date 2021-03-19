import pandas as pd

from regression_model.config import config


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """check if input_data is in the right format"""
    validated_data = input_data.copy()

    # drop nan
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    # check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )

    # check for values <= 0 for the log transformed variables
    if (input_data[config.NUM_LOG] <= 0).any().any():
        vars_with_neg_values = config.NUM_LOG[
            (input_data[config.NUM_LOG] <= 0).any()
        ]
        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]

    return validated_data
