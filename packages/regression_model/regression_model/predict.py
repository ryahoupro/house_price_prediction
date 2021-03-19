import numpy as np
import pandas as pd


from regression_model.processors.data_managers import load_pipeline
from regression_model.config import config
from regression_model.processors.validation import validate_inputs

from regression_model import __version__ as _version



filename= f"{config.SAVE_MODEL_NAME}{_version}.pkl"
my_pipeline = load_pipeline(filename)


def make_predictions(input_data) -> dict:
    """makes predictions using the saved model"""
    data = pd.read_json(input_data)
    valid_inputs = validate_inputs(data)    # validate data

    predictions = my_pipeline.predict(valid_inputs [config.FEATURES])

    output = np.exp(predictions)

    return {"predictions": output}