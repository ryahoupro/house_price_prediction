import pandas as pd
import joblib
import logging

import sys
sys.path.append("..")

from regression_model.config import config
from regression_model import __version__ as version

logger = logging.getLogger(__name__)


def load_dataset(filename):
    """load the dataset into a pandas dataframe"""
    data = pd.read_csv(config.DATASET_DIR / filename)
    return data


def save_pipeline(pipeline) -> None :
    """saves the trained model in dir trained_models"""

    filename = config.SAVE_MODEL_NAME + version + ".pkl"

    path = config.TRAINED_MODEL_DIR / filename

    remove_old_model(filename=filename)

    joblib.dump(pipeline, path)

    logger.info("file {} saved".format(filename))

    return None


def load_pipeline(filename):
    """loads the model from filename"""

    path = config.TRAINED_MODEL_DIR / filename
    my_pipeline = joblib.load(path)

    return my_pipeline


def remove_old_model(filename):
    """removes old pipeline. Takes in argument the name of the file to keep"""
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [filename, "__init__.py"]:
            model_file.unlink()




