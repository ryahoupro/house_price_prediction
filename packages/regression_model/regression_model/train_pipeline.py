import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from processors.data_managers import load_dataset, save_pipeline

from pipeline import my_pipeline
from config import config


def run_training() -> None:
    """Train the model."""

    print('Training...')
    # read training data
    data = load_dataset(filename=config.TRAINING_DATA_FILE)

    # divide train and test
    # RANDOM_STATE = 0
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )

    # transform the target
    y_train = np.log(y_train)

    my_pipeline.fit(X_train[config.FEATURES], y_train)

    save_pipeline(pipeline=my_pipeline)


if __name__ == '__main__':
    run_training()