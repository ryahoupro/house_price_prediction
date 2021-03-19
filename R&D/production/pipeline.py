import pandas as pd
import numpy as np

from preprocessors import MyPipeline
import config

import warnings
warnings.filterwarnings("ignore")


my_pipeline = MyPipeline(target= config.TARGET,
                         cat_to_impute= config.CATEGORICAL_TO_IMPUTE,
                         year_var= config.YEAR_VARIABLE,
                         num_log= config.NUMERICAL_LOG,
                         num_to_impute= config.NUMERICAL_TO_INPUT,
                         cat_to_encode= config.CATEGORICAL_ENCODE,
                         features= config.FEATURES)


if __name__ == '__main__':

    data = pd.read_csv(config.PATH_TO_DATASET)

    print("Fitting the data")
    my_pipeline.fit(data)
    print("Evaluating model performance")
    my_pipeline.evaluate()
    print("\n")
    print("Some predictions :")
    predictions = my_pipeline.predict(data)
    print(predictions)
