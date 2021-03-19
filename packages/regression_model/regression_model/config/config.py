# libraries
import pathlib

import regression_model

PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "dataset"

SAVE_MODEL_NAME = "regression_model"

# data
DATA_FILE = "houseprice.csv"
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "SalePrice"



# variables
FEATURES = [
    "MSSubClass","MSZoning",
    "Neighborhood","OverallQual",
    "OverallCond", "YearRemodAdd",
    "RoofStyle", "MasVnrType",
    "BsmtQual", "BsmtExposure",
    "HeatingQC", "CentralAir",
    "1stFlrSF", "GrLivArea",
    "BsmtFullBath", "KitchenQual",
    "Fireplaces", "FireplaceQu",
    "GarageType", "GarageFinish",
    "GarageCars", "PavedDrive",
    "LotFrontage","YrSold", ]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = "YrSold"

# numerical variables with NA in train set
NUM_VARS_WITH_NA = ["LotFrontage"]

# categorical variables with NA in train set
CAT_VARS_WITH_NA = [
    "MasVnrType", "BsmtQual",
    "BsmtExposure", "FireplaceQu",
    "GarageType", "GarageFinish"]

TEMPORAL_VARS = "YearRemodAdd"

# variables to log transform
NUM_LOG = ["LotFrontage", "1stFlrSF", "GrLivArea"]

# categorical variables to encode
CAT_VARS = [
    "MSZoning",
    "Neighborhood",
    "RoofStyle",
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CAT_VARS + NUM_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CAT_VARS if feature not in CAT_VARS_WITH_NA
]