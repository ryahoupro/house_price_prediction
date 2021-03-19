import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


class MyPipeline:
    """
    Our custom pipieline
    """

    def __init__(self, target, cat_to_impute, year_var, num_to_impute, num_log, cat_to_encode, features,
                 test_size=0.1, random_state=0, tolerance=0.01, ref_year_var='YrSold'):

        # dataset
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # engineering params
        self.imputing_dict = {}
        self.frequent_cat_dict = {}
        self.encoding_dict = {}

        # scaling and model
        self.scaler = MinMaxScaler()
        self.model = Lasso(alpha=0.005, random_state=random_state)

        # variables to engineer
        self.target = target
        self.year_var = year_var
        self.cat_to_impute = cat_to_impute
        self.num_to_impute = num_to_impute
        self.num_log = num_log
        self.cat_to_encode = cat_to_encode
        self.features = features

        # parameters
        self.test_size = test_size
        self.random_state = random_state
        self.tolerance = tolerance
        self.ref_year_var = ref_year_var

    # functions to learn parameters from data

    def find_imputation_replacement(self):
        """
        finds value to be used for imputation
        :return: self
        """
        for variable in self.num_to_impute:
            replacement = self.X_train[variable].mode()[0]

            self.imputing_dict[variable] = replacement

        return self

    def find_frequent_cat(self):
        """
        finds list of frequent categories in categorical variables
        :return:self
        """

        for variable in self.cat_to_encode:
            tmp = self.X_train.groupby(variable)[self.target].count() / len(self.X_train)

            self.frequent_cat_dict[variable] = tmp[tmp > self.tolerance].index

        return self

    def find_cat_mappings(self):
        """
        create category to integer mappings for categorical enconding
        :return: self
        """
        for variable in self.cat_to_encode:
            ordered_labels = self.X_train.groupby(variable)[self.target].mean().sort_values().index

            ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}

            self.encoding_dict[variable] = ordinal_labels

        return self

    # functions to transform data


    def get_elapsed_time(self, df):
        """
        returns time elapsed between a temporal variable and a reference temporal variable
        :return: df
        """
        df = df.copy()

        df[self.year_var] = df[self.year_var] - df[self.ref_year_var]

        return df

    def remove_rare_labels(self, df):
        """
        tags rare labels from categorical variables
        :param df:
        :return: df
        """
        df = df.copy()

        for variable in self.cat_to_encode:
            df[variable] = np.where(df[variable].isin(self.frequent_cat_dict), df[variable], 'Rare')

        return df

    def encode_cat_vars(self, df):
        """
        replace categories by numbers
        :param df:
        :return: df
        """
        df = df.copy()

        for variable in self.encoding_dict:
            df[variable] = df[variable].map(self.encoding_dict[variable])

        return  df


    # functions that will make the feature engineering

    def fit(self, data):
        """
        pipeline to learn from the data, and fit the scaler and the model (lasso)
        :param data:
        :return:
        """
        # separate the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, data[self.target],
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)

        # find the imputation parameter (for numerical / mode)

        self.find_imputation_replacement()

        # impute missing data for categorical variables
        self.X_train[self.cat_to_impute] = self.X_train[self.cat_to_impute].fillna("Missing")

        self.X_test[self.cat_to_impute] = self.X_test[self.cat_to_impute].fillna("Missing")

        # impute missing data for numerical variables

        self.X_train[self.num_to_impute] = self.X_train[self.num_to_impute].fillna \
            (self.imputing_dict[self.num_to_impute[0]])

        self.X_test[self.num_to_impute] = self.X_test[self.num_to_impute].fillna(
            self.imputing_dict[self.num_to_impute[0]])

        # elapsed time

        self.X_train = self.get_elapsed_time(self.X_train)
        self.X_test = self.get_elapsed_time(self.X_test)

        # transform numerical variables (skewed to normal)
        self.X_train[self.num_log] = np.log(self.X_train[self.num_log])
        self.X_test[self.num_log] = np.log(self.X_test[self.num_log])

        # get frequent labels
        self.find_frequent_cat()

        # tag rare labels
        self.X_train = self.remove_rare_labels(self.X_train)
        self.X_test = self.remove_rare_labels(self.X_test)

        # find categorical mappings

        self.find_cat_mappings()

        # encode categories
        self.X_train = self.encode_cat_vars(self.X_train)
        self.X_test = self.encode_cat_vars(self.X_test)

        # train scaler
        self.scaler.fit(self.X_train[self.features])

        # scale variables
        self.X_train = self.scaler.transform(self.X_train[self.features])
        self.X_test = self.scaler.transform(self.X_test[self.features])

        print(self.X_train.shape, self.X_test.shape)

        # train model

        self.model.fit(self.X_train, np.log(self.y_train))

        return self

    def transform(self, data):
        """
        transforms raw data into engineered features
        :param data:
        :return: data
        """

        data = data.copy()

        # impute categorical
        data[self.cat_to_impute] = data[self.cat_to_impute].fillna('Missing')

        # numerical
        data[self.num_to_impute] = data[
            self.num_to_impute].fillna(
            self.imputing_dict[self.num_to_impute[0]])

        # capture elapsed time
        data = self.get_elapsed_time(df=data)

        # transform numerical variables
        data[self.num_log] = np.log(data[self.num_log])

        # remove rare labels
        data = self.remove_rare_labels(data)

        # encode categorical variables
        data = self.encode_cat_vars(data)

        # scale variables
        data = self.scaler.transform(data[self.features])

        return data

    def predict(self, data):
        """
        makes prediction
        :param data:
        :return: matrix of prediction
        """

        data = self.transform(data)
        predictions = self.model.predict(data)

        return np.exp(predictions)


    def evaluate(self):
        """
        evaluate predictions
        :return:
        """

        train_pred = self.model.predict(self.X_train)
        train_pred = np.exp(train_pred)

        print('train r2: {}'.format((r2_score(self.y_train, train_pred))))

        test_pred = self.model.predict(self.X_test)
        test_pred = np.exp(test_pred)

        print('test r2: {}'.format((r2_score(self.y_test, test_pred))))

        return 1


