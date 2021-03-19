import math

from regression_model.processors.data_managers import load_dataset
from regression_model.predict import make_predictions


def test_make_predictions():
    """to test make_prediction module"""

    test_data = load_dataset(filename='test.csv')
    test_json = test_data[0:1].to_json(orient='records')

    subject = make_predictions(test_json)

    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert math.ceil(subject.get('predictions')[0]) == 112476



