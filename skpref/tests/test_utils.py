import unittest
from numpy.testing import assert_array_equal
from skpref.utils import aggregate_full_probability_predictions_to_discrete_choice
from skpref.task import ChoiceTask
from skpref.tests.shared_test_dataframes import discrete_choice_data
from skpref.data_processing import SubsetPosetVec


class TestAggregatorToDC(unittest.TestCase):

    def test_aggregator(self):
        dummy_predictions = {'a': [0.6, 0.1], 'b': [0.3, 0.1], 'c': [0.1, 0.8]}
        dummy_task = ChoiceTask(discrete_choice_data, 'alt', features_to_use=None)
        outcome = aggregate_full_probability_predictions_to_discrete_choice(
            dummy_predictions, dummy_task)

        expected_outcome = SubsetPosetVec(top_input_data=['a', 'c'],
                                          boot_input_data=[['b', 'c'],
                                                           ['a', 'b']])
        assert_array_equal(outcome.top_input_data,
                           expected_outcome.top_input_data)

        assert_array_equal(outcome.boot_input_data,
                           expected_outcome.boot_input_data)
