import unittest
from skpref.task import ChoiceTask, HeterogeneousDataError
import numpy as np
import pandas as pd

DATA_c = pd.DataFrame({'ent1': ['C', 'B', 'C', 'D'],
                       'ent2': ['B', 'A', 'D', 'C'],
                       'result': [1, 0, 0, 1]})

DATA = pd.DataFrame({'alternatives': [['B', 'C'], ['A', 'B'], ['D', 'C'],
                                      ['C', 'D']],
                     'result': ['C', 'A', 'D', 'D']})

DATA_het = pd.DataFrame({'alternatives': [['B', 'C'], ['A', 'B'], ['D', 'C'],
                                          ['C', 'D', 'A']],
                         'result': ['C', 'A', 'D', 'D']})


class TestChoiceTask(unittest.TestCase):

    def test_pandas_df_correspondence(self):
        corr_choicetask = ChoiceTask(
            primary_table=DATA_c,
            primary_table_alternatives_names=['ent1', 'ent2'],
            primary_table_target_name='result',
            target_column_correspondence='ent1')

        correct_top_array = np.array([['C'], ['A'], ['D'], ['D']])

        correct_boot_array = np.array([['B'], ['B'], ['C'], ['C']])

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.top_input_data,
            correct_top_array)

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.boot_input_data,
            correct_boot_array)

    def test_pandas_df_correspondence2(self):
        corr_choicetask = ChoiceTask(
            primary_table=DATA_c,
            primary_table_alternatives_names=['ent1', 'ent2'],
            primary_table_target_name='result',
            target_column_correspondence='ent2')

        correct_boot_array = np.array([['C'], ['A'], ['D'], ['D']])

        correct_top_array = np.array([['B'], ['B'], ['C'], ['C']])

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.top_input_data,
            correct_top_array)

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.boot_input_data,
            correct_boot_array)

    def test_with_choice_vector_data(self):
        choicetask = ChoiceTask(
            primary_table=DATA,
            primary_table_alternatives_names='alternatives',
            primary_table_target_name='result')

        correct_top_array = np.array([['C'], ['A'], ['D'], ['D']])

        correct_boot_array = np.array([['B'], ['B'], ['C'], ['C']])

        np.testing.assert_array_equal(
            choicetask.subset_vec.top_input_data,
            correct_top_array)

        np.testing.assert_array_equal(
            choicetask.subset_vec.boot_input_data,
            correct_boot_array)

    def test_het_error_is_given(self):
        with self.assertRaises(HeterogeneousDataError):
            ChoiceTask(
                primary_table=DATA_het,
                primary_table_alternatives_names='alternatives',
                primary_table_target_name='result')