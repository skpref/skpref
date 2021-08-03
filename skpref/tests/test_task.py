import unittest
from skpref.task import ChoiceTask, _table_reader, PairwiseComparisonTask
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from scipy.io import arff
from io import StringIO
from pandas.testing import assert_frame_equal
from skpref.tests.shared_test_dataframes import *

class TestTableReader(unittest.TestCase):

    def test_arff_format(self):
        content = """
@relation foo
@attribute width  numeric
@attribute height numeric
@attribute colour  {red,green,blue,yellow,black}
@data
5.0,3.25,blue
4.5,3.75,green
3.0,4.00,red
        """
        f = StringIO(content)
        data, meta = arff.loadarff(f)

        expected_result = pd.DataFrame(
            [[5.0, 3.25, b'blue'],
             [4.5, 3.75, b'green'],
             [3.0, 4.0, b'red']],
            columns=['width', 'height', 'colour']
        )

        table, name, hook = _table_reader(data)

        assert_frame_equal(table, expected_result)

    def test_error_is_given_for_random_numpy_array(self):
        with self.assertRaises(Exception):
            _table_reader(np.array([1, 2, 3]))


class TestPrefTask(unittest.TestCase):
    def test_find_merge_columns(self):
        # Define a PairwiseComparisonTask
        test_pairwise_task = PairwiseComparisonTask(
            DATA, ['ent1', 'ent2'], 'result', 'ent1', ENT1_ATTRIBUTES,
            {'ent1': ['ent1', 'ent2']}
        )
        secondary_re_indexed_returned, input_merge_cols_returned, \
        left_on_returned, right_on_returend \
            = test_pairwise_task.find_merge_columns()

        correct_secondary_reindexed = ENT1_ATTRIBUTES.set_index('ent1').copy()
        correct_input_merge_cols = []
        correct_left_on = np.array(['ent1', 'ent2'])
        correct_right_on = np.array(['ent1'])

        assert_frame_equal(correct_secondary_reindexed,
                           secondary_re_indexed_returned)
        self.assertListEqual(correct_input_merge_cols,
                             input_merge_cols_returned)
        assert_array_equal(correct_left_on, left_on_returned)
        assert_array_equal(correct_right_on, right_on_returend)

        # Try it with more than one merge column

    def test_find_merge_columns_with_input_merge_columns_populated(self):
        test_pairwise_task = PairwiseComparisonTask(
            DATA_c_several_merge_cols, ['ent1', 'ent2'], 'result', 'ent1',
            ENT1_ATTRIBUTES_several_merge_cols,
            {'ent1': ['ent1', 'ent2'], 'season': 'season'}
        )
        secondary_re_indexed_returned, input_merge_cols_returned, \
        left_on_returned, right_on_returend \
            = test_pairwise_task.find_merge_columns()

        correct_secondary_reindexed = \
            ENT1_ATTRIBUTES_several_merge_cols.set_index('ent1').copy()

        correct_input_merge_cols = ['season']
        correct_left_on = np.array(['ent1', 'ent2', 'season'])
        correct_right_on = np.array(['ent1', 'season'])

        assert_frame_equal(correct_secondary_reindexed,
                           secondary_re_indexed_returned)
        self.assertListEqual(correct_input_merge_cols,
                             input_merge_cols_returned)
        assert_array_equal(correct_left_on, left_on_returned)
        assert_array_equal(correct_right_on, right_on_returend)

    def test_find_merge_columns_for_choice_task(self):
        test_choice_task = ChoiceTask(SUBSET_CHOICE_TABLE,
                                      'alternatives',
                                      'choice',
                                      SUBSET_CHOICE_FEATS_TABLE,
                                      {'ID': ['alternatives', 'choice']}
                                      )

        secondary_re_indexed_returned, input_merge_cols_returned, \
        left_on_returned, right_on_returend \
            = test_choice_task.find_merge_columns()

        correct_secondary_reindexed = \
            SUBSET_CHOICE_FEATS_TABLE.set_index('ID').copy()

        correct_input_merge_cols = []
        correct_left_on = np.array(['alternatives', 'choice'])
        correct_right_on = np.array(['ID'])

        assert_frame_equal(correct_secondary_reindexed,
                           secondary_re_indexed_returned)
        self.assertListEqual(correct_input_merge_cols,
                             input_merge_cols_returned)
        assert_array_equal(correct_left_on, left_on_returned)
        assert_array_equal(correct_right_on, right_on_returend)

    def test_find_merge_columns_for_choice_task_mergecol(self):
        test_choice_task = ChoiceTask(SUBSET_CHOICE_TABLE_season,
                                      'alternatives',
                                      'choice',
                                      SUBSET_CHOICE_FEATS_TABLE_season,
                                      {'ID': ['alternatives', 'choice'],
                                       'season': 'season'}
                                      )

        secondary_re_indexed_returned, input_merge_cols_returned, \
        left_on_returned, right_on_returend \
            = test_choice_task.find_merge_columns()

        correct_secondary_reindexed = \
            SUBSET_CHOICE_FEATS_TABLE_season.set_index('ID').copy()

        correct_input_merge_cols = ['season']
        correct_left_on = np.array(['alternatives', 'choice', 'season'])
        correct_right_on = np.array(['ID', 'season'])

        assert_frame_equal(correct_secondary_reindexed,
                           secondary_re_indexed_returned)
        self.assertListEqual(correct_input_merge_cols,
                             input_merge_cols_returned)
        assert_array_equal(correct_left_on, left_on_returned)
        assert_array_equal(correct_right_on, right_on_returend)


class TestChoiceTask(unittest.TestCase):
    def test_with_choice_vector_data(self):
        choicetask = ChoiceTask(
            primary_table=DATA2,
            primary_table_alternatives_names='alternatives',
            primary_table_target_name='result',
            features_to_use=None
        )

        correct_top_array = np.array([['C'], ['A'], ['D'], ['D']])

        correct_boot_array = np.array([['B'], ['B'], ['C'], ['C']])

        np.testing.assert_array_equal(
            choicetask.subset_vec.top_input_data,
            correct_top_array)

        np.testing.assert_array_equal(
            choicetask.subset_vec.boot_input_data,
            correct_boot_array)

    def test_wih_heterogenous_data(self):
        choicetask = ChoiceTask(
            primary_table=DATA_het,
            primary_table_alternatives_names='alternatives',
            primary_table_target_name='result',
            features_to_use=None
        )

        correct_top_array = np.array([np.array(['C', 'B']),
                                      np.array(['A']),
                                      np.array(['D']),
                                      np.array(['D'])])

        correct_boot_array = np.array([np.array([], dtype='<U1'),
                                       np.array(['B']),
                                       np.array(['C']),
                                       np.array(['C', 'A'])])

        for i in range(len(correct_top_array)):
            np.testing.assert_array_equal(
                choicetask.subset_vec.top_input_data[i],
                correct_top_array[i])

        for j in range(len(correct_boot_array)):
            np.testing.assert_array_equal(
                choicetask.subset_vec.boot_input_data[j],
                correct_boot_array[j])


class TestPairwiseComparisonTask(unittest.TestCase):

    def test_pandas_df_correspondence(self):
        corr_choicetask = PairwiseComparisonTask(
            primary_table=DATA,
            primary_table_alternatives_names=['ent1', 'ent2'],
            primary_table_target_name='result',
            target_column_correspondence='ent1',
            features_to_use=None
        )

        correct_top_array = np.array([['C'], ['A'], ['D'], ['D']])

        correct_boot_array = np.array([['B'], ['B'], ['C'], ['C']])

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.top_input_data,
            correct_top_array)

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.boot_input_data,
            correct_boot_array)

    def test_pandas_df_correspondence2(self):
        corr_choicetask = PairwiseComparisonTask(
            primary_table=DATA,
            primary_table_alternatives_names=['ent1', 'ent2'],
            primary_table_target_name='result',
            target_column_correspondence='ent2',
            features_to_use=None
        )

        correct_boot_array = np.array([['C'], ['A'], ['D'], ['D']])

        correct_top_array = np.array([['B'], ['B'], ['C'], ['C']])

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.top_input_data,
            correct_top_array)

        np.testing.assert_array_equal(
            corr_choicetask.subset_vec.boot_input_data,
            correct_boot_array)

    def test_correct_typing(self):
        corr_choicetask = PairwiseComparisonTask(
            primary_table=DATA,
            primary_table_alternatives_names=['ent1', 'ent2'],
            primary_table_target_name='result',
            target_column_correspondence='ent2',
            features_to_use=None
        )

        self.assertEqual(corr_choicetask.subset_vec.poset_type.top_size_const,
                         True)
        self.assertEqual(corr_choicetask.subset_vec.poset_type.boot_size_const,
                         True)
        self.assertEqual(corr_choicetask.subset_vec.poset_type.top_size, 1)
        self.assertEqual(corr_choicetask.subset_vec.poset_type.boot_size, 1)
