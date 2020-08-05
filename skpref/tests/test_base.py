import unittest
import numpy as np
from numpy.testing import assert_array_equal
from skpref.task import ChoiceTask, PairwiseComparisonTask
from skpref.base import (PairwiseComparisonModel, SVMPairwiseComparisonModel,
                         ClassificationReducer)
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier
from skpref.tests.shared_test_dataframes import *


class TestPairwiseComparisonModelFunctions(unittest.TestCase):

    def test_task_indexing(self):
        test_task = ChoiceTask(SUBSET_CHOICE_TABLE, 'alternatives', 'choice',
                               secondary_table=SUBSET_CHOICE_FEATS_TABLE,
                               secondary_to_primary_link={
                                   'ID': ['choice', 'alternatives']},
                               features_to_use='feat1')

        mybt = PairwiseComparisonModel()

        d1, d2, _ = mybt.task_indexing(test_task)

        correct_d1 = test_task.subset_vec.pairwise_reducer() \
            .set_index(['alt1', 'alt2'])

        correct_d2 = SUBSET_CHOICE_FEATS_TABLE.set_index('ID')
        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))
        assert_frame_equal(d2.astype('int32'), correct_d2.astype('int32'))

    def test_task_indexing_id_merge(self):
        test_task = ChoiceTask(SUBSET_CHOICE_TABLE_season, 'alternatives',
                               'choice',
                               secondary_table=SUBSET_CHOICE_FEATS_TABLE_season,
                               secondary_to_primary_link={
                                   'ID': ['choice', 'alternatives'],
                                   'season': 'season'},
                               features_to_use=['feat1'])

        mybt = PairwiseComparisonModel()

        d1, d2, _id = mybt.task_indexing(test_task)

        correct_d1 = test_task.subset_vec.pairwise_reducer() \
            .set_index(['alt1', 'alt2'])

        correct_d1['season'] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
                                8, 8, 8]

        correct_d2 = SUBSET_CHOICE_FEATS_TABLE_season.set_index('ID')
        correct_id = ['season']

        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))
        assert_frame_equal(d2.astype('int32'), correct_d2.astype('int32'))
        self.assertListEqual(_id, correct_id)

    def test_task_indexing_primary_feat_merge(self):
        test_task = ChoiceTask(SUBSET_CHOICE_TABLE_season, 'alternatives',
                               'choice',
                               secondary_table=SUBSET_CHOICE_FEATS_TABLE,
                               secondary_to_primary_link={
                                   'ID': ['choice', 'alternatives']},
                               features_to_use=['feat1', 'season'])

        mybt = PairwiseComparisonModel()

        d1, d2, _id = mybt.task_indexing(test_task)

        correct_d1 = test_task.subset_vec.pairwise_reducer() \
            .set_index(['alt1', 'alt2'])

        correct_d1['season'] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
                                8, 8, 8]

        correct_d2 = SUBSET_CHOICE_FEATS_TABLE.set_index('ID')
        correct_id = []

        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))
        assert_frame_equal(d2.astype('int32'), correct_d2.astype('int32'))
        self.assertListEqual(_id, correct_id)

    def test_task_indexing_id_also_feat_merge(self):
        test_task = ChoiceTask(SUBSET_CHOICE_TABLE_season, 'alternatives',
                               'choice',
                               secondary_table=SUBSET_CHOICE_FEATS_TABLE_season,
                               secondary_to_primary_link={
                                   'ID': ['choice', 'alternatives'],
                                   'season': 'season'},
                               features_to_use=['feat1', 'season'])

        mybt = PairwiseComparisonModel()

        d1, d2, _id = mybt.task_indexing(test_task)

        correct_d1 = test_task.subset_vec.pairwise_reducer() \
            .set_index(['alt1', 'alt2'])

        correct_d1['season'] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
                                8, 8, 8]

        correct_d2 = SUBSET_CHOICE_FEATS_TABLE_season.set_index('ID')
        correct_id = ['season']

        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))
        assert_frame_equal(d2.astype('int32'), correct_d2.astype('int32'))
        self.assertListEqual(_id, correct_id)

    def test_task_indexing_pairwise_input(self):
        test_task = PairwiseComparisonTask(
            DATA, ['ent1', 'ent2'], 'result', 'ent1', features_to_use=None)

        mybt = PairwiseComparisonModel()

        d1, _, _ = mybt.task_indexing(test_task)

        correct_d1 = DATA.set_index(['ent1', 'ent2'])\
            .rename(columns={'result': 'alt1_top'})

        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))

    def test_task_indexing_pairwise_input_ents(self):
        test_task = PairwiseComparisonTask(
                DATA, ['ent1', 'ent2'], 'result', 'ent1',
                secondary_table=ENT1_ATTRIBUTES,
                secondary_to_primary_link={'ent1': ['ent1', 'ent2']})

        mybt = PairwiseComparisonModel()

        d1, d2, _ = mybt.task_indexing(test_task)

        correct_d1 = DATA.set_index(['ent1', 'ent2']) \
            .rename(columns={'result': 'alt1_top'})

        correct_d2 = ENT1_ATTRIBUTES.set_index('ent1')

        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))
        assert_frame_equal(d2.astype('int32'), correct_d2.astype('int32'))

    def test_fit_task_unpacker(self):
        test_task = PairwiseComparisonTask(
            DATA, ['ent1', 'ent2'], 'result', 'ent1',
            secondary_table=ENT1_ATTRIBUTES,
            secondary_to_primary_link={'ent1': ['ent1', 'ent2']})

        mybt = PairwiseComparisonModel()

        ret_dict = mybt.task_unpacker(test_task)
        correct_d1 = DATA.set_index(['ent1', 'ent2']) \
            .rename(columns={'result': 'alt1_top'})
        correct_d2 = ENT1_ATTRIBUTES.set_index('ent1')
        correct_ret_dict = {
            'df_comb': correct_d1,
            'target': 'alt1_top',
            'df_i': correct_d2,
            'merge_columns': None
        }

        self.assertListEqual(list(ret_dict.keys()),
                             list(correct_ret_dict.keys()))
        assert_frame_equal(ret_dict['df_comb'], correct_d1)
        self.assertEqual(correct_ret_dict['target'], 'alt1_top')
        assert_frame_equal(ret_dict['df_i'], correct_d2)
        self.assertEqual(correct_ret_dict['merge_columns'], None)


class TestSVMPairwiseComparisonModel(unittest.TestCase):

    def test_correct_indexing_choice(self):
        test_task = ChoiceTask(SUBSET_CHOICE_TABLE, 'alternatives', 'choice')

        mybt = SVMPairwiseComparisonModel()

        d1, _, _ = mybt.task_indexing(test_task)

        correct_d1 = pd.DataFrame(
            [
                [0, 512709, 490972, 1],
                [0, 490972, 512709, 0],
                [0, 512709, 685450, 1],
                [0, 685450, 512709, 0],
                [0, 512709, 5549502, 1],
                [0, 5549502, 512709, 0],
                [0, 529703, 490972, 1],
                [0, 490972, 529703, 0],
                [0, 529703, 685450, 1],
                [0, 685450, 529703, 0],
                [0, 529703, 5549502, 1],
                [0, 5549502, 529703, 0],
                [0, 696056, 490972, 1],
                [0, 490972, 696056, 0],
                [0, 696056, 685450, 1],
                [0, 685450, 696056, 0],
                [0, 696056, 5549502, 1],
                [0, 5549502, 696056, 0],
                [1, 723354, 550707, 1],
                [1, 550707, 723354, 0],
                [1, 723354, 551375, 1],
                [1, 551375, 723354, 0],
                [1, 723354, 591842, 1],
                [1, 591842, 723354, 0],
                [1, 723354, 601195, 1],
                [1, 601195, 723354, 0],
                [1, 723354, 732624, 1],
                [1, 732624, 723354, 0],
                [1, 723354, 778197, 1],
                [1, 778197, 723354, 0],
                [1, 723354, 813892, 1],
                [1, 813892, 723354, 0],
                [1, 723354, 817040, 1],
                [1, 817040, 723354, 0],
                [1, 723354, 576214, 1],
                [1, 576214, 723354, 0],
                [1, 723354, 673995, 1],
                [1, 673995, 723354, 0]
            ], columns=['observation', 'alt1', 'alt2', 'alt1_top']
        ).set_index(['alt1', 'alt2'])

        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))

    def test_correct_indexing_pairwise(self):
        test_task = PairwiseComparisonTask(
            DATA, ['ent1', 'ent2'], 'result', 'ent1')

        mybt = SVMPairwiseComparisonModel()
        d1, _, _ = mybt.task_indexing(test_task)

        correct_d1 = pd.DataFrame(
            {'observation': [0, 0, 1, 1, 2, 2, 3, 3],
             'alt1': ['C', 'B', 'A', 'B', 'D', 'C', 'D', 'C'],
             'alt2': ['B', 'C', 'B', 'A', 'C', 'D', 'C', 'D'],
             'alt1_top': [1, 0, 1, 0, 1, 0, 1, 0]}
        ).set_index(['alt1', 'alt2'])
        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))


class TestClassificationReducer(unittest.TestCase):

    def test_task_unpacker_pairwise(self):
        test_task = PairwiseComparisonTask(
            DATA, ['ent1', 'ent2'], 'result', 'ent1',
            secondary_table=ENT1_ATTRIBUTES,
            secondary_to_primary_link={'ent1': ['ent1', 'ent2']})

        my_model = ClassificationReducer(
            DummyClassifier('constant', constant=1))

        unpacker_dict = my_model.task_unpacker(test_task)

        correct_output_data = pd.DataFrame({
            'chosen': [1, 0, 0, 1],
            'feat1_ent1': [12, 11, 12, 15],
            'feat1_ent2': [11, 1, 15, 12]
        })

        assert_frame_equal(unpacker_dict['df_comb'], correct_output_data)

    def test_task_unpacker_diff_pairwise(self):
        test_task = PairwiseComparisonTask(
            DATA, ['ent1', 'ent2'], 'result', 'ent1',
            secondary_table=ENT1_ATTRIBUTES,
            secondary_to_primary_link={'ent1': ['ent1', 'ent2']})

        my_model = ClassificationReducer(
            DummyClassifier('constant', constant=1))

        unpacker_dict = my_model.task_unpacker(test_task,
                                               take_feautre_diff=True)

        correct_output_data = pd.DataFrame({
            'chosen': [1, 0, 0, 1],
            'feat1_diff': [1, 10, -3, 3]
        })

        assert_frame_equal(unpacker_dict['df_comb'], correct_output_data)

    def test_task_unpacker_choice_task(self):
        test_task = ChoiceTask(
            SUBSET_CHOICE_TABLE, 'alternatives', 'choice',
            SUBSET_CHOICE_FEATS_TABLE,
            {'ID': ['choice', 'alternatives']}
        )

        my_model = ClassificationReducer(
            DummyClassifier('constant', constant=1))

        unpacker_dict = my_model.task_unpacker(test_task)

        correct_output_data = pd.DataFrame([
            [1, 6],
            [1, 6],
            [1, 6],
            [0, 6],
            [0, 6],
            [0, 6],
            [1, 6],
            [0, 8],
            [0, 8],
            [0, 6],
            [0, 6],
            [0, 6],
            [0, 6],
            [0, 6],
            [0, 6],
            [0, 6],
            [0, 6]
        ], columns=['chosen', 'feat1'])

        assert_frame_equal(unpacker_dict['df_comb'].astype('int32'),
                           correct_output_data.astype('int32'))

        # [512709, 1, 6],
        # [529703, 1, 6],
        # [696056, 1, 6],
        # [490972, 0, 6],
        # [685450, 0, 6],
        # [5549502, 0, 6],
        # [723354, 1, 6],
        # [550707, 0, 8],
        # [551375, 0, 8],
        # [591842, 0, 6],
        # [601195, 0, 6],
        # [732624, 0, 6],
        # [778197, 0, 6],
        # [813892, 0, 6],
        # [817040, 0, 6],
        # [576214, 0, 6],
        # [673995, 0, 6]

    # def test_task_unpacker_diff_pairwise_no_result(self):
    #     test_task = PairwiseComparisonTask(
    #         DATA.drop('result', axis=1), ['ent1', 'ent2'], 'result', 'ent1',
    #         secondary_table=ENT1_ATTRIBUTES,
    #         secondary_to_primary_link={'ent1': ['ent1', 'ent2']})
    #
    #     my_model = ClassificationReducer(
    #         DummyClassifier('constant', constant=1))
    #primary_table_alternatives_names
    #     unpacker_dict = my_model.task_unpacker(test_task,
    #                                            take_feautre_diff=True)
    #
    #     correct_output_data = pd.DataFrame({
    #         'ent1': ['C', 'B', 'C', 'D'],
    #         'ent2': ['B', 'A', 'D', 'C'],
    #         'feat1_diff': [1, 10, -3, 3]
    #     })
    #
    #     print(unpacker_dict['df_comb'])
    #
    #     assert_frame_equal(unpacker_dict['df_comb'], correct_output_data)

    # def test_task_packer_pairwise_comparison(self):
    #     test_task = PairwiseComparisonTask(
    #         DATA, ['ent1', 'ent2'], 'result', 'ent1',
    #         secondary_table=ENT1_ATTRIBUTES,
    #         secondary_to_primary_link={'ent1': ['ent1', 'ent2']})
    #
    #     my_model = ClassificationReducer(
    #         DummyClassifier('constant', constant=1))
    #
    #     task_packer_results = my_model.task_packer(np.array([1, 1, 0, 1]),
    #                                                PairwiseComparisonTask)
    #
    #     expected_results = np.array([1, 1, 0, 1])
    #     assert_array_equal(expected_results, task_packer_results)

    # def test_task_packer_choice(self):
    #     test_task = ChoiceTask(
    #         SUBSET_CHOICE_TABLE, 'alternatives', 'choice',
    #         SUBSET_CHOICE_FEATS_TABLE,
    #         {'ID': ['choice', 'alternatives']}
    #     )
    #
    #     my_model = ClassificationReducer(
    #         DummyClassifier('constant', constant=1))
    #
    #     # Need to run unpacking before running packing
    #     _ = my_model.task_unpacker(test_task)
    #
    #     task_packer_results = my_model.task_packer(
    #         np.array([1, 0, 1, 0, 1, 0,
    #                   1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]),
    #         ChoiceTask)
    #
    #     expected_results = np.array(
    #         [np.array([512709, 696056, 685450], dtype=object),
    #          np.array([550707, 551375, 591842, 601195, 576214], dtype=object)])
    #
    #     assert_array_equal(expected_results, task_packer_results)
