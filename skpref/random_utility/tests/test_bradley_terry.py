import pandas as pd
import unittest
import numpy as np
import pytest
from pandas.util.testing import assert_frame_equal
from skpref.random_utility import (check_indexing_of_entities,
                                   get_distinct_entities,
                                   generate_entity_lookup,
                                   BradleyTerry)
from skpref.task import ChoiceTask

DATA = pd.DataFrame({'ent1': ['C', 'B', 'C', 'D'],
                     'ent2': ['B', 'A', 'D', 'C'],
                     'result': [1, 0, 0, 1]})

INDEXED_DATA = DATA.set_index(['ent1', 'ent2'])

INDEXED_DATA_NORESCOL = INDEXED_DATA.drop('result', axis=1)

TRANSITIVE_DATA = pd.DataFrame(
    {'ent1': ['A', 'A', 'D', 'D', 'B', 'D'],
     'ent2': ['B', 'C', 'A', 'B', 'C', 'C'],
     'result': [1, 1, 0, 0, 1, 0]}
)

TRANSITIVE_DATA_INDEXED = TRANSITIVE_DATA.set_index(['ent1', 'ent2'])

ENT1_ATTRIBUTES = pd.DataFrame(
    {'ent1': ['A', 'B', 'C', 'D'],
     'feat1': [1, 11, 12, 15]}
)
INDEXED_ENT1_ATTRIBUTES = ENT1_ATTRIBUTES.set_index(['ent1'])

ENT2_ATTRIBUTES = pd.DataFrame(
    {'ent2': ['A', 'B', 'C', 'D'],
     'feat1': [1, 1, 1, 0]}
)
INDEXED_ENT2_ATTRIBUTES = ENT2_ATTRIBUTES.set_index(['ent2'])

DATA_WITH_MERGECOL = pd.DataFrame(
    {'ent1': ['C', 'B', 'C', 'D'],
     'ent2': ['B', 'A', 'D', 'C'],
     'result': [1, 0, 0, 1],
     'mergecol': [1, 1, 1, 2]})

INDEXED_DATA_WITH_MERGECOL = DATA_WITH_MERGECOL.set_index(['ent1', 'ent2'])

ENT1_ATTRIBUTES_MERGECOL = pd.DataFrame(
    {'ent': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
     'feat1': [1, 11, 12, 15, 1, 2, 3, 4],
     'mergecol': [1, 1, 1, 1, 2, 2, 2, 2]}
)
INDEXED_ENT1_ATTRIBUTES_MERGECOL = ENT1_ATTRIBUTES_MERGECOL.set_index(['ent'])

ENT2_ATTRIBUTES_MERGECOL = pd.DataFrame(
    {'ent': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
     'feat1': [1, 1, 1, 0, 1, 1, 1, 1],
     'mergecol': [1, 1, 1, 1, 2, 2, 2, 2]}
)
INDEXED_ENT2_ATTRIBUTES_MERGECOL = ENT2_ATTRIBUTES_MERGECOL.set_index(['ent'])

EVERYONE_WINS_ONCE_DATA = pd.DataFrame(
    {'ent1': ['C', 'B', 'C', 'D', 'D'],
     'ent2': ['B', 'A', 'D', 'C', 'B'],
     'result': [1, 0, 0, 1, 0]})

INDEXED_EVERYONE_WINS_ONCE_DATA = EVERYONE_WINS_ONCE_DATA.set_index(['ent1',
                                                                     'ent2'])

EVERYONE_WINS_ONCE_DATA_MERGECOL = pd.DataFrame(
    {'ent1': ['C', 'B', 'C', 'D', 'D'],
     'ent2': ['B', 'A', 'D', 'C', 'B'],
     'result': [1, 0, 0, 1, 0],
     'mergecol': [1, 1, 1, 2, 2]})

EVERYONE_WINS_ONCE_DATA_MERGECOL_INDEXED = EVERYONE_WINS_ONCE_DATA_MERGECOL\
    .set_index(['ent1', 'ent2'])

# EVERYONE_WINS_ONCE_DATA_DIFFERENT_ENTITY_TYPES = pd.DataFrame(
#     {'ent1': ['C', 'B', 'B', 'D', 'D'],
#      'ent2': ['E', 'F', 'F', 'F', 'E'],
#      'result': [1, 0, 0, 1, 0]}
# )
# EVERYONE_WINS_ONCE_DATA_DIFFERENT_ENTITY_TYPES_INDECES = \
# EVERYONE_WINS_ONCE_DATA_DIFFERENT_ENTITY_TYPES.set_index(['ent1','ent2'])

SUBSET_CHOICE_TABLE = pd.DataFrame(
    {'choice': [[512709, 529703, 696056], [723354]],
     'alternatives': [[512709, 529703, 696056, 490972,  685450, 5549502],
                      [550707, 551375, 591842, 601195, 732624, 778197, 813892,
                       817040, 576214, 673995, 723354]]}
)

SUBSET_CHOICE_FEATS_TABLE = pd.DataFrame(
    {'ID': [490972,  512709,  529703,  550707,  551375,  576214,  591842,
            601195,  673995,  685450,  696056,  723354,  732624,  778197,
            813892,  817040, 5549502],
     'feat1': [6, 6, 6, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
     }
)

class TestPairwiseModelFunctions(unittest.TestCase):

    def test_check_indexing_of_entities(self):
        data = DATA.set_index(['ent1', 'ent2', 'result'])
        with self.assertRaises(Exception):
            check_indexing_of_entities(data)

        try:
            check_indexing_of_entities(INDEXED_DATA)
        except:
            self.fail("check_indexing_of_entities failed unexpectedly")

        data3 = DATA.set_index(['ent1'])
        with self.assertRaises(Exception):
            check_indexing_of_entities(data3)

    def test_get_distinct_entities(self):
        data = DATA.set_index(['ent1', 'ent2'])

        correct_unique_index = ['A', 'B', 'C', 'D']

        unique_indices = get_distinct_entities(data)
        self.assertListEqual(unique_indices, correct_unique_index)

    def test_generate_entity_lookup(self):
        rplc, find = generate_entity_lookup(['A', 'B', 'C', 'D'])
        correct_rplc = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        correct_find = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        self.assertEqual(rplc.keys(), correct_rplc.keys())
        for key in rplc:
            self.assertEqual(rplc[key], correct_rplc[key])

        self.assertEqual(find.keys(), correct_find.keys())
        for key in find:
            self.assertEqual(find[key], correct_find[key])


class TestBradleyTerryFunctions(unittest.TestCase):

    def test_replace_entities_with_lkp(self):
        bt = BradleyTerry()
        lookup = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        replaced_data = bt.replace_entities_with_lkp(INDEXED_DATA, lookup)
        correct_replaced_data = pd.DataFrame(
            {'ent1': [3, 2, 3, 4],
             'ent2': [2, 1, 4, 3],
             'result': [1, 0, 0, 1]}
        )

        assert_frame_equal(replaced_data, correct_replaced_data)

    def test_unpack_data_for_choix(self):
        bt = BradleyTerry()
        bt.rplc_lkp, bt.lkp = generate_entity_lookup(
            get_distinct_entities(INDEXED_DATA))
        bt.target_col_name = 'result'
        data, n_ents = bt.unpack_data_for_choix(
            INDEXED_DATA, INDEXED_DATA.index.names)
        correct_data = {'winner': [(2, 1), (0, 1), (3, 2), (3, 2)]}
        corrrect_n_ents = 4
        self.assertEqual(n_ents, corrrect_n_ents)
        self.assertEqual(data.keys(), correct_data.keys())
        for key in data:
            for iteration, _tuple in enumerate(data[key]):
                self.assertTupleEqual(_tuple, correct_data[key][iteration])

    def test_rank_entities(self):
        bt = BradleyTerry()
        bt.fit(TRANSITIVE_DATA_INDEXED, 'result')
        asc_rank = bt.rank_entities()
        correct_asc_rank = ['D', 'C', 'B', 'A']
        self.assertListEqual(asc_rank, correct_asc_rank)

        desc_rank = bt.rank_entities(ascending=False)
        correct_desc_rank = ['A', 'B', 'C', 'D']
        self.assertListEqual(desc_rank, correct_desc_rank)

    def test_check_for_no_new_entities(self):
        bt = BradleyTerry()
        bt.is_fitted = True
        bt.target_col_name = 'result'
        bt.lkp = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        wrong_data = pd.DataFrame({
            'ent1': ['K', 'A'],
            'ent2': ['A,', 'B'],
            'result': [1, 0]
        })
        wrong_data = wrong_data.set_index(['ent1', 'ent2'])
        with self.assertRaises(Exception):
            bt.check_for_no_new_entities(wrong_data)

        try:
            bt.check_for_no_new_entities(TRANSITIVE_DATA_INDEXED)
        except:
            self.fail("check_for_no_new_entities failed unexpectedly")

    # This functionality has been commented out so that it doesn't throw too
    # many warnings during GridSearch, so the unit test for it is commented out

    # def test_check_if_target_column_passed_in_pred(self):
    #     bt = BradleyTerry()
    #     bt.fit(TRANSITIVE_DATA_INDEXED, 'result')
    #     with self.assertWarns(Warning):
    #         bt.check_if_target_column_passed_in_pred(TRANSITIVE_DATA_INDEXED)
    #
    #     with pytest.warns(None) as record:
    #         bt.check_if_target_column_passed_in_pred(INDEXED_DATA_NORESCOL)
    #
    #     assert len(record) == 0

    def test_find_strength_diff(self):
        bt = BradleyTerry()
        bt.is_fitted = True
        bt.target_col_name = 'result'
        bt.lkp = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        bt.rplc_lkp = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        bt._params = np.array([0.3, 0.2, -0.2, -0.4])
        bt.params_ = np.array([0.3, 0.2, -0.2, -0.4])

        calc_strength = bt.find_strength_diff(INDEXED_DATA_NORESCOL)
        correct_strength = np.array([-0.4, -0.1, 0.2, -0.2])
        np.testing.assert_array_almost_equal(calc_strength, correct_strength,
                                             decimal=10)

    def test_predict_proba(self):
        bt = BradleyTerry()
        bt.is_fitted = True
        bt.target_col_name = 'result'
        bt.lkp = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        bt.rplc_lkp = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        bt._params = np.array([0.3, 0.2, -0.2, -0.4])
        bt.params_ = np.array([0.3, 0.2, -0.2, -0.4])
        bt.pylogit_fit = False

        def exp_func(x):
            return 1/(1+np.exp(-x))

        pred_probs = bt.predict_proba(INDEXED_DATA_NORESCOL)
        corect_probs = np.array([exp_func(-0.4), exp_func(-0.1), exp_func(0.2),
                                 exp_func(-0.2)])

        np.testing.assert_array_equal(pred_probs, corect_probs)

    def test_predict_choice(self):
        bt = BradleyTerry()
        bt.fit(TRANSITIVE_DATA_INDEXED, 'result')
        choice = bt.predict_choice(INDEXED_DATA_NORESCOL)
        correct_choice = np.array(['B', 'A', 'C', 'C'])
        np.testing.assert_array_equal(choice, correct_choice)

    def test_predict(self):
        bt = BradleyTerry()
        bt.fit(TRANSITIVE_DATA_INDEXED, 'result')
        pred = bt.predict(INDEXED_DATA_NORESCOL)
        correct_pred = np.array([0, 0, 1, 0])
        np.testing.assert_array_equal(pred, correct_pred)

    def test_unpack_data_for_pylogit(self):
        bt = BradleyTerry()

        bt.rplc_lkp, bt.lkp = generate_entity_lookup(
            get_distinct_entities(INDEXED_EVERYONE_WINS_ONCE_DATA))

        bt.target_col_name = 'result'

        long_format_pylogit = bt.unpack_data_for_pylogit(
            INDEXED_EVERYONE_WINS_ONCE_DATA, ['ent1', 'ent2'])

        correct_lf_output = pd.DataFrame(
            {'observation': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
             'entity': [1, 2, 0, 1, 2, 3, 2, 3, 1, 3],
             'CHOICE': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]}
        )

        assert_frame_equal(long_format_pylogit.astype('int32'),
                           correct_lf_output.astype('int32'))

    def test_join_up_dataframes(self):
        bt = BradleyTerry()

        # Testing case when only df_i is fed in but df_j is meant to be the same
        bt.rplc_lkp, bt.lkp = generate_entity_lookup(
            get_distinct_entities(INDEXED_EVERYONE_WINS_ONCE_DATA))

        bt.target_col_name = 'result'

        long_format_pylogit = bt.unpack_data_for_pylogit(
            INDEXED_EVERYONE_WINS_ONCE_DATA, ['ent1', 'ent2'])

        x_comb = bt.join_up_dataframes(long_format_pylogit,
                                       df_i=INDEXED_ENT1_ATTRIBUTES)

        correct_x_comb = pd.DataFrame(
            {'observation': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
             'entity': [1, 2, 0, 1, 2, 3, 2, 3, 1, 3],
             'CHOICE': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
             'feat1': [11, 12, 1, 11, 12, 15, 12, 15, 11, 15]}
        )

        assert_frame_equal(x_comb.astype('int32'), correct_x_comb.astype('int32')
                           )

        # Testing the case when different df_i and df_j is fed in
        # x_comb_1_2 = \
        #     bt.join_up_dataframes(INDEXED_EVERYONE_WINS_ONCE_DATA,
        #                           INDEXED_ENT1_ATTRIBUTES,
        #                           INDEXED_ENT2_ATTRIBUTES)
        #
        # correct_x_comb_1_2 = pd.DataFrame(
        #     {'observation': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        #      'entity': [1, 2, 0, 1, 2, 3, 2, 3, 1, 3],
        #      'CHOICE': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
        #      'feat1': [11, 12, 1, 11, 12, 15, 12, 15, 11, 15],
        #      'feat2'}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_1_2, correct_x_comb_1_2)
        #
        # # Testing the case when df_j is fed in but df_i is meant to be the same
        # x_comb_2_1, x_comb_entnames_2_1, run_choix_2_1 = \
        #     bt.join_up_dataframes(INDEXED_DATA, df_j=INDEXED_ENT2_ATTRIBUTES)
        #
        # correct_x_comb_2_1 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'feat1_ent2': [1, 1, 0, 1],
        #      'feat1_ent1': [1, 1, 1, 0]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_2_1, correct_x_comb_2_1)
        # self.assertListEqual(x_comb_entnames_2_1, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_2_1)
        #
        # # Case when df_j is fed in but df_i is not meant to exist
        # x_comb_2, x_comb_entnames_2, run_choix_2 = \
        #     bt.join_up_dataframes(INDEXED_DATA, df_j=INDEXED_ENT2_ATTRIBUTES,
        #                           same_ent_data=False)
        #
        # correct_x_comb_2 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'feat1': [1, 1, 0, 1]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_2, correct_x_comb_2)
        # self.assertListEqual(x_comb_entnames_2, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_2)
        #
        # # Case when df_i is fed in but df_j does not exist
        # x_comb_1, x_comb_entnames_1, run_choix_1 = \
        #     bt.join_up_dataframes(INDEXED_DATA, df_i=INDEXED_ENT1_ATTRIBUTES,
        #                           same_ent_data=False)
        #
        # correct_x_comb_1 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'feat1': [12, 11, 12, 15]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_1, correct_x_comb_1)
        # self.assertListEqual(x_comb_entnames_1, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_1)
        #
        # # Case when simple Bradley Terry is fed in
        # x_comb_na, x_comb_entnames_na, run_choix_na = \
        #     bt.join_up_dataframes(INDEXED_DATA)
        #
        # assert_frame_equal(x_comb_na, INDEXED_DATA)
        # self.assertListEqual(x_comb_entnames_na, ['ent1', 'ent2'])
        # self.assertTrue(run_choix_na)
        #
        # Case when df_i is fed in with a merge column and df_j is meant to be
        # the same
        long_format_pylogit = bt.unpack_data_for_pylogit(
            EVERYONE_WINS_ONCE_DATA_MERGECOL_INDEXED, ['ent1', 'ent2'])

        x_comb_m = bt.join_up_dataframes(long_format_pylogit,
                                         df_i=INDEXED_ENT1_ATTRIBUTES_MERGECOL,
                                         merge_columns=['mergecol'])

        correct_x_comb_m = pd.DataFrame(
            {'observation': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
             'entity': [1, 2, 0, 1, 2, 3, 2, 3, 1, 3],
             'CHOICE': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
             'feat1': [11, 12, 1, 11, 12, 15, 3, 4, 2, 4]}
        )

        assert_frame_equal(x_comb_m.astype('int32'),
                           correct_x_comb_m.astype('int32'))
        #
        # # Case when df_j is fed in with a merge column and df_i is meant to be
        # # the same
        # x_comb_m_2_1, x_comb_entnames_m_2_1, run_choix_m_2_1 = \
        #     bt.join_up_dataframes(INDEXED_DATA_WITH_MERGECOL,
        #                           df_j=INDEXED_ENT2_ATTRIBUTES_MERGECOL,
        #                           merge_columns=['mergecol'])
        #
        # correct_x_comb_m_2_1 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'mergecol': [1, 1, 1, 2],
        #      'feat1_ent2': [1, 1, 0, 1],
        #      'feat1_ent1': [1, 1, 1, 1]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_m_2_1, correct_x_comb_m_2_1)
        # self.assertListEqual(x_comb_entnames_m_2_1, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_m_2_1)
        #
        # # Case when df_j is fed in with a merge column and df_i is meant to be
        # # empty
        # x_comb_m_2, x_comb_entnames_m_2, run_choix_m_2 = \
        #     bt.join_up_dataframes(INDEXED_DATA_WITH_MERGECOL,
        #                           df_j=INDEXED_ENT2_ATTRIBUTES_MERGECOL,
        #                           merge_columns=['mergecol'],
        #                           same_ent_data=False)
        #
        # correct_x_comb_m_2 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'mergecol': [1, 1, 1, 2],
        #      'feat1': [1, 1, 0, 1]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_m_2, correct_x_comb_m_2)
        # self.assertListEqual(x_comb_entnames_m_2, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_m_2)
        #
        # # Case when df_i is fed in with a merge column and df_j is meant to be
        # # empty
        # x_comb_m_1, x_comb_entnames_m_1, run_choix_m_1 = \
        #     bt.join_up_dataframes(INDEXED_DATA_WITH_MERGECOL,
        #                           INDEXED_ENT1_ATTRIBUTES_MERGECOL,
        #                           merge_columns=['mergecol'],
        #                           same_ent_data=False)
        #
        # correct_x_comb_m_1 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'mergecol': [1, 1, 1, 2],
        #      'feat1': [12, 11, 12, 4]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_m_1, correct_x_comb_m_1)
        # self.assertListEqual(x_comb_entnames_m_1, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_m_1)
        #
        # # Case when df_i and df_j is fed in with a merge column
        # x_comb_m_1_2, x_comb_entnames_m_1_2, run_choix_m_1_2 = \
        #     bt.join_up_dataframes(INDEXED_DATA_WITH_MERGECOL,
        #                           INDEXED_ENT1_ATTRIBUTES_MERGECOL,
        #                           INDEXED_ENT2_ATTRIBUTES_MERGECOL,
        #                           merge_columns=['mergecol'],
        #                           same_ent_data=False)
        #
        # correct_x_comb_m_1_2 = pd.DataFrame(
        #     {'ent1': ['C', 'B', 'C', 'D'],
        #      'ent2': ['B', 'A', 'D', 'C'],
        #      'result': [1, 0, 0, 1],
        #      'mergecol': [1, 1, 1, 2],
        #      'feat1_ent1': [12, 11, 12, 4],
        #      'feat1_ent2': [1, 1, 0, 1]}
        # ).set_index(['ent1', 'ent2'])
        #
        # assert_frame_equal(x_comb_m_1_2, correct_x_comb_m_1_2)
        # self.assertListEqual(x_comb_entnames_m_1_2, ['ent1', 'ent2'])
        # self.assertFalse(run_choix_m_1_2)

    def test_task_indexing(self):
        test_task = ChoiceTask(SUBSET_CHOICE_TABLE, 'alternatives', 'choice',
                               secondary_table=SUBSET_CHOICE_FEATS_TABLE,
                               secondary_to_primary_link={
                                   'ID': ['choice', 'alternatives']})

        mybt = BradleyTerry(method='BFGS', alpha=1e-5)

        d1, d2 = mybt.task_indexing(test_task)

        correct_d1 = pd.DataFrame(
            [
                [0, 512709, 490972, 1],
                [0, 512709, 685450, 1],
                [0, 512709, 5549502, 1],
                [0, 529703, 490972, 1],
                [0, 529703, 685450, 1],
                [0, 529703, 5549502, 1],
                [0, 696056, 490972, 1],
                [0, 696056, 685450, 1],
                [0, 696056, 5549502, 1],
                [1, 723354, 550707, 1],
                [1, 723354, 551375, 1],
                [1, 723354, 591842, 1],
                [1, 723354, 601195, 1],
                [1, 723354, 732624, 1],
                [1, 723354, 778197, 1],
                [1, 723354, 813892, 1],
                [1, 723354, 817040, 1],
                [1, 723354, 576214, 1],
                [1, 723354, 673995, 1]
            ], columns=['observation', 'top', 'boot', 'choice']
        ).set_index(['top', 'boot'])

        correct_d2 = SUBSET_CHOICE_FEATS_TABLE.set_index('ID')
        assert_frame_equal(d1.astype('int32'), correct_d1.astype('int32'))
        assert_frame_equal(d2.astype('int32'), correct_d2.astype('int32'))
