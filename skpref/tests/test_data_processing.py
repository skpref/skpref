import unittest
from skpref.data_processing import SubsetPosetVec
import numpy as np
import pandas as pd

top_choices = np.array([np.array(['a', 'b', 'c']), np.array(['a', 'b', 'd']),
                        np.array(['d', 'b', 'c'])])
boots = np.array([np.array(['d']), np.array(['c']), np.array(['a'])])

top_ch_int = np.array([np.array([512709, 529703, 696056]),
                      np.array([723354])])
boot_ch_int = np.array([
    np.array([490972,  685450, 5549502]),
    np.array([550707, 551375, 591842, 601195, 732624, 778197, 813892, 817040,
              576214, 673995])])


class TestSubsetPosetVec(unittest.TestCase):
    test_spv = SubsetPosetVec(top_choices, boots, subset_type_vars={
        'top_size_const': True, 'top_size': 3, 'boot_size_const': True,
        'boot_size': 1
    })

    test_spv_int = SubsetPosetVec(top_ch_int, boot_ch_int)

    def test_entity_universe(self):
        np.testing.assert_array_equal(self.test_spv.entity_universe,
                                      np.array(['a', 'b', 'c', 'd']))

    def test_efficient_representation(self):
        correct_top_array = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 1]
        ])
        
        correct_boot_array = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ])

        np.testing.assert_array_equal(
            self.test_spv.efficient_representation[0].A,
            correct_top_array)

        np.testing.assert_array_equal(
            self.test_spv.efficient_representation[1].A,
            correct_boot_array)

        self.assertEqual(self.test_spv.efficient_representation[0].shape,
                         (3, 4))

        self.assertEqual(self.test_spv.efficient_representation[1].shape,
                         (3, 4))

        self.assertEqual(self.test_spv.efficient_representation[0].nnz, 9)

        self.assertEqual(self.test_spv.efficient_representation[1].nnz, 3)

    def test_pairwise_reducer(self):

        correct_positive_table = pd.DataFrame(
            [
                [0, 512709, 490972],
                [0, 512709, 685450],
                [0, 512709, 5549502],
                [0, 529703, 490972],
                [0, 529703, 685450],
                [0, 529703, 5549502],
                [0, 696056, 490972],
                [0, 696056, 685450],
                [0, 696056, 5549502],
                [1, 723354, 550707],
                [1, 723354, 551375],
                [1, 723354, 591842],
                [1, 723354, 601195],
                [1, 723354, 732624],
                [1, 723354, 778197],
                [1, 723354, 813892],
                [1, 723354, 817040],
                [1, 723354, 576214],
                [1, 723354, 673995]
            ], columns=['observation', 'top', 'boot']
        )

        output_red = self.test_spv_int.pairwise_reducer(scramble=False)
        pd.testing.assert_frame_equal(
            output_red.astype('int32'),
            correct_positive_table.astype('int32'))

        correct_reciprocal_table = pd.DataFrame(
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
        )

        output_red_rec = self.test_spv_int.pairwise_reducer(style="reciprocal")
        pd.testing.assert_frame_equal(
            output_red_rec.astype('int32'),
            correct_reciprocal_table.astype('int32'))

        correct_reciprocal_table_neg_fail = pd.DataFrame(
            [
                [0, 512709, 490972, 1],
                [0, 490972, 512709, -1],
                [0, 512709, 685450, 1],
                [0, 685450, 512709, -1],
                [0, 512709, 5549502, 1],
                [0, 5549502, 512709, -1],
                [0, 529703, 490972, 1],
                [0, 490972, 529703, -1],
                [0, 529703, 685450, 1],
                [0, 685450, 529703, -1],
                [0, 529703, 5549502, 1],
                [0, 5549502, 529703, -1],
                [0, 696056, 490972, 1],
                [0, 490972, 696056, -1],
                [0, 696056, 685450, 1],
                [0, 685450, 696056, -1],
                [0, 696056, 5549502, 1],
                [0, 5549502, 696056, -1],
                [1, 723354, 550707, 1],
                [1, 550707, 723354, -1],
                [1, 723354, 551375, 1],
                [1, 551375, 723354, -1],
                [1, 723354, 591842, 1],
                [1, 591842, 723354, -1],
                [1, 723354, 601195, 1],
                [1, 601195, 723354, -1],
                [1, 723354, 732624, 1],
                [1, 732624, 723354, -1],
                [1, 723354, 778197, 1],
                [1, 778197, 723354, -1],
                [1, 723354, 813892, 1],
                [1, 813892, 723354, -1],
                [1, 723354, 817040, 1],
                [1, 817040, 723354, -1],
                [1, 723354, 576214, 1],
                [1, 576214, 723354, -1],
                [1, 723354, 673995, 1],
                [1, 673995, 723354, -1]
            ], columns=['observation', 'alt1', 'alt2', 'alt1_top']
        )

        output_red_rec_neg = self.test_spv_int.pairwise_reducer(
            style="reciprocal", rejection=-1)

        pd.testing.assert_frame_equal(
            output_red_rec_neg.astype('int32'),
            correct_reciprocal_table_neg_fail.astype('int32'))

        correct_string_pairwise_red = pd.DataFrame([
            [0, 'a', 'd'],
            [0, 'b', 'd'],
            [0, 'c', 'd'],
            [1, 'a', 'c'],
            [1, 'b', 'c'],
            [1, 'd', 'c'],
            [2, 'd', 'a'],
            [2, 'b', 'a'],
            [2, 'c', 'a']
        ], columns=['observation', 'top', 'boot'])

        output_red_str = self.test_spv.pairwise_reducer(scramble=False)

        pd.testing.assert_frame_equal(
            correct_string_pairwise_red.astype({'observation': 'int32'}),
            output_red_str.astype({'observation': 'int32'}))

        correct_string_pairwise_red_rec = pd.DataFrame([
            [0, 'a', 'd', 1],
            [0, 'd', 'a', 0],
            [0, 'b', 'd', 1],
            [0, 'd', 'b', 0],
            [0, 'c', 'd', 1],
            [0, 'd', 'c', 0],
            [1, 'a', 'c', 1],
            [1, 'c', 'a', 0],
            [1, 'b', 'c', 1],
            [1, 'c', 'b', 0],
            [1, 'd', 'c', 1],
            [1, 'c', 'd', 0],
            [2, 'd', 'a', 1],
            [2, 'a', 'd', 0],
            [2, 'b', 'a', 1],
            [2, 'a', 'b', 0],
            [2, 'c', 'a', 1],
            [2, 'a', 'c', 0]
        ], columns=['observation', 'alt1', 'alt2', 'alt1_top'])

        output_red_str_rec = self.test_spv.pairwise_reducer(style='reciprocal')

        pd.testing.assert_frame_equal(
            correct_string_pairwise_red_rec.astype({'observation': 'int32',
                                                    'alt1_top': 'int32'}),
            output_red_str_rec.astype({'observation': 'int32',
                                       'alt1_top': 'int32'}))

        correct_pairwise_scrambled = pd.DataFrame(
            [[0, 512709, 490972, 1],
             [0, 512709, 685450, 1],
             [0, 5549502, 512709, 0],
             [0, 529703, 490972, 1],
             [0, 529703, 685450, 1],
             [0, 529703, 5549502, 1],
             [0, 696056, 490972, 1],
             [0, 696056, 685450, 1],
             [0, 5549502, 696056, 0],
             [1, 723354, 550707, 1],
             [1, 723354, 551375, 1],
             [1, 723354, 591842, 1],
             [1, 723354, 601195, 1],
             [1, 723354, 732624, 1],
             [1, 723354, 778197, 1],
             [1, 723354, 813892, 1],
             [1, 817040, 723354, 0],
             [1, 723354, 576214, 1],
             [1, 673995, 723354, 0]],
            columns=['observation', 'alt1', 'alt2', 'alt1_top']
        )
        result_pairwise_scrambled = self.test_spv_int.pairwise_reducer(
            random_seed_scramble=42)

        pd.testing.assert_frame_equal(
            correct_pairwise_scrambled.astype('int32'),
            result_pairwise_scrambled.astype('int32'))

    def test_pairwise_reducer_for_scoring(self):
        # Create a subset poset vector with no depvar
        alts = np.array([np.array([111, 222]), np.array([111, 222, 333])])
        test_scoring_vec = SubsetPosetVec(alts, alts)

        pairwise_reduction = test_scoring_vec.pairwise_reducer(scramble=False)

        expected_outcome = pd.DataFrame({
            'observation': [0, 0, 1, 1, 1, 1, 1, 1],
            'top': [111, 222, 111, 111, 222, 222, 333, 333],
            'boot': [222, 111, 222, 333, 111, 333, 111, 222]
        })

        pd.testing.assert_frame_equal(pairwise_reduction.astype('int32'),
                                      expected_outcome.astype('int32'))

    def test_if_name_error_raised(self):
        with self.assertRaises(NameError):
            self.test_spv.pairwise_reducer(style='somethingelse')

    def test_classifier_reducer(self):
        correct_int_classifier_output = pd.DataFrame([
            [0, 512709, 1],
            [0, 529703, 1],
            [0, 696056, 1],
            [0, 490972, 0],
            [0, 685450, 0],
            [0, 5549502, 0],
            [1, 723354, 1],
            [1, 550707, 0],
            [1, 551375, 0],
            [1, 591842, 0],
            [1, 601195, 0],
            [1, 732624, 0],
            [1, 778197, 0],
            [1, 813892, 0],
            [1, 817040, 0],
            [1, 576214, 0],
            [1, 673995, 0]
        ], columns=['observation', 'alternative', 'chosen'])

        result_int_class_red = self.test_spv_int.classifier_reducer()

        pd.testing.assert_frame_equal(
            correct_int_classifier_output.astype('int32'),
            result_int_class_red.astype('int32'))

    def test_classifier_reducer_for_scoring(self):
        # Create a subset poset vector with no depvar
        alts = np.array([np.array([111, 222]), np.array([111, 222, 333])])
        test_scoring_vec = SubsetPosetVec(alts, alts)

        pairwise_reduction = test_scoring_vec.classifier_reducer()

        expected_outcome = pd.DataFrame({
            'observation': [0, 0, 1, 1, 1],
            'alternative': [111, 222, 111, 222, 333],
            'chosen': [1, 1, 1, 1, 1]
        })

        pd.testing.assert_frame_equal(pairwise_reduction.astype('int32'),
                                      expected_outcome.astype('int32'))
