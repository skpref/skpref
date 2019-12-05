import unittest
from skpref.task import SubsetPosetVec
import numpy as np

top_choices = np.array([np.array(['a', 'b', 'c']), np.array(['a', 'b', 'd']),
                        np.array(['d', 'b', 'c'])])
boots = np.array([np.array(['d']), np.array(['c']), np.array(['a'])])


class TestSubsetPosetVec(unittest.TestCase):
    test_spv = SubsetPosetVec(top_choices, boots, subset_type_vars={
        'top_size_const': True, 'top_size': 3, 'boot_size_const': True,
        'boot_size': 1
    })

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
