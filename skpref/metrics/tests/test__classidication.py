import unittest
from skpref.metrics import (true_positives, true_negatives, false_positives,
                            false_negatives, accuracy, precision, recall,
                            f1_score, log_loss)
from skpref.data_processing import SubsetPosetVec
import numpy as np
from sklearn.metrics import log_loss as skll

dummy_gt = SubsetPosetVec(
            top_input_data=np.array(['a', 'b', 'c', 'd']),
            boot_input_data=np.array([
                np.array(['b', 'c', 'd']),
                np.array(['c']),
                np.array(['a', 'd']),
                np.array(['a', 'c'])
            ], dtype=object)
        )

dummy_predictions = SubsetPosetVec(
    top_input_data=np.array(['b', 'c', 'c', 'd']),
    boot_input_data=np.array([
        np.array(['a', 'c', 'd']),
        np.array(['b']),
        np.array(['a', 'd']),
        np.array(['a', 'c'])
    ], dtype=object)
)


class TestClassificationMetrics(unittest.TestCase):
    def test_true_positives(self):
        calculated = true_positives(dummy_gt, dummy_predictions)
        expected = 2
        self.assertEqual(calculated, expected)

    def test_true_negatives(self):
        calculated = true_negatives(dummy_gt, dummy_predictions)
        expected = 6
        self.assertEqual(calculated, expected)

    def test_false_positives(self):
        calculated = false_positives(dummy_gt, dummy_predictions)
        expected = 2
        self.assertEqual(calculated, expected)

    def test_false_negatives(self):
        calculated = false_negatives(dummy_gt, dummy_predictions)
        expected = 2
        self.assertEqual(calculated, expected)

    def test_accuracy(self):
        calculated = accuracy(dummy_gt, dummy_predictions)
        expected = 8 / 12
        self.assertEqual(calculated, expected)

    def test_precision(self):
        calculated = precision(dummy_gt, dummy_predictions)
        expected = 2 / (2 + 2)
        self.assertEqual(calculated, expected)

    def test_recall(self):
        calculated = recall(dummy_gt, dummy_predictions)
        expected = 2 / (2 + 2)
        self.assertEqual(calculated, expected)

    def test_f1_score(self):
        calculated = f1_score(dummy_gt, dummy_predictions)
        expected = (
            (2 * (2 / (2 + 2)) * (2 / (2 + 2)))
        )
        self.assertEqual(calculated, expected)

    def test_log_loss(self):
        preds = {
            'a': [0.33, 0.5, 0.1],
            'b': [0.33, 0.5, 0.1],
            'c': [0.33, 0, 0.8]
        }

        true = SubsetPosetVec(
            top_input_data=np.array(['a', 'b', 'c']),
            boot_input_data=np.array([
                np.array(['b', 'c']),
                np.array(['b']),
                np.array(['a', 'b'])
            ], dtype=object))

        calculated = log_loss(true, preds)
        expected = {
            'a_log_loss': -(np.log(0.33) + np.log(0.5) + np.log(0.9))/3,
            'b_log_loss': -(np.log(1-0.33) + np.log(0.5) + np.log(0.9))/3,
            'c_log_loss': -(np.log(1-0.33) + np.log(1) + + np.log(0.8))/3
        }

        for key in list(expected.keys()):
            self.assertAlmostEquals(expected[key], calculated[key])
