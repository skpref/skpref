from skpref.data_processing import SubsetPosetVec
import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss
from scipy.stats import ttest_ind


def true_positives(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> int:
    """
    The number of alternatives predicted to be chosen and were actually chosen
    (tp)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """
    tp = 0
    for i in range(len(actuals.top_input_data)):
        tp += len(
            np.intersect1d(actuals.top_input_data[i],
                           predicted.top_input_data[i]))

    return tp


def true_negatives(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> int:
    """
    The number of alternatives predicted to be bot chosen and were actually not
    chosen (tn)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values

    """

    tn = 0
    for i in range(len(actuals.boot_input_data)):
        tn += len(
            np.intersect1d(actuals.boot_input_data[i],
                           predicted.boot_input_data[i]))

    return tn


def false_positives(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> int:
    """
    The number of alternatives predicted to be chosen and were actually not
    chosen (fp)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """
    fp = 0
    for i in range(len(actuals.top_input_data)):
        fp += len(
            np.intersect1d(actuals.boot_input_data[i],
                           predicted.top_input_data[i]))

    return fp


def false_negatives(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> int:
    """
    The number of alternatives predicted to be not chosen and were actually
    chosen (fn)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """

    fn = 0
    for i in range(len(actuals.boot_input_data)):
        fn += len(
            np.intersect1d(actuals.top_input_data[i],
                           predicted.boot_input_data[i]))

    return fn


def accuracy(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> float:
    """
    The percentage of alternatives correctly identified:
    (tp + tn) / (tp+fp+tn+fn)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """

    tp = true_positives(actuals, predicted)
    tn = true_negatives(actuals, predicted)
    fp = false_positives(actuals, predicted)
    fn = false_negatives(actuals, predicted)

    return (
        (tp + tn) / (tp + tn + fp + fn)
    )


def precision(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> float:
    """
    Percentage of the alternatives predicted to be chosen and were actually
    chosen
    tp/(tp+fp)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """

    tp = true_positives(actuals, predicted)
    fp = false_positives(actuals, predicted)

    return tp /(tp + fp)


def recall(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> float:
    """
    The percentage of chosen alternatives identified correctly by the predictions
    tp / (tp + fn)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """

    tp = true_positives(actuals, predicted)
    fn = false_negatives(actuals, predicted)

    return tp / (tp + fn)


def f1_score(actuals: SubsetPosetVec, predicted: SubsetPosetVec) -> float:
    """
    The harmonic mean between precision and recall
    (2 x precision x recall) / (precision + recall)
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """

    rc = recall(actuals, predicted)
    pc = precision(actuals, predicted)

    return (
        (2 * pc * rc) / (pc + rc)
    )


def log_loss(actuals: SubsetPosetVec, predicted: dict, **kwargs) -> dict:
    """
    Calculates the mean log loss for each alternative, can provide arguments
    as in scikit-learn log loss
    Parameters
    ----------
    actuals: the true values
    predicted: predicted values
    """
    ll = {}
    for _alternative in list(predicted.keys()):
        binarized_outcome = np.where(
            actuals.top_input_data == _alternative, 1 , 0)
        ll[str(_alternative) + '_log_loss'] = sklearn_log_loss(
            binarized_outcome, predicted[_alternative], **kwargs
        )

    return ll


def log_loss_compare_with_t_test(actuals: SubsetPosetVec, predicted1: dict,
                                 predicted2: dict) -> dict:
    """
    For two different probabilistic predictions calculates whether they are
    significantly different
    Parameters
    ----------
    actuals: the true values
    predicted1: predicted values 1
    predicted2: predicted values 2
    """
    ll_t = {}
    for _alternative in list(predicted1.keys()):
        binarized_outcome = np.where(
            actuals.top_input_data.flatten() == _alternative, 1, 0)
        inv_binarized_outcome = np.where(
            actuals.top_input_data.flatten() == _alternative, 0, 1)
        logged = np.log(predicted1[_alternative])
        logged_oneminus = np.log(np.ones(len(predicted1[_alternative])) -
                                 predicted1[_alternative])
        final_losses1 = (
                np.nan_to_num(logged * binarized_outcome) +
                np.nan_to_num(logged_oneminus * inv_binarized_outcome))

        logged2 = np.log(predicted2[_alternative])
        logged_oneminus2 = np.log(np.ones(len(predicted2[_alternative])) -
                                 predicted2[_alternative])
        final_losses2 = (
                np.nan_to_num(logged2 * binarized_outcome) +
                np.nan_to_num(logged_oneminus2 * inv_binarized_outcome))

        ll_t[_alternative] = ttest_ind(final_losses1, final_losses2)

    return ll_t
