from skpref.data_processing import SubsetPosetVec
from skpref.task import PrefTask
import numpy as np
import pandas as pd


class UnderDevError(Exception):
    """
    Error that we give when something is being tried that is planned to be
    developed in the package, but isn't yet done
    """

    def __init__(self, message):
        self.message = message


def aggregate_full_probability_predictions_to_discrete_choice(
        predictions: dict, task: PrefTask) -> SubsetPosetVec:
    """
    Method that aggregates probabilistic predictions into discrete choice,
    shared by ClassificationReducer and PairwiseComparisonModel. For example if
    the predictions are {'a': [0.6,0.1], 'b': [0.3,0.1], 'c': [0.1, 0.8]} then
    the output will be a SubsetPosetVector that indicates that for decision 1 'a'
    was chosen and for decision 2 'c' was chosen.

    Parameters
    ----------
    predictions (dict): The probabilistic prediction output in the standard
        format that we use in skpref
    task (PrefTask): The task that is being used for prediction

    Returns
    -------
    SubsetPosetVec of predcited discrete choice outcomes
    """

    df_preds = pd.DataFrame(predictions)
    top = list(df_preds.idxmax(axis=1).values)
    alts = task.primary_table[task.primary_table_alternatives_names].copy().values
    boot = [np.setdiff1d(alts[i], top[i]) for i in range(0, len(top))]

    return SubsetPosetVec(top_input_data=np.array(top, dtype=object),
                          boot_input_data=np.array(boot, dtype=object))


def nice_print_results(results: dict, dp: int = 2) -> None:
    """
    Prints a dictionary where the values are numpy arrays in a nicer format

    Parameters
    ----------
    results (dict): A dictionary where the values are numpy arrays
    dp (int): Decimal places to round np array to
    """

    padding = np.max([len (keystring) for keystring in results.keys()]) + 1
    for i in results.keys():
        print(f'{i: <{padding}}', np.around(results[i], dp))
