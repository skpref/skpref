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

    return SubsetPosetVec(top_input_data=np.array(top),
                          boot_input_data=np.array(boot))
