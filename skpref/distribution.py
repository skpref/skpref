from typing import Union, Callable, List
from skpref.data_processing import PosetVector
import pandas as pd
import numpy as np


class DiscreteDistribution:
    """
    DiscreteDistribution is designed to be the return object of predict_proba
    functions. When looking at subset selection and ranking the number of
    possible permutations become very large, so it's best to ask the user for
    specific outcomes rather than automatically calculate the probabilities of
    all the possible combinations.
    """

    def __init__(self, df: pd.DataFrame, predictor: Callable) -> None:
        self.df = df
        self.predictor = predictor

    def pmf(self,
            outcome: Union[str, PosetVector, List[str], List[PosetVector]] = None,
            column: str = None, obs: Union[int, List[int]] = None,
            full: bool = False) -> np.array:
        """
        Runs the calculation of the probability of outcomes of interest

        Parameters
        ----------
        outcome
        column
        obs
        full

        Returns
        -------

        """

        if outcome is None and column is None and not full:
            raise NameError("Please define the outcome to get the probability "
                            "for. Or in the pairwise case a column for which "
                            "to make predictions, or set full to true, which "
                            "will return the probabilities of all possible "
                            "permutations.")

        if outcome is not None and column is not None:
            raise NameError("Both outcome and column has been specified, "
                            "please only feed one of these options in.")

        if obs is not None:
            df = self.df.iloc[obs].copy()
        else:
            df = self.df.copy()

        if outcome is not None:
            return self.predictor(outcome)

        if column is not None:
            return self.predictor(df[column])
