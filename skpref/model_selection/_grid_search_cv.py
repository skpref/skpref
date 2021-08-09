from sklearn.model_selection import GridSearchCV as skGS
import pandas as pd
from skpref.base import ClassificationReducer
from skpref.task import PrefTask
from sklearn.metrics import log_loss
from typing import List, Type, Union
from skpref.data_processing import PosetVector

LOSS_FUNCTIONS = {
    'neg_log_loss': [log_loss, -1]
}


class GridSearchCV(object):
    """
    An adaption of scikit-learn's GridSearchCV into a choice model interface.
    Parameters
    -----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    kwargs : args
        all arguments that GridSearchCV takes from scikit-learn

    Example usage:
    --------------
     >>> import pickle
    >>> import sys
    >>> sys.path.insert(0, "..")
    >>> from skpref.random_utility import BradleyTerry
    >>> from skpref.task import ChoiceTask, PairwiseComparisonTask
    >>> import pandas as pd
    >>> # Using product choice data
    # >>> with open('skpref/examples/data/product_choices.pickle', 'rb') as handle:
    # ...     choice_data = pickle.load(handle)
    # >>> products_bought_train = ChoiceTask(
    # ... choice_data[:100], 'alternatives', 'choice', features_to_use=None)
    # >>> to_tune = {'alpha': [0.1, 0.5, 1], 'method': ['BFGS', 'Newton-CG']}
    # >>> gs_bt = GridSearchCV(BradleyTerry(), to_tune,  cv=3)
    # >>> gs_bt.fit_task(products_bought_train)

    >>> # using basketball match data
    >>> NBA_file_loc = 'skpref/examples/data/'
    >>> NBA_results = pd.read_csv(NBA_file_loc + 'NBA_matches.csv')
    >>> NBA_team_salary_budget = pd.read_csv(NBA_file_loc
    ...     + 'team_salary_budgets.csv')
    >>> season_split = 2016
    >>> train_data = NBA_results[NBA_results.season_start == season_split].copy()
    >>> NBA_results_task_train = PairwiseComparisonTask(
    ...     primary_table=train_data,
    ...     primary_table_alternatives_names=['team1', 'team2'],
    ...     primary_table_target_name ='team1_wins',
    ...     target_column_correspondence='team1',
    ...     features_to_use=['salary'],
    ...     secondary_table=NBA_team_salary_budget,
    ...     secondary_to_primary_link={'team': ['team1', 'team2'],
    ...                                'season_start': 'season_start'})
    >>> to_tune = {'alpha': [1, 2, 4], 'method': ['BFGS']}
    >>> gs_bt = GridSearchCV(BradleyTerry(), to_tune,  cv=3)
    >>> gs_bt.fit_task(NBA_results_task_train)
    """

    def __init__(self, estimator, param_grid, scoring=None, **kwargs):
        self.estimator = estimator
        self.kwargs = kwargs
        if isinstance(estimator, ClassificationReducer):
            self.param_grid = {}
            for _key in param_grid.keys():
                self.param_grid['model__'+_key] = param_grid[_key]

        else:
            self.param_grid = param_grid

        self.scoring = scoring

        self.gs = skGS(self.estimator, self.param_grid, return_train_score=True,
                       **self.kwargs)

    def fit(self, df_comb, target, df_i=None, df_j=None, merge_columns=None):
        """
        This function fits the model that is fed into the GridSearchCV
        Parameters
        ------------
        df_comb : DataFrame
                  DataFrame with multi-index where each index is an entity and object
                  of comparison made. This table should contain the target value and
                  any other predictive features that the user would like to use and
                  store on the observational level, for example the weather for a
                  particular match.
        target : str
                 name of target variable column.
        df_i : DataFrame, default: None
               DataFrame where the index is an entity and the values are
               features for predictions that the user would like to store on the
               entity level. For example the budget of a team. This information
               may be merged onto the df_comb table, however this is available
               for purpose of ease if the user has this stored in a way most
               relational database users would stored data. If the column names
               are repeated in df_i and df_comb there will be an error raised.
        df_j : DataFrame, default: None
               Same as df_i, but can be used in case the second entity compared
               is always of a different nature and isn't stored in the same
               DataFrame. For example movies might be compared with songs.
        merge_columns : list of str, default: None
                        Any columns that exists in the df_comb DataFrame, df_i and df_j
                        DataFrames that the users would like to also merge on such as year,
                        note that the entities in the indices will be automatically
                        considered.
        Returns
        ---------
        Self
        """

        self.gs.fit(df_comb, target=target, df_i=df_i, df_j=df_j,
                    merge_columns=merge_columns)

        # make sure all the GridSearchCV attributes are callable as expected
        self.cv_results_ = self.gs.cv_results_
        self.best_params_ = self.gs.best_params_
        self.best_estimator_ = self.gs.best_estimator_
        self.best_score_ = self.gs.best_score_
        self.best_params_ = self.gs.best_params_
        self.best_index_ = self.gs.best_index_
        self.scorer_ = self.gs.scorer_
        self.n_splits_ = self.gs.n_splits_
        self.refit_time_ = self.gs.refit_time_

    def fit_task(self, task: PrefTask):
        if isinstance(self.scoring, str) or callable(self.scoring):

            def task_scorer(function: callable, table: pd.DataFrame,
                            depvar=task.primary_table_target_name):
                """
                Making special scorer
                Parameters
                ----------
                function
                table
                depvar

                Returns
                -------

                """
                X = table.drop([depvar], axis=1).copy()
                if isinstance(self.scoring, str):
                    lf = LOSS_FUNCTIONS[self.scoring][0]
                else:
                    lf = self.scoring
                try:
                    lf([1, 0], [0.5, 0.5])
                    _preds = function.predict_proba(X)
                    if len(_preds.shape) == 2:
                        if _preds.shape[1] == 2:
                            pred = [pred[1] for pred in _preds]
                        else:
                            raise Exception("too many predictions")
                    else:
                        pred = _preds.copy()
                except ValueError:
                    pred = function.predict(X)

                y = table[depvar].copy()
                if isinstance(self.scoring, str):
                    return lf(y, pred) * LOSS_FUNCTIONS[self.scoring][1]
                else:
                    return lf(y, pred)

            self.gs.scoring = task_scorer


        self.fit(**self.estimator.task_unpacker(task))

        # make sure all the GridSearchCV attributes are callable as expected
        self.cv_results_ = self.gs.cv_results_
        self.best_params_ = self.gs.best_params_
        self.best_estimator_ = self.gs.best_estimator_
        self.best_score_ = self.gs.best_score_
        self.best_params_ = self.gs.best_params_
        self.best_index_ = self.gs.best_index_
        self.scorer_ = self.gs.scorer_
        self.n_splits_ = self.gs.n_splits_
        self.refit_time_ = self.gs.refit_time_

        if task.annotations['features_to_use'] is not None:
            self.best_estimator_.task_fit_features = task.annotations[
                'features_to_use'].copy()
        else:
            self.best_estimator_.task_fit_features = None

    def inspect_results(self):
        """
        Returns the results of the grid-search in a user-friendly way.
        """
        print("The model with the best parameters was:")
        print(self.best_estimator_)
        print("With a score of", self.best_score_)
        print("All the trials results summarised in descending score")
        res = pd.DataFrame(self.cv_results_['params'])
        res['mean_test_score'] = self.cv_results_['mean_test_score']
        print(res.sort_values('mean_test_score', ascending=False))

    def rank_entities(self, ascending=True):
        """ Outputs the ranked order of entities.
        Parameters
        ----------
        ascending : Boolean, default=True
                    When True the weakest entity will be first in the list,
                    when False the strongest entity will be first in the list.
        Returns
        -------
        rank : ndarray, shape (n_ents)
               The ranks of the entities.
        """
        return self.best_estimator_.rank_entities(ascending=ascending)

    def predict(self, df):
        """ Predicts the result (1,0) of comparison where the leftmost indexed
        entity being chosen is labelled as 1.
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.
        Returns
        -------
        y : ndarray, shape (n_samples)
            The entity which is expected to be chosen.
        """
        return self.gs.predict(df)

    def predict_task(self, task):

        return self.best_estimator_.predict_task(task)

    def predict_proba(self, df):
        """ Predicts the probability of the result = 1 in the match up.
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The probability for each match up that the left indexed entity
            is chosen over the right indexed entity.
        """
        return self.gs.predict_proba(df)

    def predict_proba_task(self, task: PrefTask,
            outcome: Union[str, PosetVector, List[str], List[PosetVector]] = None,
            column: str = None):

        return self.best_estimator_.predict_proba_task(task, outcome, column)

    def predict_choice(self, df):
        """ Predicts the entity that will be selected.
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.
        Returns
        -------
        y : ndarray, shape (n_samples)
            The entity which is expected to be chosen.
        """
        return self.best_estimator_.predict_choice(df)

    def predict_choice_task(self, task):

        return self.best_estimator_.predict_choice_task(task)
