"""
This module is to be used for pairwise comparison models
"""
from collections import defaultdict, OrderedDict
from warnings import warn
import warnings
import choix
import pandas as pd
import numpy as np
import pylogit as pl
from scipy.special import expit
from numpy import unique
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import log_loss


def check_indexing_of_entities(df):
    """
    Raises an error if we have the wrong amount of relationships in the
    EntitySet

    Parameters
    ----------
    df : DataFrame
         To run pairwise models on
    """

    if len(df.index.names) != 2:
        raise Exception(
            "Entity specification error! Please load in a MultiIndex pandas "
            "DataFrame which includes the entities being compared in the index."
            "example code for this: df.set_index(['entity1','entity2'])"
        )


def get_distinct_entities(df):
    """
    Returns the distinct entities compared in the DataFrame
    Note this only works for multi-indexed tables at the moment.
    Inputs:
    -------
    df : DataFrame,
         DataFrame with multi-index where each index is an entity and object
         of comparison made.

    Returns:
    --------
    list of distinct indices
    """
    unique_index_1 = unique(df.index.levels[0])
    unique_index_2 = unique(df.index.levels[1])
    both_unique_indices = np.append(unique_index_1, unique_index_2)
    return sorted(set(both_unique_indices))


def generate_entity_lookup(all_unique_ents):
    """
    Returns an entity lookup which can be used to replace entity names in the
    data to simple integers. Both choix and pylogit tend to do this.
    Inputs:
    -------
    all_unique_ents: list
        List of distinct entities.
    """

    # check that the entities provided are distinct
    if len(all_unique_ents) > len(set(all_unique_ents)):
        raise ValueError(
            "The entities provided in generate_entity_lookup are not unique!")

    rplc_lkp = {i: c for c, i in enumerate(all_unique_ents)}
    lkp = {c: i for c, i in enumerate(all_unique_ents)}

    return rplc_lkp, lkp


class BradleyTerry(BaseEstimator):
    """
    Class which fits a Bradley Terry Model based on the choix package
    hyperparamters can be recognised from the opt_pairwise function in choix

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under an isotropic Gaussian prior with variance ``1 / alpha``.

    Parameters
    ----------
    alpha : float
            Regularization strength

    method : str
            Optimization method. Either "BFGS" or "Newton-CG"

    initial_params : array_like
                     Parameters used to initialize the iterative procedure
    max_iter : int
               Maximum number of iterations allowed

    tol : float
          Tolerance for termination (method-specific)
    Examples
    ----------
    # >>> from skchoice import generate_data
    # >>> from skchoice.pairwise_model import BradleyTerry
    # >>> from sklearn.model_selection import train_test_split
    # >>> # Generate a simple pairwise experimental table
    # >>> generated_data = generate_data.GenerateData(entities=10)
    # >>> my_results = generated_data.generate_results(gen_type='full', rounds=1)
    # >>> indexed_results = my_results.set_index(['ent_i','ent_j'])
    # >>> train, test = train_test_split(indexed_results, test_size=0.33, random_state=42)
    # >>> # Fit Bradley Terry model without covariates
    # >>> bt_model = BradleyTerry()
    # >>> bt_model.fit(train, 'result')
    # >>> probabilities = bt_model.predict_proba(test)

    """
    def __init__(self, alpha=1e-6, method="Newton-CG", initial_params=None,
                 max_iter=None, tol=1e-5):
        self.alpha = alpha
        self.method = method
        self.initial_params = initial_params
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def replace_entities_with_lkp(df, lkp):
        """
        Replaces entities with a lookup
        Inputs:
        -------
        df : DataFrame,
             DataFrame with multi-index where each index is an entity and object
             of comparison made.

        lkp : dict
              lookup dictionary for replacing of the format:
              {val1: replace_val1, val2: replace_val2..}

        Returns:
        --------
        replaced_df : DataFrame
        """
        _df = df.reset_index().copy()
        for name in df.index.names:
            _df[name] = _df[name].map(lkp)

        return _df

    def unpack_data_for_choix(self, df, entnames):
        """
        Unpacks a multi-indexed pandas DataFrame in a way that can be consumed by
        the choix package.
        Inputs:
        -------
        df : DataFrame,
             DataFrame with multi-index where each index is an entity and object
             of comparison made.

        entnames : list of str with len 2
            The 'column names' of the entities e.g. ['team1', 'team2]

        Returns:
        --------
        data : dict, list
               {'winner': [(winner_id, loser_id),...]}

        n_ents : int,
                 number of entities
        """

        # Get total number of entities, a required parameter in choix
        all_unique_indices = get_distinct_entities(df)
        n_ents = len(all_unique_indices)
        _df = self.replace_entities_with_lkp(df, self.rplc_lkp)

        # Format the data the way its required in choix
        data = defaultdict(list)
        for counter, _res in enumerate(df[self.target_col_name].values):
            if _res == 1:
                data['winner'].append(
                    (_df[entnames[0]].iloc[counter],
                     _df[entnames[1]].iloc[counter])
                )

            elif _res == 0:
                data['winner'].append(
                    (_df[entnames[1]].iloc[counter],
                     _df[entnames[0]].iloc[counter])
                )

        return data, n_ents

    def unpack_data_for_pylogit(self, df, entnames):
        """
        Unpacks a Multi-indexed pandas DataFrame in a way that can be consumed
        by pylogit in long-format.

        Inputs:
        -------
        df: DataFrame
            DataFrame with multi-index where each index is an entity and object
            of comparison made, for this code to be triggered these would also
            contain some features.

        entnames: list of strings
            The name of the columns that contain the entity names.


        Returns:
        --------
        long_x_comb: DataFrame
            The DataFrame ready for pylogit modelling in long-format
        """

        availability_vars = {}

        # Get the distinct entities
        all_unique_indices = get_distinct_entities(df)
        x_comb = df.reset_index()

        ind_variables = df.drop([self.target_col_name], axis=1).columns.tolist()

        for name in entnames:
            x_comb[name] = x_comb[name].map(self.rplc_lkp)

        for i in all_unique_indices:
            colname = str(i) + '_AV'
            x_comb[colname] = np.where(
                (x_comb[entnames[0]] == self.rplc_lkp[i]) |
                (x_comb[entnames[1]] == self.rplc_lkp[i]), 1, 0)
            availability_vars[self.rplc_lkp[i]] = colname

        x_comb['CHOICE'] = np.where(x_comb[self.target_col_name] == 1,
                                    x_comb[entnames[0]],
                                    x_comb[entnames[1]])

        x_comb.drop(self.target_col_name, axis=1, inplace=True)

        custom_alt_id = 'entity'
        obs_id_column = 'observation'
        x_comb[obs_id_column] = np.arange(x_comb.shape[0], dtype=int) + 1
        choice_column = "CHOICE"

        x_comb.drop(entnames, axis=1, inplace=True)

        long_x_comb = pl.convert_wide_to_long(x_comb,
                                              ind_variables,
                                              {},
                                              availability_vars,
                                              obs_id_column,
                                              choice_column,
                                              new_alt_id_name=custom_alt_id)

        return long_x_comb

    def join_up_dataframes(self, df_comb, df_i=None, df_j=None,
                           merge_columns=None):
        """
        This function joins up the DataFrames that are provided in the fit method

        Inputs:
        -------
        df_comb : DataFrame,
             DataFrame with multi-index where each index is an entity and object
             of comparison made. This table should contain the target value and
             any other predictive features that the user would like to use and
             store on the observational level, for example the weather for a
             particular match.

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

        """

        if merge_columns is None:
            merge_columns = []

        if df_i is not None or df_j is not None:
            x_comb = df_comb.copy()

            if df_i is not None:
                x_i_entname = df_i.index.names.copy()
                _df_i = df_i.reset_index().copy()

                # dropping null values ensures that we don't have teams which
                # weren't present in the multi-indexed frame.
                _df_i['entity'] = _df_i[x_i_entname[0]].map(self.rplc_lkp)
                _df_i.dropna(inplace=True)
                _df_i.drop(x_i_entname[0], axis=1, inplace=True)

                x_comb = x_comb.merge(_df_i, how='left',
                                      left_on=['entity'] + merge_columns,
                                      right_on=['entity'] + merge_columns,
                                      validate='m:1'
                                      )
            if df_j is not None:
                x_j_entname = df_j.index.names.copy()
                _df_j = df_j.reset_index().copy()
                _df_j['entity'] = _df_j[x_j_entname[0]].map(self.rplc_lkp)
                _df_j.dropna(inplace=True)
                _df_j.drop(x_j_entname[0], axis=1, inplace=True)

                x_comb = x_comb.merge(_df_j, how='left',
                                      left_on=['entity'] + merge_columns,
                                      right_on=['entity'] + merge_columns,
                                      validate='m:1'
                                      )

        else:
            x_comb = df_comb.reset_index().copy()

        if len(merge_columns) > 0:
            x_comb.drop(merge_columns, axis=1, inplace=True)

        return x_comb

    def fit(self, df_comb, target, df_i=None, df_j=None, merge_columns=None):
        """
        This function computes the maximum-likelihood estimate of model parameters
        given pairwise-comparison data, using optimizers
        provided by the ``scipy.optimize`` module.

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

        check_indexing_of_entities(df_comb)
        self.x_comb_entnames = df_comb.index.names.copy()
        # Remember the results column so that it can be removed later
        self.target_col_name = target

        self.rplc_lkp, self.lkp = generate_entity_lookup(
            get_distinct_entities(df_comb))

        self.hyperparameters = {
            'alpha': self.alpha,
            'method': self.method,
            'initial_params': self.initial_params,
            'max_iter': self.max_iter,
            'tol': self.tol
        }

        # Training with choix
        if df_i is None and df_j is None and \
                (list(df_comb.columns) == [self.target_col_name]):
            training_data, n_ents = self.unpack_data_for_choix(df_comb,
                                                               self.x_comb_entnames)
            # Fit Bradley Terry
            self._params = choix.opt_pairwise(n_ents, training_data["winner"],
                                              **self.hyperparameters)

            self.params_ = pd.DataFrame.from_dict(self.lkp, orient='index',
                                                  columns=['entity'])

            self.params_['learned_strength'] = self._params.copy()

            self.is_fitted_ = True
            self.pylogit_fit = False

        # Training with pylogit
        else:
            if self.hyperparameters['method'] == "Newton-CG":
                warn("Note that method specified for pylogit descent is" +
                     " Newton-CG, at the point we last checked there was an" +
                     " open issue regarding the Hessian not being correct which" +
                     " is used for this type of optimization. If this issue has" +
                     " been resolved in pylogit please contact us to remove this" +
                     " warning"
                     )

            self.hyperparameters['ridge'] = self.alpha

            if df_i is not None:
                self.df_i = df_i.copy()
            else:
                self.df_i = None

            if df_j is not None:
                self.df_j = df_j.copy()
            else:
                self.df_j = None

            if merge_columns is not None:
                self.merge_columns = merge_columns.copy()
            else:
                self.merge_columns = None

            long_format = self.unpack_data_for_pylogit(df_comb,
                                                       self.x_comb_entnames)

            x_comb = self.join_up_dataframes(long_format, df_i, df_j,
                                             merge_columns)

            basic_specification = OrderedDict()
            basic_names = OrderedDict()
            columns = 0
            for i in x_comb.columns:
                if i not in ['observation', 'entity', 'CHOICE']:
                    basic_specification[i] = [list(self.lkp.keys())]
                    basic_names[i] = [i]
                    columns += 1

            basic_specification['intercept'] = list(self.lkp.keys())
            basic_names['intercept'] = [str(i) for i in self.rplc_lkp.keys()]

            self.bt_with_feats = pl.create_choice_model(
                data=x_comb, alt_id_col='entity', obs_id_col='observation',
                choice_col='CHOICE', specification=basic_specification,
                model_type="MNL", names=basic_names)

            self.x_comb = x_comb.copy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.bt_with_feats.fit_mle(np.zeros(columns + len(self.lkp)),
                                           **self.hyperparameters,
                                           print_res=False)
            self.is_fitted_ = True
            _feat_params = self.bt_with_feats.params.reset_index()
            self.params_ = _feat_params[~_feat_params['index'].isin(
                basic_names.keys())]
            self.params_.columns = ['entity', 'learned_strength']
            self.pylogit_fit = True

    @staticmethod
    def task_indexing(task):
        """
        Re-indexes the pandas DataFrame that sits in the task
        Parameters:
        -----------
        task: Choice type task

        Returns:
        ---------
        The indexed DataFrame
        """
        if task.target_column_correspondence is not None:
            _re_indexed_df = task.primary_table.set_index([
                task.target_column_correspondence,
                task.inverse_correspondence_column[0]
            ])
        else:
            _re_indexed_df = task.primary_table.set_index(
                task.annotations['primary_tale_alternatives_names'])

        return _re_indexed_df

    def fit_task(self, task):
        """
        Fits the Bradley-Terry model to a task rather than in its raw DataFrame
        form.
        Parameters:
        -----------
        task : ChoiceTask that has been loaded for the data.
        """
        _re_indexed_df = self.task_indexing(task)

        if task.annotations['features_to_use'] is None:
            model_input = _re_indexed_df[
                [task.annotations['primary_table_target_names']]].copy()
        elif task.annotations['features_to_use'] != 'all':
            model_input = _re_indexed_df[
                list(task.annotations['primary_table_target_names']) +
                task.annotations['features_to_use']
            ].copy()
        else:
            model_input = _re_indexed_df.copy()

        self.fit(model_input, task.annotations['primary_table_target_names'])

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
        check_is_fitted(self, ['params_'])

        if self.pylogit_fit:
            try:
                order = self.params_.sort_values(
                    'learned_strength', ascending=ascending).entity.values.astype(int)
            except:
                order = self.params_.sort_values(
                    'learned_strength', ascending=ascending).entity.values

            return order

        else:
            ordered_params = np.argsort(self._params)
            if ascending:
                return [self.lkp[i] for i in ordered_params]
            else:
                return [self.lkp[i] for i in ordered_params[::-1]]

    def check_for_no_new_entities(self, df):
        """ Checks if no new entities are being introduced in the scoring. It
        raises an exception if that happens.
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.

        Returns
        -------
        Exception if conditions are met
        """
        check_indexing_of_entities(df)
        new_df_ents = get_distinct_entities(df)
        newcomers = np.setdiff1d(new_df_ents, list(self.lkp.values()))

        if len(newcomers) > 0:
            raise Exception("""This DataFrame contains entities that have not
            been observed in the fitting process, which assumes an interconnected
            graph. Please resolve either by removing these observations in the
            scoring process or adding some interactions in the fitting process.
            The unobserved new entities are: %r
            """ % newcomers)

    def check_if_target_column_passed_in_pred(self, df):
        """ Checks if the results column is passed in pred and gives a warning to
        the user and removes it
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.

        Returns
        -------
        _df : DataFrame
              Without the results column
        """
        if self.target_col_name in df.columns:
            # warn("""Note that the target column ['{}'] has been passed in a
            #        prediction method, it will be ignored during this
            #        process""".format(self.target_col_name))

            return df.drop(self.target_col_name, axis=1).copy()

        else:
            return df.copy()

    def fit_checks(self, df):
        """
        Checks if the fitting process has happened correctly
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.
        """
        check_is_fitted(self, ['params_'])
        check_indexing_of_entities(df)
        self.check_for_no_new_entities(df)

    def find_strength_diff(self, df):
        """ Finds the difference in strength between a comparison
        Parameters
        ----------
        df : DataFrame
             DataFrame with multi-index where each index is an entity and object
             of comparison made.

        Returns
        -------
        diff : ndarray, shape (n_samples,)
            The difference between the strengths of the entities presented as
            comparisons in df
        """
        # Check if table is correctly loaded and fit had been called
        self.fit_checks(df)
        _df_ = self.check_if_target_column_passed_in_pred(df)

        _df = self.replace_entities_with_lkp(_df_, self.rplc_lkp)
        i_st = self._params[_df[df.index.names[0]].values]
        j_st = self._params[_df[df.index.names[1]].values]
        return i_st - j_st

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
        if self.pylogit_fit:
            self.fit_checks(df)

            reference_df = df.reset_index().copy()

            reference_df['observation'] = [
                i for i in range(1, len(reference_df) + 1)]

            reference_df['entity'] = reference_df[self.x_comb_entnames[0]]\
                .map(self.rplc_lkp)

            _df_long = self.unpack_data_for_pylogit(df, self.x_comb_entnames)

            _df_merged = self.join_up_dataframes(_df_long, self.df_i, self.df_j,
                                                 self.merge_columns)

            _df_long['preds'] = self.bt_with_feats.predict(_df_merged)

            ref_preds = reference_df.merge(
                _df_long[['preds', 'observation', 'entity']], how='left',
                on=['entity', 'observation'], validate='1:1'
            )
            return ref_preds.preds.values

        else:
            # Select the appropriate team strengths
            diff = self.find_strength_diff(df)
            return expit(diff)

    def unpack_task_for_predict(self, task):
        """
        Unpacks tasks for the consumption of the existing predict methods
        Parameters:
        -----------
        task: ChoiceTask type

        Returns:
        --------
        The DataFrame in the correct formatting for the predict methods
        """
        _re_indexed_df = self.task_indexing(task)

        if (task.annotations['features_to_use'] != 'all') and (
                task.annotations['features_to_use'] is not None):
            model_input = _re_indexed_df[
                list(task.annotations['primary_table_target_names']) +
                task.annotations['features_to_use']
                ].copy()
        else:
            model_input = _re_indexed_df.copy()

        return model_input

    def predict_proba_task(self, task):
        """
        Predicts the probability that the corresponding entity will win in the
        task.
        Parameters:
        -----------
        task: ChoiceTask type

        Returns:
        --------
        predict_proba
        """
        model_input = self.unpack_task_for_predict(task)
        return self.predict_proba(model_input)

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
        if self.pylogit_fit:
            self.fit_checks(df)
            probs = self.predict_proba(df)
            diff = probs - 0.5

        else:
            diff = self.find_strength_diff(df)

        y = np.where(diff >= 0, df.reset_index()[df.index.names[0]],
                     df.reset_index()[df.index.names[1]])
        return y

    def predict_choice_task(self, task):
        """
        Predicts the probability that the corresponding entity will win in the
        task.
        Parameters:
        -----------
        task: ChoiceTask type

        Returns:
        --------
        predict_choice
        """
        model_input = self.unpack_task_for_predict(task)
        return self.predict_choice(model_input)

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
        if self.pylogit_fit:
            self.fit_checks(df)
            probs = self.predict_proba(df)
            diff = probs - 0.5

        else:
            diff = self.find_strength_diff(df)

        return np.where(diff >= 0, 1, 0)

    def predict_task(self, task):
        """
        Predicts the probability that the corresponding entity will win in the
        task.
        Parameters:
        -----------
        task: ChoiceTask type

        Returns:
        --------
        predict
        """
        model_input = self.unpack_task_for_predict(task)
        return self.predict(model_input)

    def score(self, X):
        """
        Score function to make class scikit-learn compatible
        Inputs:
        -------
        X :
        """
        return -log_loss(X[self.target_col_name], self.predict_proba(X))
