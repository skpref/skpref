from sklearn.base import BaseEstimator
from skpref.task import PrefTask, PairwiseComparisonTask, ChoiceTask
import pandas as pd
from typing import List, Type, Union
import numpy as np
from skpref.utils import (
    UnderDevError, aggregate_full_probability_predictions_to_discrete_choice)
from skpref.data_processing import PosetVector, SubsetPosetVec


class Model(BaseEstimator):
    """Base Class for all models

    Every model will have a fit and predict method. This is defined on the level
    in the model object e.g. BradleyTerry.
    Sometimes models will be fit on different tasks than their original design,
    for example a pairwise comparison model might be fit on a choice data that
    isn't pairwise comparison. Pairwise comparison models will assume a
    different data structure to choice models. For example a pairwise comparison
    dataset might have the following format:

    Table 1: pairwise comparison table

    +----------+----------+--------+
    | option 1 | option 2 | choice |
    +==========+==========+========+
    |   Alt A  | Alt B    | Alt B  |
    +----------+----------+--------+

    A choice dataset might have this format:

    Table 2: choice table

    +-------------------+----------------+
    | options_presented | options_chosen |
    +===================+================+
    |     [A, B, C]     | [A, C]         |
    +-------------------+----------------+

    So pairwise comparison models and choice models would deal with different
    input data. But the user might want to run the choice task as if it were a
    pairwise comparison task which would have to look like table 1.

    The task_unpackers will be methods whose job it would be to translate any
    dataset type into the format that is required by the model, for example all
    pairwise comparison models will have a task_unpacker that leaves the data
    as is if it looks like table 1 and changes the data into something like
    table 1 when it looks like table 2.

    The task_packers will be used to predict the data on the right level. For
    example if a pairwise comparison model is used to predict a choice then the
    task_packer will be what aggregates the data up to something like table 2.

    The fit_task function will be simply a wrapper that returns
    model.fit(task_unpacker(data)) and the predict_task function will be one that
    returns task_packer(model.preidct(data)) for every model.

    This architecture is defined at the highest level and is fixed.
    """
    def __init__(self):
        self.model_type = None

    def task_unpacker(self, task: PrefTask) -> dict:
        pass

    def task_packer(self, predictions: Union[PosetVector, np.array],
                    task: PrefTask
                    ) -> PosetVector:
        pass

    def fit(self, df_comb: pd.DataFrame, target: str,
            df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
            merge_columns: List[str] = None) -> None:
        pass

    def fit_task(self, task: PrefTask) -> None:
        """
        Fits the model using the details given in the task

        Parameters
        ----------
        task: PrefTask
              The task that has been set up by the user

        """
        if task.annotations['features_to_use'] is not None:
            self.task_fit_features = task.annotations['features_to_use'].copy()
        else:
            self.task_fit_features = None

        return self.fit(**self.task_unpacker(task))

    def predict(self, df_comb: pd.DataFrame,
                df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                merge_columns: List[str] = None) -> PosetVector:
        pass

    def _prepare_data_for_prediction(self, task: PrefTask) -> dict:
        if self.task_fit_features != task.annotations['features_to_use']:
            raise Exception("The task has been fitted with different features")

        # remove the target column if it exists and drop it from the dict
        task_unpack_dict = self.task_unpacker(task)
        task_unpack_dict['df_comb'] = task_unpack_dict['df_comb']\
            .drop(task_unpack_dict['target'], axis=1, errors='ignore')
        del task_unpack_dict['target']
        return task_unpack_dict

    def predict_task(self, task: PrefTask) -> PosetVector:
        """
        Predicts outcomes using a task

        Parameters
        ----------
        task: PrefTask
              The task that has been set up by the user

        Returns
        -------
        PosetVector of the predicted preferences

        """
        predictions = self.predict(**self._prepare_data_for_prediction(task))
        return self.task_packer(predictions, task)

    # Creating this to aid aggregation of probabilistic models that need a
    # probability prediction to aggregate an outcome prediction, such as classifiers
    # into discrete choice
    base_predict_task = predict_task


class ProbabilisticModel(Model):

    def predict_proba(self, df_comb: pd.DataFrame, df_i: pd.DataFrame = None,
                      df_j: pd.DataFrame = None, merge_columns: List[str] = None
                      ) -> np.array:
        pass

    def prediction_wrapper(self, task: PrefTask, predictions: np.array,
                           outcome: Union[str, PosetVector, List[str], List[PosetVector]] = None,
                           column: Union[str, PosetVector, List[str], List[PosetVector]] = None) -> dict:

        """
        Wraps predictions given by models into the format of a dictionary of SubsetPoset vectors
        Parameters
        ----------
        task: The preference task
        predictions: The predictions of the model
        outcome: The specific outcome that the user wants to query, this can be a string in the form of the "name" of
            the alternative, or a list of strings. It can be also be a SubsetPosetVector or a list of these.
        column: If the alternatives are in a column then it can be the column that the SubsetPosetVector contains.

        Returns
        -------
        Dictionary of results as keys and the probabilities that they will be selected in each observation, or the name
        of the column and the probability that the alternative or alternatives in the column are selected in each
        observation
        """

        if len(predictions.shape) == 2:
            target_col_wins = np.array([l[1] for l in predictions])
            target_col_loses = np.array([l[0] for l in predictions])

        else:
            target_col_wins = predictions
            target_col_loses = np.ones(len(target_col_wins)) - target_col_wins

        if type(task) is PairwiseComparisonTask and self.keep_pairwise_format:

            target_col = task.target_column_correspondence
            other_col = np.setdiff1d(task.primary_table_alternatives_names,
                                     target_col)

            if isinstance(outcome, str):
                return {
                    outcome: np.where(

                        task.primary_table[target_col].values == outcome,
                        target_col_wins, np.where(
                            task.primary_table[other_col[0]].values == outcome,
                            target_col_loses, 0))
                }

            elif isinstance(outcome, list):
                pred_probs = {}
                for i in outcome:
                    pred_probs[i] = np.where(
                        task.primary_table[target_col].values == i,
                        target_col_wins, np.where(
                            task.primary_table[other_col[0]].values == i,
                            target_col_loses, 0))

                return pred_probs

            elif isinstance(column, str):
                return {
                    column + ' is preferred': np.where(
                        task.primary_table[column].values == task.primary_table[target_col].values,
                        target_col_wins, np.where(
                            task.primary_table[column].values == task.primary_table[other_col[0]].values,
                            target_col_loses, 0))
                }

            elif isinstance(column, list):
                pred_probs = {}
                for i in column:
                    pred_probs[i + ' is preferred'] = np.where(
                        task.primary_table[i].values == task.primary_table[target_col].values,
                        target_col_wins, np.where(
                            task.primary_table[i].values == task.primary_table[other_col[0]].values,
                            target_col_loses, 0))

                return pred_probs

        if type(task) == ChoiceTask:

            if isinstance(outcome, str):
                outcome = [outcome]

            pred_probs = {}

            for _outcome in outcome:
                # For each observation check if the outcome is a possibility amongst the alternatives
                preds = []
                for n in range(np.max(task.obs_col)+1):
                    # Find all the alternatives that are in the observation
                    pos = np.where(task.obs_col == n)[0].tolist()
                    alts = task.alternative[tuple([[pos]])][0].copy()

                    # If the outcome is not a possibility set the probability to 0,
                    # otherwise find the relevant probability
                    if _outcome in alts:
                        outcome_pos = pos[np.where(alts == _outcome)[0][0]]
                        preds.append(target_col_wins[outcome_pos])
                    else:
                        preds.append(0)

                pred_probs[_outcome] = np.array(preds)

            return pred_probs

    def predict_proba_task(
            self, task: PrefTask,
            outcome: Union[str, PosetVector, List[str], List[PosetVector]] = None,
            column: str = None,
            aggregation_method: str = None
    ) -> dict:

        """
        Predicts the probability of specified outcomes for a specific task

        Parameters
        ----------
        task: PrefTask
              The task for which predictions should be made

        outcome: List
                 The outcome for which predictions should be made, for example if
                 the alternatives are 'Car', 'Train', 'Bicycle' then the user can
                 ask for probabilities of ['Car', 'Train] if they're only interested
                 in the probability of choosing 'Car' or 'Train'

        column: str
                Can also take a column name for which predictions should be made,
                probably more useful in pairwise comparison set ups, where team1
                is in one column and team2 in another.

        Returns
        -------
        A dictionary with the alternatives being the keys and for each key there's
        a numpy array of floats which reflects the probability with which that
        alternative will be selected. When the alternative is not in the list of
        choices for a specific row the value will be 0. When a column is given
        instead of an outcome then the keys are the column name.
        """

        if outcome is None and column is None:
            raise NameError("Please define the outcome to get the probability "
                            "for. Or in the pairwise case a column for which "
                            "to make predictions, or set full to true, which "
                            "will return the probabilities of all possible "
                            "permutations.")

        if outcome is not None and column is not None:
            raise NameError("Both outcome and column has been specified, "
                            "please only feed one of these options in.")

        predictions = self.predict_proba(
            **self._prepare_data_for_prediction(task))

        return self.prediction_wrapper(task, predictions, outcome, column)

    predict_proba_task_base = predict_proba_task

    def predict_task(self, task: ChoiceTask, aggregation_method: str = None
                     ) -> PosetVector:
        if type(task) == ChoiceTask and (
                self.model_type == "Classifier" or
                self.model_type == 'Pairwise Comparison'):
            # Get all the distinct alternatives that are in alternatives
            # Run probabilistic predictions for all the alternatives
            preds = self.predict_proba_task(
                task, outcome=list(task.subset_vec.entity_universe),
                aggregation_method=aggregation_method)
            return self.task_packer(preds, task)
        else:
            return self.base_predict_task(task)


class PairwiseComparisonModel(Model):
    """
    All pairwise models will take a pandas DataFrame with multi-index where
    each index is an entity and object of comparison made. This table should
    contain the target value and any other predictive features that the user
    would like to use and store on the observational level, for example the
    weather for a particular match.
    """

    def __init__(self, pairwise_red_args: dict = None):
        if pairwise_red_args is None:
            pairwise_red_args = {}
        self.pairwise_red_args = pairwise_red_args
        self.model_type = "Pairwise Comparison"

    def task_indexing(self, task: PrefTask) -> \
            (pd.DataFrame, pd.DataFrame, List[str]):
        """
        Re-indexes the pandas DataFrame that sits in the task
        Parameters:
        -----------
        task: Choice type task

        Returns:
        ---------
        The indexed DataFrame
        """
        secondary_re_indexed, input_merge_columns, _, _ =\
            task.find_merge_columns()

        if type(task) is ChoiceTask and task.primary_table_target_name is None:
            self.pairwise_red_args['scramble'] = False
            self.pairwise_red_args['style'] = 'positive'

        if isinstance(task, PairwiseComparisonTask) and (
                self.pairwise_red_args == {}):

            _re_indexed_df = task.primary_table.set_index(
                task.primary_table_alternatives_names)

            if task.annotations['primary_table_target_names'] is not None:
                _re_indexed_df.rename(columns={
                    task.annotations['primary_table_target_names']:
                        task.primary_table_target_name},
                    inplace=True
                )

        elif (type(task) is ChoiceTask) or (
                (type(task) is PairwiseComparisonTask) and
                (self.pairwise_red_args['style'] == 'reciprocal')
        ):
            pairwise_comparisons = \
                task.subset_vec.pairwise_reducer(
                    target_colname=task.primary_table_target_name,
                    **self.pairwise_red_args)[0].copy()

            self.unpacked_observations = task.subset_vec.pairwise_reducer(
                    target_colname=task.primary_table_target_name,
                    **self.pairwise_red_args)[1]

            if task.annotations['features_to_use'] is not None:
                feats_in_primary = []

                if input_merge_columns is not None:
                    feats_in_primary += input_merge_columns

                if len(np.intersect1d(task.primary_table.columns,
                                      task.annotations['features_to_use'])) > 0:

                    feats_in_primary += np.intersect1d(
                        task.primary_table.columns,
                        task.annotations['features_to_use']).tolist()

                if len(feats_in_primary) > 0:
                    feats_in_primary = np.unique(feats_in_primary).tolist()
                    pairwise_comparisons['observation'] = \
                        self.unpacked_observations
                    pairwise_comparisons = pairwise_comparisons.merge(
                        task.primary_table[feats_in_primary]
                            .reset_index(drop=True).reset_index()
                            .rename(columns={'index': 'observation'}),
                        how='left', on='observation', validate='m:1')

                    pairwise_comparisons.drop('observation', axis=1,
                                              inplace=True)

            _re_indexed_df = pairwise_comparisons.set_index(['alt1', 'alt2'])

        else:
            raise Exception(f'Conversion from {type(task)} to Pairwise '
                            f'Comparison models currently not supported')

        if input_merge_columns is None:
            input_merge_columns = []

        return _re_indexed_df, secondary_re_indexed, input_merge_columns

    def task_unpacker(self, task: PrefTask) -> dict:

        _re_indexed_df, secondary_re_indexed, input_merge_columns = \
            self.task_indexing(task)

        if task.annotations['features_to_use'] is None:

            if task.primary_table_target_name is not None:
                model_input = _re_indexed_df[[task.primary_table_target_name]].copy()
            else:
                model_input = _re_indexed_df.copy()

            # Secondary table can always be ignored when no features are being
            # used
            secondary_input = None

        elif task.annotations['features_to_use'] != 'all':
            model_input = _re_indexed_df[
                [task.primary_table_target_name] +
                list(task.primary_table_features_to_use) +
                input_merge_columns
                ].copy()

            if len(task.secondary_table_features_to_use) == 0:
                secondary_input = None

            else:
                secondary_input = secondary_re_indexed[
                    list(task.secondary_table_features_to_use) +
                    input_merge_columns
                    ].copy()
        else:
            model_input = _re_indexed_df.copy()
            secondary_input = secondary_re_indexed.copy()

        return {'df_comb': model_input,
                'target': task.primary_table_target_name,
                'df_i': secondary_input,
                'merge_columns': input_merge_columns}

    def task_packer(self, predictions: Union[np.array, dict], task: PrefTask
                    ) -> SubsetPosetVec:
            """
            When a task is set up as a pairwise comparison, the format will be
            option1, option 2, 1/0 var that corresponds to on of the options being
            successful. This function takes 1/0 predictions and maps them to the
            alternatives to pack a SubsetPosetVector with the top and boot results.
            For example if the data is
            Option 1: [A, B, C]
            Option 2: [B, A, A]
            predictions: [1, 1, 0]
            Then we'd return
            SubsetPosetVector.top_input_data = np.array([A,B,A])
            SubsetPosetVector.boot_input_data = np.array([B,A,C])

            Parameters
            ----------
            predictions: numpy arrays
                         The 1/0 predictions for pairwise comparisons
            task: PairwiseComparisonTask
                  The task for which predictions have been made

            Returns
            -------
            A SubsetPosetVector with the results
            """
            if type(task) is PairwiseComparisonTask:
                target_col = task.target_column_correspondence
                other_col = np.setdiff1d(task.primary_table_alternatives_names,
                                         target_col)
                top = np.array([])
                boot = np.array([])

                for _i, pred in enumerate(predictions):
                    if pred == 1:
                        top = np.append(top,
                                        task.primary_table[target_col].iloc[_i])
                        boot = np.append(boot,
                                         task.primary_table[other_col].iloc[_i])

                    else:
                        top = np.append(top,
                                        task.primary_table[other_col].iloc[_i])
                        boot = np.append(boot,
                                         task.primary_table[target_col].iloc[_i])

                return SubsetPosetVec(top_input_data=np.array(top),
                                      boot_input_data=np.array(boot))

            elif type(task) is ChoiceTask and type(predictions) is dict:

                return aggregate_full_probability_predictions_to_discrete_choice(
                    predictions, task)

            else:
                print("This type of aggregation has not yet been developed")


class GLMPairwiseComparisonModel(PairwiseComparisonModel, ProbabilisticModel):
    def __init__(self):
        super(GLMPairwiseComparisonModel, self).__init__()

    @staticmethod
    def compile_predictions(
                            outcome: Union[str, List[str]],
                            df: pd.DataFrame,
                            task: PrefTask) -> dict:

        if type(outcome) != list:
            _outcome = list(outcome)
        else:
            _outcome = outcome
        all_preds = {}
        for _ocm in _outcome:
            preds = []
            for obs in range(len(task.primary_table)):
                obs_table = df[df['obs_index'] == obs]
                if _ocm in obs_table[task.primary_table_alternatives_names].values:
                    prob = obs_table[
                        obs_table[
                            task.primary_table_alternatives_names]
                        == _ocm].probability.values[0]
                else:
                    prob = 0

                preds.append(prob)

            all_preds[_ocm] = np.array(preds)

        return all_preds

    def predict_proba_task(
            self, task: PrefTask,
            outcome: Union[str, PosetVector, List[str], List[PosetVector]] = None,
            column: str = None,
            aggregation_method: str = 'independent transitive'
    ) -> dict:

        if type(task) == PairwiseComparisonTask:
            return self.predict_proba_task_base(task, outcome=outcome,
                                                column=column)
        elif (type(task) == ChoiceTask and
              aggregation_method == 'independent transitive'):

            all_comb = (
                task.primary_table[[task.primary_table_alternatives_names]]
                    .explode(task.primary_table_alternatives_names)
                    .merge(task.primary_table
                           .drop(task.primary_table_target_name, axis=1)
                           .explode(task.primary_table_alternatives_names),
                           how='outer', left_index=True, right_index=True
                           )
            )

            no_repeats = (
                all_comb[
                    all_comb[task.primary_table_alternatives_names + '_x'] !=
                    all_comb[task.primary_table_alternatives_names + '_y']]
                .copy()
                .rename(columns={
                    task.primary_table_alternatives_names + '_x': 'alt1',
                    task.primary_table_alternatives_names + '_y': 'alt2'})
            )

            df_mapping = pd.DataFrame({
                'sort_values': task.primary_table.index.values,
            })

            sort_mapping = df_mapping.reset_index().set_index('sort_values')

            no_repeats['prob'] = (
                self.predict_proba(no_repeats.set_index(['alt1', 'alt2']))
            )

            choice_prob = (
                no_repeats.reset_index()[['alt1', 'index', 'prob']]
                .groupby(['alt1', 'index'], as_index=False).prod()
            )

            choice_prob['normaliser'] = (
                choice_prob[['index', 'prob']]
                .groupby('index').transform('sum')
            )

            choice_prob['probability'] = (
                    choice_prob['prob'] / choice_prob['normaliser']
            )

            choice_prob['obs_index'] = (
                choice_prob['index'].map(sort_mapping['index'])
            )

            tab_merged = (
                choice_prob.sort_values('obs_index')
                .rename(
                    columns={'alt1': task.primary_table_alternatives_names})
            )

            return self.compile_predictions(outcome, tab_merged, task)

    predict_proba_task_GLM_base = predict_proba_task


class SVMPairwiseComparisonModel(PairwiseComparisonModel):
    def __init__(self):
        super(SVMPairwiseComparisonModel, self).__init__(
            {'style': 'reciprocal'}
        )


class ClassificationReducer(ProbabilisticModel):
    """Allows users to fit scikit-learn classifiers as predictors

    This is an object that allows users to model tasks using models that
    follow the scikit-learn structure of objects that have fit and predict
    methods.

    Parameters
    ----------
    model: scikit-learn type model that user would like to fit
    take_feature_diff_for_pairwise_comparison: bool, default = False

        Assume we have a pairwise comparison with an alternative level table:

        .. list-table:: Assume an alternative level table of the format
           :widths: 25 25 25
           :header-rows: 1

           * - alternative 1
             - alternative 2
             - alt1_chosen

           * -
             -
             -

        .. list-table:: Assume an alternative level table of the format
           :widths: 25 25
           :header-rows: 1

           * - alternative
             - feature

           * -
             -

        There are two options for users to use these features:

        .. list-table:: Option 1: default setting, keeps the features separate
                for both entities and learns separate parameters on them
           :widths: 30 30 25
           :header-rows: 1

           * - feauture alt 1
             - feature alt 2
             - alt1_chosen

           * -
             -
             -

        .. list-table:: Option 2: when take_feature_diff_for_pairwise_comparison
              is set to  True then it creates one covariate which is the
              difference between the values of for the two alternatives in the
              pairwise comparison
           :widths: 60 25
           :header-rows: 1

           * - feauture alt 1 - feature alt 2
             - alt1_chosen

           * -
             -

    Example
    --------
    >>> import sys
    >>> sys.path.insert(0, "..")
    >>> from skpref.base import ClassificationReducer
    >>> from skpref.task import PairwiseComparisonTask
    >>> from sklearn.linear_model import LogisticRegression
    >>> import pandas as pd
    >>> # Using product choice data
    >>> # using basketball match data
    >>> NBA_file_loc = 'skpref/examples/data/NBA_matches.csv'
    >>> NBA_results = pd.read_csv(NBA_file_loc)
    >>> season_split = 2016
    >>> train_data = NBA_results[NBA_results.season_start == season_split].copy()
    >>> NBA_results_task_train = PairwiseComparisonTask(
    ... primary_table=train_data,
    ... primary_table_alternatives_names=['team1', 'team2'],
    ... primary_table_target_name ='team1_wins',
    ... target_column_correspondence='team1', features_to_use=['team_1_home'])
    >>> my_log_red = ClassificationReducer(LogisticRegression(solver='lbfgs'))
    >>> my_log_red.fit_task(NBA_results_task_train)

    """

    def __init__(self, model, take_feature_diff_for_pairwise_comparison: bool = False):
        self.model = model
        self.obs_col = None
        self.keep_pairwise_format = True
        self.alternative = None
        self.model_type = "Classifier"
        self.take_feature_diff_for_pairwise_comparison = take_feature_diff_for_pairwise_comparison

    def task_unpacker(self, task: PrefTask, keep_pairwise_format: bool = True) -> dict:
        """

        * We can break down the table into individual observations like so in
          long format:

            | alternative | feature | chosen | observation |
            |-------------|---------|--------|-------------|

          And then aggregate up by predicting probability on an observation
          level and the alternative inside the observation that has the highest
          probability gets chosen. this would be the same approach with subset
          selection.

        Parameters
        ----------
        task: PrefTask
            The task that has been initialised for the preference learning

        keep_pairwise_format: bool default is True
            This will control whether to keep pairwise format for a pairwise task
            it is only relevant for a PairwiseComparisonTask.

        Returns
        -------
        dict
            {'df_comb': contains the dataset with the dependent variable and
                features to predict
             'target': str of the column name of the target variable}

        """

        self.keep_pairwise_format = keep_pairwise_format
        # Create aggregation for pairwise task
        if isinstance(task, PairwiseComparisonTask) and keep_pairwise_format:

            # Create the table like it is in Option 1
            if len(task.secondary_table_features_to_use) > 0:

                _, _, left_on, right_on, = task.find_merge_columns()

                first_alternative_on = left_on[np.where(
                    left_on != task.primary_table_alternatives_names[1])]

                second_alternative_on = left_on[np.where(
                    left_on != task.primary_table_alternatives_names[0]
                )]

                # Hacky solution to make sure suffixes work as we'd like them to
                initialise_cols = list(task.secondary_table.columns)
                initialise_cols = [ele for ele in initialise_cols
                                   if ele not in right_on]

                model_input = task.primary_table.copy()
                for i in initialise_cols:
                    model_input[i] = None

                # prepare to drop alternative keys merged on
                drop_cols = [i + '_' + task.primary_table_alternatives_names[0]
                             for i in right_on]

                drop_cols += [i + '_' + task.primary_table_alternatives_names[1]
                              for i in right_on]

                drop_cols += initialise_cols

                drop_cols += task.primary_table_alternatives_names

                # When column names to merge on are different between primary
                # and secondary tables, make sure to remove these also
                for _key in (task.secondary_to_primary_link.keys()):
                    _val = task.secondary_to_primary_link[_key]
                    if len(_val) < 2:
                        if _key != _val:
                            drop_cols.append(_key)
                    else:
                        _strikes = 0
                        for _part_val in _val:
                            if _part_val != _key:
                                _strikes += 1
                        if _strikes == len(_val):
                            drop_cols.append(_key)

                model_input = model_input.merge(
                    task.secondary_table, how='left',
                    left_on=list(first_alternative_on),
                    right_on=list(right_on),
                    validate='m:1',
                    suffixes=['', '_' + task.primary_table_alternatives_names[0]]
                ).merge(
                    task.secondary_table, how='left',
                    left_on=list(second_alternative_on),
                    right_on=list(right_on),
                    validate='m:1',
                    suffixes=['', '_' + task.primary_table_alternatives_names[1]]
                ).drop(drop_cols, axis=1, errors='ignore')

                # Create the table like it is in Option 2
                if self.take_feature_diff_for_pairwise_comparison:
                    for root_col in initialise_cols:
                        model_input[root_col + '_diff'] = model_input[
                            root_col + '_' +
                            task.primary_table_alternatives_names[0]
                                                          ] - model_input[
                            root_col + '_' +
                            task.primary_table_alternatives_names[1]
                        ]

                        model_input.drop([
                            root_col + '_' +
                            task.primary_table_alternatives_names[0],
                            root_col + '_' +
                            task.primary_table_alternatives_names[1]
                        ], inplace=True, axis=1)

            else:
                model_input = task.primary_table[
                    [task.primary_table_target_name] +
                    task.primary_table_features_to_use.tolist()].copy()

        # Create aggregation for choice task
        elif isinstance(task, ChoiceTask):
            model_input, task.obs_col, task.alternative = \
                task.subset_vec.classifier_reducer(
                chosen_name=task.primary_table_target_name)
            if task.primary_table_target_name is None:
                model_input.drop(column=task.primary_table_target_name,
                                 inplace=True)

            """
            If we have primary table features or keys that need merging on after
            a reduction we do it here. First we need to figure out what are the
            keys for merging on that aren't already in the table, the name of the
            alternatives column might have change to 'alternative', so we need to
            remove the original name given from the list of columns.
            """

            if task.secondary_to_primary_link is not None:
                merge_keys = list(np.setdiff1d(
                    np.setdiff1d(list(task.secondary_to_primary_link.values()),
                                 task.primary_table_alternatives_names),
                    list(model_input.columns)))
            else:
                merge_keys = []

            if len(task.primary_table_features_to_use) > 0 or len(merge_keys) > 0:
                model_input['observation'] = task.obs_col
                model_input = model_input.merge(
                    task.primary_table[
                        task.primary_table_features_to_use.tolist() + merge_keys]
                    .reset_index(drop=True),
                    how='inner', left_on='observation', right_index=True,
                    validate='m:1'
                )

            if len(task.secondary_table_features_to_use) > 0:
                _, _, left_on, right_on = task.find_merge_columns(
                    original_naming=False)

                model_input = model_input.merge(
                    task.secondary_table, how='left', left_on=list(left_on),
                    right_on=list(right_on), validate='m:1')

                model_input = model_input[
                    [task.primary_table_target_name] +
                    list(np.setdiff1d(task.secondary_table_features_to_use,
                                      right_on))
                ]
        else:
            raise UnderDevError(
                "This method has not yet been developed for this type of task")

        if ('observation' in model_input.columns and
                'alternative' in model_input.columns):

            model_input.drop(['observation', 'alternative'],
                             axis=1, inplace=True)

        self.model_input = model_input

        return {'df_comb': model_input,
                'target': task.primary_table_target_name,
                'df_i': None,
                'merge_columns': None}

    def fit(self, df_comb: pd.DataFrame, target: str,
            df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
            merge_columns: List[str] = None):

        self.model.fit(df_comb.drop(target, axis=1),
                       df_comb[target])

    def predict(self, df_comb: pd.DataFrame,
                df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                merge_columns: List[str] = None) -> np.array:

        return self.model.predict(df_comb)

    def predict_proba(self, df_comb: pd.DataFrame,
                      df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                      merge_columns: List[str] = None) -> np.array:

        return self.model.predict_proba(df_comb)

    def task_packer(self, predictions, task) -> SubsetPosetVec:
        if type(task) is PairwiseComparisonTask and self.keep_pairwise_format:
            _dummy = PairwiseComparisonModel()
            return _dummy.task_packer(predictions, task)

        elif (type(task) is ChoiceTask and type(predictions) is dict and
                hasattr(self.model, "predict_proba")):

            return aggregate_full_probability_predictions_to_discrete_choice(
                predictions, task
            )

        else:

            preds_df_acc = pd.DataFrame({
                'prediction': np.where(predictions == 1, task.alternative,
                                       None),
                'obs': task.obs_col}).dropna().groupby('obs')

            preds_df_rej = pd.DataFrame({
                'rejection': np.where(predictions == 0, task.alternative,
                                      None),
                'obs': task.obs_col}).dropna().groupby('obs')

            return SubsetPosetVec(
                top_input_data=preds_df_acc['prediction'].unique().values,
                boot_input_data=preds_df_rej['rejection'].unique().values)

    def score(self, X, *args, **kwargs):
        y = X.chosen.values
        _X = X.drop('chosen', axis=1)
        return self.model.score(_X, y, *args, **kwargs)
