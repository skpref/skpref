from sklearn.base import BaseEstimator
from skpref.task import PrefTask, PairwiseComparisonTask, ChoiceTask
import pandas as pd
from typing import List, Type
import numpy as np
from skpref.utils import UnderDevError


class Model(BaseEstimator):
    """
    Base Class for all models
    Every model will have a fit and predict method. This is defined on the level
    in the model object e.g. BradleyTerry.
    Sometimes models will be fit on different tasks than their original design,
    for example a pairwise comparison model might be fit on a choice data that
    isn't pairwise comparison. Pairwise comparison models will assume a
    different data structure to choice models. For example a pairwise comparison
    dataset might have the following format:

    Table 1: pairwise comparison table

    |option1|option2|choice|
    |-------|-------|------|
    |Alt A  | Alt B |Alt B |

    A choice dataset might have this format:

    Table 2: choice table

    |options_presented| options_chosen|
    |-----------------|---------------|
    |[A,B,C]          |[A,C]          |

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
    def task_unpacker(self, task: PrefTask) -> dict:
        pass

    def task_packer(self, predictions: np.array, task: Type[PrefTask]
                    ) -> np.array:
        pass

    def fit(self, df_comb: pd.DataFrame, target: str,
            df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
            merge_columns: List[str] = None) -> None:
        pass

    def fit_task(self, task: PrefTask) -> None:
        if task.annotations['features_to_use'] is not None:
            self.task_fit_features = task.annotations['features_to_use'].copy()
        else:
            self.task_fit_features = None

        return self.fit(**self.task_unpacker(task))

    def predict(self, df_comb: pd.DataFrame,
                df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                merge_columns: List[str] = None) -> np.array:
        pass

    def _prepare_data_for_prediction(self, task: PrefTask) -> dict:
        if self.task_fit_features != task.annotations['features_to_use']:
            raise Exception("The task has been fitted with different features")

        # remove the target column if it exists and drop it from the dict
        task_unpack_dict = self.task_unpacker(task)
        task_unpack_dict['df_comb'] = task_unpack_dict['df_comb']\
            .drop(task_unpack_dict['target'], errors='ignore')
        del task_unpack_dict['target']
        return task_unpack_dict

    def predict_task(self, task: PrefTask) -> np.array:
        predictions = self.predict(**self._prepare_data_for_prediction(task))
        return self.task_packer(predictions, type(task))


class ProbabilisticModel(Model):

    def predict_proba(self, df_comb: pd.DataFrame, df_i: pd.DataFrame = None,
                      df_j: pd.DataFrame = None, merge_columns: List[str] = None
                      ) -> np.array:
        pass

    def predict_proba_task(self, task: PrefTask) -> np.array:
        predictions = self.predict_proba(
            **self._prepare_data_for_prediction(task))
        return self.task_packer(predictions, type(task))


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
                    task.annotations['primary_table_target_names']: 'alt1_top'},
                    inplace=True
                )

        elif (type(task) is ChoiceTask) or (
                (type(task) is PairwiseComparisonTask) and
                (self.pairwise_red_args['style'] == 'reciprocal')
        ):
            pairwise_comparisons = task.subset_vec.pairwise_reducer(
                **self.pairwise_red_args)
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
                    pairwise_comparisons = pairwise_comparisons.merge(
                        task.primary_table[feats_in_primary].reset_index()
                        .rename(columns={'index': 'observation'}),
                        how='left', on='observation', validate='m:1')

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

            if type(task) is not PairwiseComparisonTask:
                _re_indexed_df.drop(['observation'], axis=1, inplace=True)

            model_input = _re_indexed_df[['alt1_top']].copy()
            secondary_input = None

        elif task.annotations['features_to_use'] != 'all':
            model_input = _re_indexed_df[
                ['alt1_top'] +
                task.primary_table_features_to_use.tolist() +
                input_merge_columns
                ].copy()
            secondary_input = secondary_re_indexed[
                task.secondary_table_features_to_use.tolist() +
                input_merge_columns
                ].copy()
        else:
            model_input = _re_indexed_df.copy()
            secondary_input = secondary_re_indexed.copy()

        return {'df_comb': model_input,
                'target': 'alt1_top',
                'df_i': secondary_input,
                'merge_columns': input_merge_columns}


class GLMPairwiseComparisonModel(PairwiseComparisonModel, ProbabilisticModel):
    def __init__(self):
        super(GLMPairwiseComparisonModel, self).__init__()


class SVMPairwiseComparisonModel(PairwiseComparisonModel):
    def __init__(self):
        super(SVMPairwiseComparisonModel, self).__init__(
            {'style': 'reciprocal'}
        )


class ClassificationReducer(ProbabilisticModel):
    """
    This will be an object that allows users to model tasks using models that
    follow the scikit-learn structure of objects that have fit and predict
    methods.
    """
    def __init__(self, model):
        self.model = model
        self.obs_col = None
        self.keep_pairwise_format = True
        self.alternative = None

    def task_unpacker(self, task: PrefTask, keep_pairwise_format: bool = True,
                      take_feautre_diff: bool = False) -> dict:
        """
        If we have a pairwise comparison task there are two options:
        * If we allow for the data to be kept as is, in the format of
            | alternative 1 | alternative 2 | alt1_chosen |
            |---------------|---------------|-------------|

          Assume a secondary table of the form:
            | alternative | feature  |
            |-------------|----------|

          I would like to allow for two options, one is to create a final table
          as:

            Option 1

            | feauture alt 1| feature alt 2 | alt1_chosen |
            |---------------|---------------|-------------|

            Option 2

            | feauture alt 1 - feature alt 2 | alt1_chosen |
            |--------------------------------|-------------|

            For aggregation these features are recreated and then prediction is
            given

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

        take_feautre_diff: bool default is False
            This will take the difference in features like described in Option 2
            above

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
                if take_feautre_diff:
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
                model_input = task.primary_table.drop(
                    task.primary_table_alternatives_names, axis=1).copy()

            model_input.rename(columns={task.primary_table_target_name:
                                        'chosen'}, inplace=True)

        # Create aggregation for choice task
        elif isinstance(task, ChoiceTask):
            model_input = task.subset_vec.classifier_reducer()
            if task.primary_table_target_name is None:
                model_input.drop(column='chosen', inplace=True)

            if len(task.primary_table_features_to_use) > 0:
                model_input = model_input.merge(
                    task.primary_table[task.primary_table_features_to_use],
                    how='inner', left_on='observation', right_index=True,
                    validate='m:1'
                )

            if len(task.secondary_table_features_to_use) > 0:
                _, _, left_on, right_on = task.find_merge_columns(
                    original_naming=False)

                model_input = model_input.merge(
                    task.secondary_table, how='left', left_on=list(left_on),
                    right_on=list(right_on), validate='m:1')

                self.obs_col = model_input.observation.values
                self.alternative = model_input.alternative.values

                model_input = model_input[
                    ['chosen'] +
                    list(np.setdiff1d(task.secondary_table_features_to_use,
                                      right_on))
                ]

        else:
            raise UnderDevError(
                "This method has not yet been developed for this type of task")

        return {'df_comb': model_input,
                'target': 'chosen',
                'df_i': None,
                'merge_columns': None}

    def fit(self, df_comb: pd.DataFrame, target: str,
            df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
            merge_columns: List[str] = None):

        self.model.fit(df_comb.drop('chosen', axis=1),
                       df_comb['chosen'])

    def predict(self, df_comb: pd.DataFrame,
                df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                merge_columns: List[str] = None) -> np.array:

        return self.model.predict(df_comb.drop(
            'chosen', axis=1, errors='ignore'))

    def predict_proba(self, df_comb: pd.DataFrame,
                      df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                      merge_columns: List[str] = None) -> np.array:

        return self.model.predict_proba(df_comb.drop(
            'chosen', axis=1, errors='ignore'))

    def task_packer(self, predictions, task_type):
        if task_type is PairwiseComparisonTask and self.keep_pairwise_format:
            return predictions
        else:
            return pd.DataFrame({
                'prediction': np.where(predictions == 1, self.alternative,
                                       None),
                'obs': self.obs_col}).dropna().groupby('obs') \
                ['prediction'].unique().values
