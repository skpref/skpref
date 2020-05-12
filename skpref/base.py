from sklearn.base import BaseEstimator
from skpref.task import PrefTask, PairwiseComparisonTask, ChoiceTask
import pandas as pd
from typing import List, Type
import numpy as np


class Model(BaseEstimator):
    """Base Class for all models"""
    def task_unpacker(self, task: PrefTask) -> dict:
        pass

    def task_packer(self, predictions: np.array,
                    task: Type[PrefTask]) -> np.array:
        pass

    def fit(self, df_comb: pd.DataFrame, target: str,
            df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
            merge_columns: List[str] = None) -> None:
        pass

    def fit_task(self, task: PrefTask) -> None:
        self.task_fit_features = task.annotations['features_to_use'].copy()
        return self.fit(**self.task_unpacker(task))

    def predict(self, df_comb: pd.DataFrame,
                df_i: pd.DataFrame = None, df_j: pd.DataFrame = None,
                merge_columns: List[str] = None) -> np.array:
        pass

    def predict_task(self, task: PrefTask) -> np.array:
        if self.task_fit_features != task.annotations['features_to_use']:
            raise Exception("The task has been fitted with different features")
        predictions = self.predict(**self.task_unpacker(task))
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
        input_merge_columns = None
        secondary_re_indexed = None

        if (task.secondary_table is not None) and (
                task.annotations['features_to_use'] is not None):
            if (len(np.intersect1d(task.secondary_table.columns,
                                   task.annotations['features_to_use'])) > 0)\
                    or (task.annotations['features_to_use'] == 'all'):

                found_correspondence = False

                for key in task.annotations['secondary_to_primary_link'].keys():
                    value = task.annotations['secondary_to_primary_link'][key]
                    if (value == [task.primary_table_target_name,
                                  task.primary_table_alternatives_names] or
                            value == [task.primary_table_alternatives_names,
                                      task.primary_table_target_name] or
                            value == task.primary_table_alternatives_names or
                            value == task.primary_table_target_name):

                        secondary_re_indexed = task.secondary_table.set_index(key)
                        found_correspondence = True

                    else:
                        if input_merge_columns is None:
                            input_merge_columns = [key]
                        else:
                            input_merge_columns += [key]

                if not found_correspondence:
                    raise Exception("key linking to alternatives not provided")

        if isinstance(task, PairwiseComparisonTask) and (
                self.pairwise_red_args == {}):

            _re_indexed_df = task.primary_table.set_index([
                task.target_column_correspondence,
                task.inverse_correspondence_column[0]
            ])

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

    # def predict_task_packer(self):
    #     pass


class GLMPairwiseComparisonModel(PairwiseComparisonModel):
    def __init__(self):
        super(GLMPairwiseComparisonModel, self).__init__()


class SVMPairwiseComparisonModel(PairwiseComparisonModel):
    def __init__(self):
        super(SVMPairwiseComparisonModel, self).__init__(
            {'style': 'reciprocal'}
        )
