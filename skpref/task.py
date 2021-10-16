from abc import ABC
import pandas as pd
from scipy.io import arff
from typing import List, Tuple, Union
import numpy as np
import os
from skpref.data_processing import SubsetPosetType, SubsetPosetVec, PosetType


# class UnsupportedListError(Exception):
#     error_msg = ("Currently a list of columns that correspond to either the "
#                  "target variable or the alternatives is not supported. "
#                  "Please provide all choices in one column for example: "
#                  "[Alternative1, Alternative2, Alternative3]")
#
#     def __init__(self):
#         Exception.__init__(self, self.error_msg)


class PrefTaskType(ABC):
    """
    Abstract class for preference type classes.
    Parameters:
    -----------
    entity_slot_type: PosetType
        type of the entity slot ordering/exchangeability
    entity_universe_type: str
        "label": all potential entities in the universe are known
        "object": an entity is observed at most once
        "semi": "object" except entities can reoccur
    target_type: PosetType
        type of the desired preference target
    proba: Boolean, default=False
        True  is probabilistic output is required
    """
    def __init__(self, entity_slot_type: PosetType, entity_universe_type: str,
                 target_type: PosetType, proba: bool):
        self.entity_slot_type = entity_slot_type
        self.entity_universe_type = entity_universe_type
        self.target_type = target_type
        self.proba = proba


class ChoiceTaskType(PrefTaskType):
    """
    Class type for choice based data
    """
    def __init__(self, entity_slot_type_kwargs, target_type_kwargs):
        if entity_slot_type_kwargs is None:
            entity_slot_type_kwargs = {}
        if target_type_kwargs is None:
            target_type_kwargs = {}
        super(ChoiceTaskType, self).__init__(
            entity_slot_type=SubsetPosetType(**entity_slot_type_kwargs),
            entity_universe_type='label',
            target_type=SubsetPosetType(**target_type_kwargs),
            proba=False
        )


class PrefTask(ABC):
    """
    Abstract class for preference tasks
    Parameters:
    -----------
    pref_task_type: PrefTaskType
        The task type of what to learn
    data_hook: str
        The directory where the data is
    annotations: dict
        {primary_table_name: the name of the table that contains the ground truth
         primary_table_alternatives_name: field that contains the alternatives
            in the primary table.
         primary_table_target_names: field that contains the target variable.
         secondary_table_name: the name of the secondary table
        secondary_to_primary_link: columns by which the two tables link together
        }
    """

    def __init__(self, pref_task_type: PrefTaskType, primary_table: pd.DataFrame,
                 primary_table_alternatives_names: Union[List[str], str],
                 primary_table_target_name: str = None,
                 secondary_table: pd.DataFrame = None,
                 features_to_use: Union[List[str], None] = 'all',
                 secondary_to_primary_link: dict = None):

        self.pref_task_type = pref_task_type

        # Read in primary table
        self.primary_table, prim_name, prim_hook = \
            _table_reader(primary_table)

        self.data_hook = prim_hook

        if type(features_to_use) == list:
            self.features_to_use = features_to_use
        elif features_to_use is not None and features_to_use != 'all':
            self.features_to_use = [features_to_use]
        else:
            self.features_to_use = features_to_use

        if features_to_use is not None and features_to_use != 'all':
            self.primary_table_features_to_use = np.intersect1d(
                features_to_use, self.primary_table.columns)

        elif features_to_use == 'all':
            if isinstance(primary_table_alternatives_names, list):
                if primary_table_target_name is None:
                    first_comp_element = primary_table_alternatives_names

                else:
                    first_comp_element = primary_table_alternatives_names + \
                                         [primary_table_target_name]
            elif primary_table_target_name is None:
                first_comp_element = [primary_table_alternatives_names]
            else:
                first_comp_element = [primary_table_alternatives_names,
                                      primary_table_target_name]

            self.primary_table_features_to_use = np.setdiff1d(
                first_comp_element, self.primary_table.columns)

        else:
            self.primary_table_features_to_use = []

        # Read in secondary table
        self.secondary_to_primary_link = secondary_to_primary_link
        if secondary_table is not None:
            self.secondary_table, sec_name, sec_hook = \
                _table_reader(secondary_table)

            if features_to_use is not None and features_to_use != 'all':
                self.secondary_table_features_to_use = np.intersect1d(
                    features_to_use, self.secondary_table.columns)

            elif features_to_use == 'all':
                self.secondary_table_features_to_use = np.setdiff1d(
                    self.secondary_table.columns,
                    [primary_table_alternatives_names]
                )

        else:
            self.secondary_table = None
            self.secondary_table_features_to_use = []

        self.primary_table_alternatives_names = primary_table_alternatives_names
        self.primary_table_target_name = primary_table_target_name

        self.annotations = {
            'primary_table_name': prim_name,
            'primary_table_alternatives_names':
                self.primary_table_alternatives_names,
            'primary_table_target_names': self.primary_table_target_name,
            'secondary_table_name': secondary_table,
            'secondary_to_primary_link': secondary_to_primary_link,
            'features_to_use': features_to_use
        }

        if features_to_use is not None and len(
                self.secondary_table_features_to_use) == 0 and len(
            self.primary_table_features_to_use) == 0:
            raise Exception("Columns in features_to_use could not be found in"
                            " any of the tables, please check spelling. Features"
                            " to use is automatically set to 'all', if there are"
                            " no features in the data please set the "
                            "features_to_use parameter to None")

    def find_merge_columns(self, original_naming: bool = True
                           ) -> Tuple[pd.DataFrame, List[str], np.array,
                                      np.array]:
        """
        Given some annotations expresses the column names on which data should
        be merged in a pandas merge, it also indexes the secondary table by the
        name of the alternatives

        Returns
        -------
        DataFrame:
            Re-indexed secondary table
        List[str]:
            Column names on which to do inner join
        List[str]
        """
        input_merge_columns = []
        secondary_re_indexed = pd.DataFrame()
        left_on = []
        right_on = []

        if len(self.secondary_table_features_to_use) > 0:
            found_correspondence = False

            for key in self.annotations['secondary_to_primary_link'].keys():
                value = self.annotations['secondary_to_primary_link'][key]
                if (value == [self.primary_table_target_name,
                              self.primary_table_alternatives_names] or
                        value == [self.primary_table_alternatives_names,
                                  self.primary_table_target_name] or
                        value == self.primary_table_alternatives_names or
                        value == self.primary_table_target_name):

                    secondary_re_indexed = self.secondary_table.set_index(key).copy()
                    found_correspondence = True
                    if original_naming:
                        if type(value) == list:
                            left_on += value
                        else:
                            left_on.append(value)
                    else:
                        left_on.append('alternative')
                    right_on.append(key)

                else:

                    left_on.append(value)
                    right_on.append(key)
                    input_merge_columns.append(key)

            if not found_correspondence:
                raise Exception("key linking to alternatives not provided")

        return secondary_re_indexed, input_merge_columns, \
               np.array(left_on).flatten(), np.array(right_on).flatten()


def _table_reader(table):
    """
    Reads in the table for a ChoiceTask
    """
    if type(table) is str:

        table_format = table.split(".")[-1]

        if "\\" in table:
            loc_split = table.split('\\')
            table_name = loc_split[-1]
            hook = '\\'.join(loc_split[:-1])+'\\'
        elif '/' in table:
            loc_split = table.split('/')
            table_name = loc_split[-1]
            hook = '/'.join(loc_split[:-1]) + '/'
        else:
            table_name = table
            hook = os.getcwd()

        if table_format == 'csv':
            _return_table = pd.read_csv(table)

        elif table_format == 'arff':
            _return_table_arff, _ = arff.loadarff(table)
            _return_table = pd.DataFrame(_return_table_arff)

        else:
            raise Exception("Currently only supporting csv and arff formats")

    elif type(table) is pd.DataFrame:
        _return_table = table.copy()
        table_name = 'global_data_frame'
        hook = 'globals'

    elif type(table) is np.ndarray:
        if table.dtype.names is None:
            raise Exception("Please set the column names of the data in "
                            "dtype.names for the array")

        else:
            _return_table = pd.DataFrame(table)
            table_name = 'gobal_arff_table'
            hook = 'globals'

    else:
        raise Exception("Currently only supporting csv and structured numpy array"
                        " formats")

    return _return_table, table_name, hook


def _convert_array_of_lists_to_array_of_arrays(ar):
    """
    Converts an array of lists into an array of arrays
    Inputs:
    -------
    ar : ndarray,
        The array to inspect
    """
    if (type(ar[0]) is list) and (type(ar) is np.ndarray):
        return np.array([np.array(x) for x in ar], dtype=object)
    else:
        return ar


class ChoiceTask(PrefTask):
    """
    Task for choice based models
    Parameters:
    -----------
    primary_table: DataFrame
        The primary table is the one that contains the target variable and
        covariates that vary on the target observation level. For example the
        available methods of transportation for an individual and weather it
        rained or not at the time of the journey.
    primary_table_alternatives_names: str
        The column or attribute which corresponds to the alternatives in the
        primary table.
    primary_table_target_name: str, default=None
        The column name or attribute which corresponds to the ground truth
    secondary_table: DataFrame, default=None
        The secondary table that usually contains information about the
        alternatives in the primary table. For example the cleanliness perception
        of public transportation.
    secondary_to_primary_link: dict, default:None
        How to link the primary and secondary tables together. The key in the
        dictionary will correspond to the field in the secondary table and the
        value for each key will be the field in the primary table
    entity_slot_type_kwargs: dict of PosetType args
        arguments to tell about the PosetType of the entity slot type
    target_type_kwargs: dict of PosetType args
        arguments to tell about the PosetType of the entity slot type
    features_to_use: list of strings, default = 'all'
        Column names of the features to use, by default the task will try to use
        every column as features. If the user wants to use a model that doesn't
        use any features then it should be set to None
    """
    def __init__(self, primary_table: pd.DataFrame,
                 primary_table_alternatives_names: Union[List[str], str],
                 primary_table_target_name: str = None,
                 secondary_table: pd.DataFrame = None,
                 secondary_to_primary_link: dict = None,
                 entity_slot_type_kwargs: dict = None,
                 target_type_kwargs: dict = None,
                 features_to_use: Union[List[str], str, None] = 'all') -> None:

        super(ChoiceTask, self).__init__(
            pref_task_type=ChoiceTaskType(entity_slot_type_kwargs,
                                          target_type_kwargs),
            primary_table=primary_table,
            primary_table_alternatives_names=primary_table_alternatives_names,
            primary_table_target_name=primary_table_target_name,
            secondary_table=secondary_table,
            secondary_to_primary_link=secondary_to_primary_link,
            features_to_use=features_to_use
        )

        if not hasattr(self, 'top') and self.primary_table_target_name is not None:
            if type(self.primary_table_target_name) is list:
                self.top = self.primary_table[self.primary_table_target_name]\
                    .copy().values
            else:
                self.top = self.primary_table[self.primary_table_target_name]\
                    .copy().values

                # If the top choice is always 1 then the elements of top under this
                # condition come as the element not the element inside a list which
                # is the general format we'd like to follow.
                if type(self.top[0]) not in [np.ndarray, list]:
                    self.top = self.top.reshape(len(self.top), 1)

            self.top = _convert_array_of_lists_to_array_of_arrays(self.top)

        elif not hasattr(self, 'top') and self.primary_table_target_name is None:
            self.top = None

        if type(self.primary_table_alternatives_names) is list:
            alts = self.primary_table[self.primary_table_alternatives_names].copy()\
                .values
        else:
            alts = self.primary_table[self.primary_table_alternatives_names]\
                .copy().values

        alts = _convert_array_of_lists_to_array_of_arrays(alts)

        if self.top is not None:
            joint_alts = np.array([[self.top[i], alts[i]]
                                   for i in range(len(alts))], dtype=object)
            boot = np.array([np.setdiff1d(x[1], x[0], assume_unique=True)
                             for x in joint_alts], dtype=object)
        else:
            boot = alts
            self.top = boot

        self.subset_vec = SubsetPosetVec(
            self.top,
            boot,
            subset_type_vars=entity_slot_type_kwargs
        )

        # Temporary Line to be removed once PairwiseModel() gets developed
        if not hasattr(self, 'target_column_correspondence'):
            self.target_column_correspondence = None


class PairwiseComparisonTask(ChoiceTask):
    """
        Task for choice based models
        Parameters:
        -----------
        primary_table: str, DataFrame, scipy.io.arff
            The primary table is the one that contains the target variable and
            covariates that vary on the target observation level. For example the
            available methods of transportation for an individual and weather it
            rained or not at the time of the journey.
            If str it will be the directory where the primary table sits. Otherwise
            will also read in pandas DataFrames and scipy.io.arff
        primary_table_alternatives_names: str
            The column or attribute which corresponds to the alternatives in the
            primary table.
        primary_table_target_name: str
            The column name or attribute which corresponds to the ground truth
        secondary_table: str, DataFrame, scipy.io.arff, default=None
            The secondary table that usually contains information about the
            alternatives in the primary table. For example the cleanliness perception
            of public transportation.
            If str it will be the directory where the primary table sits. Otherwise
            will also read in pandas DataFrames and scipy.io.arff
        secondary_to_primary_link: dict, default:None
            How to link the primary and secondary tables together. The key in the
            dictionary will correspond to the field in the secondary table and the
            value for each key will be the field in the primary table
        target_type_kwargs: dict of PosetType args
            arguments to tell about the PosetType of the entity slot type
        features_to_use: list of strings, default = 'all'
            Column names of the features to use, by default the task will try to use
            every column as features. If the user wants to use a model that doesn't
            use any features then it should be set to None
        target_column_correspondence: str, default=None
            If the choice is a pairwise comparison and the target is not the name
            of an entity but a {1,0} variable that corresponds to one of the entities
            being chosen in one of the columns, then this should be the name of the
            column for which when the target variable is 1 then that column's entity
            has been chosen. i.e. there is a column with home team another one with
            away team and target is 1 when home team wins.
        """
    def __init__(self, primary_table: pd.DataFrame,
                 primary_table_alternatives_names: List[str],
                 primary_table_target_name: str = None,
                 target_column_correspondence: str = None,
                 secondary_table: pd.DataFrame = None,
                 secondary_to_primary_link: dict = None,
                 target_type_kwargs: dict = None,
                 features_to_use: Union[str, List[str], None] = 'all') -> None:

        self.target_column_correspondence = target_column_correspondence

        if self.target_column_correspondence is not None \
                and primary_table_target_name is not None:
            self.inverse_correspondence_column = \
                primary_table_alternatives_names.copy()
            self.inverse_correspondence_column.remove(
                self.target_column_correspondence)
            self.top = np.where(
                primary_table[primary_table_target_name] == 1,
                primary_table[target_column_correspondence],
                primary_table[self.inverse_correspondence_column[0]]
            ).reshape(len(primary_table), 1)

        else:
            self.top = None

        entity_slot_type_kwargs = {
            'top_size_const': True,
            'top_size': 1,
            'boot_size_const': True,
            'boot_size': 1
        }

        super(PairwiseComparisonTask, self).__init__(
            primary_table, primary_table_alternatives_names,
            primary_table_target_name, secondary_table,
            secondary_to_primary_link, entity_slot_type_kwargs,
            target_type_kwargs, features_to_use
        )
