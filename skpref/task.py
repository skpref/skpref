from abc import ABC
import pandas as pd
from scipy.io import arff
import numpy as np
import os
from skpref.data_processing import SubsetPosetType, SubsetPosetVec


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
    def __init__(self, entity_slot_type, entity_universe_type, target_type,
                 proba):
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

    def __init__(self, pref_task_type, data_hook, annotations):
        self.pref_task_type = pref_task_type
        self.data_hook = data_hook
        self.annotations = annotations


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
        return np.array([np.array(x) for x in ar])
    else:
        return ar


class ChoiceTask(PrefTask):
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
        dictionary will correspond to the field in the primary table and the
        value for each key will be the field in the secondary table
    entity_slot_type_kwargs: dict of PosetType args
        arguments to tell about the PosetType of the entity slot type
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
    def __init__(self, primary_table, primary_table_alternatives_names,
                 primary_table_target_name, secondary_table=None,
                 secondary_to_primary_link=None, entity_slot_type_kwargs=None,
                 target_type_kwargs=None, features_to_use='all',
                 target_column_correspondence=None):

        # Read in primary table
        self.primary_table, prim_name, prim_hook =\
            _table_reader(primary_table)

        # Read in secondary table
        if secondary_table is not None:
            self.secondary_table, sec_name, sec_hook = \
                _table_reader(secondary_table)

            self.secondary_to_primary_link = secondary_to_primary_link

        self.primary_table_alternatives_names = primary_table_alternatives_names
        self.primary_table_target_name = primary_table_target_name
        self.target_column_correspondence = target_column_correspondence

        annotations = {
            'primary_table_name': prim_name,
            'primary_table_alternatives_names':
                self.primary_table_alternatives_names,
            'primary_table_target_names': self.primary_table_target_name,
            'secondary_table_name': secondary_table,
            'secondary_to_primary_link': secondary_to_primary_link,
            'features_to_use': features_to_use
        }

        super(ChoiceTask, self).__init__(
            pref_task_type=ChoiceTaskType(entity_slot_type_kwargs,
                                          target_type_kwargs),
            data_hook=prim_hook,
            annotations=annotations
        )

        if self.target_column_correspondence is not None:
            self.inverse_correspondence_column = \
                self.primary_table_alternatives_names.copy()
            self.inverse_correspondence_column.remove(self.target_column_correspondence)
            top = np.where(
                self.primary_table[self.primary_table_target_name] == 1,
                self.primary_table[self.target_column_correspondence],
                self.primary_table[self.inverse_correspondence_column[0]]
            ).reshape(len(self.primary_table), 1)
        elif type(self.primary_table_target_name) is list:
            top = self.primary_table[self.primary_table_target_name]\
                .copy().values
        else:
            top = self.primary_table[self.primary_table_target_name]\
                .copy().values

            # If the top choice is always 1 then the elements of top under this
            # condition come as the element not the element inside a list which
            # is the general format we'd like to follow.
            if type(top[0]) not in [np.ndarray, list]:
                top = top.reshape(len(top), 1)

        if type(self.primary_table_alternatives_names) is list:
            alts = self.primary_table[self.primary_table_alternatives_names].copy()\
                .values
        else:
            alts = self.primary_table[self.primary_table_alternatives_names]\
                .copy().values

        alts = _convert_array_of_lists_to_array_of_arrays(alts)
        top = _convert_array_of_lists_to_array_of_arrays(top)

        joint_alts = np.array([[top[i], alts[i]] for i in range(len(alts))])
        boot = np.array([np.setdiff1d(x[1], x[0]) for x in joint_alts])

        self.subset_vec = SubsetPosetVec(
            top,
            boot,
        )
