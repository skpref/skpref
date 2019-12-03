from abc import ABC
import pandas as pd
from scipy.io import arff
import numpy as np
import os
from data_processing import SubsetPosetType, SubsetPosetVec


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


def _table_reader(table, meta, format):
    """
    Reads in the table for a ChoiceTask
    """
    if type(table) is str:

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

        if format == 'csv':
            _return_table = pd.read_csv(table)
            _return_metadata = None

        elif format == 'arff':
            _return_table, _return_metadata = arff.loadarff(table)

        else:
            raise Exception("Currently only supporting csv and arff formats")

    elif type(table) is pd.DataFrame:
        _return_table = table.copy()
        _return_metadata = None
        table_name = [x for x in globals() if globals()[x] is table][0]
        hook = 'globals'

    elif type(table) is np.ndarray:
        if meta is None:
            raise Exception("Please provide metadata alongside the data"
                            " values")
        else:
            _return_table = table.copy()
            _return_metadata = meta.copy()
            table_name = [x for x in globals() if globals()[x] is table][0]
            hook = 'globals'

    else:
        raise Exception("Parameters given don't fit into any of the file read"
                        "in cases")

    return _return_table, _return_metadata, table_name, hook


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
    primary_table_alternatives_names: str or list of str
        The column or attribute which corresponds to the alternatives in the
        primary table.
    primary_table_target_name: str or list of str
        The column name or attribute which corresponds to the ground truth
    primary_table_metadata: scipy.arff.metadata, default=None
        The metadata field in a scipy arff load
    primary_table_type: str, possible values {'csv', 'arff'}, default='csv'
        If the primary table is a string then this defines what type of file is
        being read in. Code will use pandas to read csv and scipy to read arff
    secondary_table: str, DataFrame, scipy.io.arff, default=None
        The secondary table that usually contains information about the
        alternatives in the primary table. For example the cleanliness perception
        of public transportation.
        If str it will be the directory where the primary table sits. Otherwise
        will also read in pandas DataFrames and scipy.io.arff
    secondary_table_metadata: scipy.arff.metadata, default=None
        The metadata field in a scipy arff load
    secondary_table_type: str, possible values {'csv', 'arff'}, default='csv'
        If the secondary table is a string then this defines what type of file
        is being read in. Code will use pandas to read csv and scipy to read
        arff
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
    """
    def __init__(self, primary_table, primary_table_alternatives_names,
                 primary_table_target_name, primary_table_metadata=None,
                 primary_table_type='csv', secondary_table=None,
                 secondary_table_metadata=None,
                 secondary_table_type='csv',
                 secondary_to_primary_link=None, entity_slot_type_kwargs=None,
                 target_type_kwargs=None, features_to_use='all'):

        # Read in primary table
        self.primary_table, self.primary_table_metadata, prim_name, prim_hook =\
            _table_reader(primary_table, primary_table_metadata,
                          primary_table_type)

        # Read in secondary table
        if secondary_table is not None:
            self.secondary_table, self.secondary_table_metadata, sec_name,\
            sec_hook = _table_reader(secondary_table, secondary_table_metadata,
                              secondary_table_type)

            self.secondary_to_primary_link = secondary_to_primary_link

        self.primary_table_alternatives_names = primary_table_alternatives_names
        self.primary_table_target_name = primary_table_target_name

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

        if type(self.primary_table_target_name) is list:
            top = self.primary_table[self.primary_table_target_name]\
                .copy().values
        else:
            top = self.primary_table[[self.primary_table_target_name]]\
                .copy().values

        if type(self.primary_table_alternatives_names) is list:
            alts = self.primary_table[self.primary_table_alternatives_names].copy()\
                .values
        else:
            alts = self.primary_table[[self.primary_table_alternatives_names]]\
                .copy().values

        alts_size = alts.shape[1]
        top_size = top.shape[1]

        if alts_size > 1:
            boot = alts[np.where(alts != top)].reshape(top.shape[0],
                                                       alts_size-top_size)
        else:
            raise Exception("heterogeneous alternative presentation or "
                            "heterogeneous choices currently not developed")

        self.subset_vec = SubsetPosetVec(
            top,
            boot,
        )
