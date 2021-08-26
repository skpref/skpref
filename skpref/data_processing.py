from abc import ABC
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
import random
from typing import Union

# Poset Types


class PosetType(ABC):
    """
    Abstract class for PosetType Used by models and data containers to query
    what kind of Poset is being fed in.
    Parameters:
    ------------
    entity_universe_type: str
        "label": all potential entities in the universe are known
        "object": an entity is observed at most once
        "semi": "object" except entities can reoccur
    """
    def __init__(self, entity_universe_type='label'):
        self.entity_universe_type = entity_universe_type

    def get_params(self):
        pass

    def set_params(self):
        pass


class OrderPosetType(PosetType):
    """
    Class that represents fully ordered Posets for example, the ranks of
    competitors in a swimming competition.
    Parameters:
    -----------
    size_const: Boolean, default=False
        True when there is always the same number of entities being ranked

    size: int, default=None
        If size_const is True then the number of entities being ranked
    """
    def __init__(self, entity_universe_type='label', size_const=False, size=None):
        super(OrderPosetType, self).__init__(entity_universe_type)
        self.size_const = size_const
        self.size = size


class BagPosetType(PosetType):
    """
    Class that represents a vector of bags
    """
    pass


class SubsetPosetType(PosetType):
    """
    Choice poset, where there is always one chosen from the alternatives
    presented.

    Parameters:
    ------------
    top_size_const: Boolean, default=False
        True when there is always the same amount of entities chosen, False
        otherwise

    boot_size_const: Boolean, default=False
        True when the size of the not chosen entities is always the same, False
        otherwise

    top_size: int, default=None
        If top_size_const is True then the number of entities usually chosen

    boot_size: int, default=None
        If boot_size_const is True then the number entities usually not chosen
    """
    def __init__(self, entity_universe_type='label', top_size_const=False,
                 boot_size_const=False, top_size=None, boot_size=None):

        super(SubsetPosetType, self).__init__(entity_universe_type)
        self.top_size_const = top_size_const
        self.boot_size_const = boot_size_const
        self.top_size = top_size
        self.boot_size = boot_size


# Poset Vectors


class PosetVector(ABC):
    """
    Represents samples of posets in a unified fashion. Internally special kinds
    of posets are stored efficiently. Special posets inherit from this abstract
    type

    Parameters:
    -----------
    entity_universe: numpy array
        The distinct entities that have been presented

    poset_type: PosetType
        The PosetType specified in the class above

    dims: int
        The length of the poset vector

    efficient_representation: scipy sparse matrix
        Compressed representation of the poset
    """
    def __init__(self, entity_universe, poset_type, dims,
                 efficient_representation):
        self.entity_universe = entity_universe
        self.poset_type = poset_type
        self.dims = dims
        self.efficient_representation = efficient_representation


class OrderedPosetVec(PosetVector):

    pass


def sparsify_list_of_entities(entity_universe, data):
    """
    Creates a sparse matrix of a list of choice arrays that contain entities

    Parameters:
    -----------
    entity_universe: nunmpy array
        The distinct entities that have been presented

    data: numpy array
        The input choice data to convert into a sparse matrix
    """
    entity_lkp = {}
    for ent_num, ent in enumerate(entity_universe):
        entity_lkp[ent] = ent_num

    entities = len(entity_universe)

    top_sparse_list = []
    for row in data:
        _append_top_sparse = np.zeros(entities).astype(int)

        if type(data[0]) is np.ndarray:
            for ent in row:
                _append_top_sparse[entity_lkp[ent]] = 1
        else:
            _append_top_sparse[entity_lkp[row]] = 1

        top_sparse_list.append(_append_top_sparse)

    return csr_matrix(top_sparse_list), entity_lkp


class SubsetPosetVec(PosetVector):
    """
    Poset Vector class representation of vectors where a subset of items has
    been chosen from many.

    Parameters:
    -----------
    top_input_data: numpy array
        An array that contains arrays which contain choices made

    boot_input_data: numpy array
        An array that contains arrays which contain the entities not chosen

    subset_type_vars: dict
        An option for the user to specify variables which are related to the
        PosetType such as {'boot_size_const': True, 'boot_size_length': 5}
    """
    def __init__(self, top_input_data: np.array, boot_input_data: np.array,
                 subset_type_vars: dict = None):
        self.top_input_data = top_input_data
        self.boot_input_data = boot_input_data
        self.pairwise_comparison_reduction_dict = defaultdict()

        dat_for_unique = np.array([])

        for in_dat in [top_input_data, boot_input_data]:
            if type(in_dat[0]) is np.ndarray:
                dat_for_unique = np.append(dat_for_unique,
                                           np.concatenate(in_dat))
            else:
                dat_for_unique = np.append(dat_for_unique, in_dat)

        entity_universe = np.unique(dat_for_unique)

        if subset_type_vars is None:
            poset_type = SubsetPosetType('list')
        else:
            poset_type = SubsetPosetType(**subset_type_vars)

        if len(top_input_data) != len(boot_input_data):
            raise Exception("Choice data mismatch! There seem to be an unequal"
                            "length of lists of entities chosen and not chosen")
        else:
            dims = len(top_input_data)

        # We will improve this just a placeholder to check if classes are working
        # correctly

        efficient_top, self.lkp = sparsify_list_of_entities(entity_universe,
                                                            self.top_input_data)

        efficient_boot, _ = sparsify_list_of_entities(entity_universe,
                                                      self.boot_input_data)

        efficient_representation = [efficient_top, efficient_boot]

        super(SubsetPosetVec, self).__init__(entity_universe, poset_type, dims,
                                             efficient_representation)

    def pairwise_reducer(self, style: str = "positive",
                         rejection: Union[int, float] = 0, scramble: bool = True,
                         random_seed_scramble: int = None,
                         target_colname='alt1_top') -> [pd.DataFrame, np.array]:
        """
        Breaks a SubsetPosetVec into the most elementary parts of a pairwise
        comparison.
        Inputs
        -------
        style: "positive" or "reciprocal", default="positive"
            When positive top=[A], boot=[B,C] will be created as the following
            comparisons: [A,B], [A,C]
            When reciprocal the same will be created as [A,B], [A,C], [B,A],
            [C,A]

        rejection: int or float, default=0
            The response variable to give for the objects not chosen, sometimes
            -1 might be used for SVM rather than 0 for linear models.

        scramble: boolean, default=True
            In the case of a positive return, this will scramble the entities
            in the left right column so that the target output isn't a vector
            of ones, which later on throws errors in scikit-learn gridsearch

        random_seed_scramble: int, default=None
            The random seed to use for scrambling

        Returns
        --------
        Pairwise DataFrame and observations indexing numpy array
        """

        if style == "positive" and not scramble:
            divisor = 3
            columns = ['observation', 'alt1', 'alt2']

        elif scramble or style == "reciprocal":
            divisor = 4
            columns = ['observation', 'alt1', 'alt2', target_colname]

        else:
            raise NameError("style can only be positive or "
                            "reciprocal")

        if scramble:
            random.seed(random_seed_scramble)

        if (style, rejection, scramble) in \
                self.pairwise_comparison_reduction_dict.keys():
            return self.pairwise_comparison_reduction_dict[style, rejection,
                                                           scramble]

        else:
            alt_type = type(self.top_input_data[0][0])
            pairwise_comparisons = np.array(
                [], dtype=alt_type)

            for i, choice in enumerate(self.top_input_data):
                for j in choice:
                    for k in self.boot_input_data[i]:
                        if j == k:
                            continue
                        if style == "positive" and not scramble:
                            observation = np.array([i, j, k])
                        elif style == "positive" and scramble:
                            if random.randint(0, 1) == 0:
                                observation = np.array([i, j, k, 1])
                            else:
                                observation = np.array([i, k, j, rejection])
                        elif style == "reciprocal":
                            observation = np.array([i, j, k, 1, i, k, j,
                                                    rejection])
                        else:
                            raise NameError("style can only be positive or "
                                            "reciprocal")

                        pairwise_comparisons = np.append(pairwise_comparisons,
                                                         observation)
            pairwise_comps = len(pairwise_comparisons)

            pairwise_comparison_reduction = pd.DataFrame(
                pairwise_comparisons.reshape(int(pairwise_comps / divisor),
                                             divisor), columns=columns
            )

            if alt_type == np.str_:
                if style == 'positive' and not scramble:
                    pairwise_comparison_reduction = \
                        pairwise_comparison_reduction.astype(
                            {'observation': int})

                if style == 'reciprocal':
                    pairwise_comparison_reduction = \
                        pairwise_comparison_reduction.astype(
                            {'observation': int, target_colname: int})

            observations = \
                pairwise_comparison_reduction.observation.astype(int).values

            pairwise_comparison_reduction.drop(
                'observation', axis=1, inplace=True)

            self.pairwise_comparison_reduction_dict[style, rejection, scramble] = \
                (pairwise_comparison_reduction, observations)

            return pairwise_comparison_reduction, observations

    def classifier_reducer(self, rejection: int = 0, chosen_name: str = 'chosen'
                           ) -> [pd.DataFrame, np.array]:
        """
        Reduces observations to classifiers for example when the data looks like
        this top = [A], boot = [B, C] it will return the following:
        pd.DataFrame({
        option: [A, B, C],
        chosen: [1, 0, 0]
        })
        Inputs:
        --------
        rejection: int, default=0
            controls what value gets assigned to rejections, -1 might be
            preferred for SVM
        chosen_name: str, default='chosen'
            creates the name for the column that corresponds with 1/0 to whether
            the product was chosen or not.
        Outputs:
        --------
        DataFrame
        """

        obs = []
        choice = []

        for top_observation, item in enumerate(self.top_input_data):
            obs.append(np.ones(len(item) +
                               len(self.boot_input_data[top_observation]),
                               dtype=int) *
                       top_observation
                       )
            choice.append(np.append(
                np.ones(len(item), dtype=int),
                np.ones(len(self.boot_input_data[top_observation]), dtype=int) *
                rejection
            ))

        obs_in = np.hstack(obs)
        alts_in = np.hstack(np.hstack(np.hstack(
            np.dstack((self.top_input_data.reshape(-1,1), self.boot_input_data.reshape(-1,1)))
        )))
        choice_in = np.hstack(choice)

        return (pd.DataFrame({
            'observation': obs_in,
            'alternative': alts_in,
            chosen_name: choice_in,
        }).drop_duplicates(subset=['observation', 'alternative'])\
            .drop(['observation'], axis=1).reset_index(drop=True),
            obs_in, alts_in)
