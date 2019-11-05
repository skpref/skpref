from abc import ABC
import numpy as np
from scipy.sparse import csr_matrix

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
    def __init__(self, entity_universe_type='list'):
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
    def __init__(self, entity_universe_type='list', size_const=False, size=None):
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
    def __init__(self, entity_universe_type='list', top_size_const=False,
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
    def __init__(self, top_input_data, boot_input_data, subset_type_vars=None):
        self.top_input_data = top_input_data
        self.boot_input_data = boot_input_data

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


