from abc import ABC

# Poset Types


class PosetType(ABC):
    """
    Abstract class for PosetType Used by models and data containers to query
    what kind of Poset is being fed in.
    """
    pass


class OrderPosetType(PosetType):
    """
    Class that represents fully ordered Posets for example, the ranks of
    competitors in a swimming competition.
    """
    pass


class BagPosetType(PosetType):
    """
    Class that repsresents relationless posets.
    """
    pass


class SubsetPosetType(PosetType):
    pass

# Poset Vectors


class PosetVector(ABC):
    pass


class OrderedPosetVec(PosetVector):
    pass
