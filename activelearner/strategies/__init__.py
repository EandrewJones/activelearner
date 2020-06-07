"""
Concrete query strategy classes
"""
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .uncertainty_sampling import UncertaintySampling

__all__ = [
    'QueryByCommittee',
    'QUIRE',
    'RandomSampling',
    'UncertaintySampling'
    ]