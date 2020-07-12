"""
Top-level package for ActiveLearner.

Available submodules
--------------------

dataset
    Defines dataset extensions
interfaces
    Defines interfaces and dataset objects
labeler
    Interface for labeling data
models
    Machine learning models for predicting labels
strategies
    Active learning query strategies
"""

__author__ = """Evan Andrew Jones"""
__email__ = 'evan.a.jones3@gmail.com'
__version__ = '0.1.1'

import activelearner.dataset
import activelearner.interfaces
import activelearner.labeler
import activelearner.models
import activelearner.strategies
import activelearner

__all__ = [
    'dataset',
    'interfaces',
    'labeler',
    'models',
    'strategies',
    'activelearner'
    ]

