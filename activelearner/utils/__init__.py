import sys
import pickle
import numpy as np


__all__ = ['inherit_docstring_from', 'seed_random_state', 'calc_cost', 'save_object',
           'update_progress', 'ceildiv']
      
        
# Decorator function for classes to inherit docstrings from cls
def inherit_docstring_from(cls):
    """
    Decorator for class methods to inherit docstring from :code:`cls`
    """
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


# Function to instantiate random state
def seed_random_state(seed):
    """
    Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to generate numpy.random.RandomState"
                     "instance" % seed)


# Function to calculate cost/loss for label preditions
def calc_cost(y, yhat, cost_matrix):
    """
    Calculate the cost with given cost matrix

    y: ground truth

    yhat: prediction

    cost_matrix: array-like, shape=(n_classes, n_classes)
        The ith row, jth column represents the cost of the ground truth being
        ith class and prediction as jth class
    """
    return np.mean(cost_matrix[list(y), list(yhat)])


# Function to save (pickle) python objects
def save_object(obj, filename):
    '''
    Saves python objects to specified filename. Will
    overwrite file

    Arguments
    ---------
    obj: python object

    filename: file path + name
    '''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# function for performing ceiling division without importing math module
def ceildiv(a, b):
    return -(-a // b)
