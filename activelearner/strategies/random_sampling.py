'''
Random Sampling
'''
from activelearner.interfaces import Query
from activelearners.utils import inherit_docstring_from, seed_random_state


class RandomSampling(Query):
    r"""
    Random sampling
    
    This class implements the random sample query strategy. A random entry from the
    unlabeledd pool is returned for each query.
    
    Parameters
    ----------
    random_state: {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. If np.random.RandomState instance,
        random_state is the random number generator. 
        
    Attributes
    ----------
    random_states\_: np.random.RandomState instance
        The random number generator being used.
        
    Examples
    --------
    Here is an example of declaring a RandomSampling Query object:
    
    .. code-block:: python
    
        from active_learner.query import RandomSampling
        
        qs = RandomSampling(dataset,)
    """
    
    def __init__(self, dataset, **kwargs):
        super(RandomSampling, self).__init__(dataset, **kwargs)
        
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)
        
    @inherit_docstring_from(Query)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()
        ask_id = self.random_state_.randint(0, len(unlabeled_entry_ids))
        return unlabeled_entry_ids[ask_id]
        
