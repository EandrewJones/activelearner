'''
Uncertainty Sampling

This module contains a class that implements two of the most well-known 
uncertainty sampling query strategies: the least confidence method and
the smallest margin method (margin sampling).
'''
import numpy as np

from activelearner.interfaces import Query, ContinuousModel, ProbabilisticModel
from activelearner.utils import inherit_docstring_from


class UncertaintySampling(Query):
    r"""
    Uncertainty Sampling
    
    This class implements Uncertainty Sampling active learning algorithms [1]_. 
    
    
    Parameters
    ----------
    model: :py:class:`ContinuousModel` or :py:class:`ProbabilisticModel` object instance.
        The base model used for training
        
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        least confidence (lc), queries instance whose posterior probability of being postive
            is nearest 0.5 (for binary classification);
        smallest margin (sm), it queries the instance whose posterior probability gap between the 
            most and second most probable labels is minimal;
        entropy, requires :py:class:`ProbabilisticModel` to be passed in as model parameter;
        
        
    Attributes
    ----------
    model: :py:class:`ContinuousModel` or :py:class:`ProbabilisticModel` object instance. 
        The model trained in last query. 
        
        
    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:
    
    .. code-block:: python
       from Query import UncertaintySampling
       from models import LogisticRegression
       qs = UncertaintySampling(
                dataset, # Dataset object
                model=LogisticRegression(C=0.1)
            )
    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.
    
    
    References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """
    
    def __init__(self, *args, **kwargs):
        super(UncertaintySampling, self).__init__(*args, **kwargs)
        
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel) and \
            not isinstance(self.model, ProbabilisticModel):
                raise TypeError(
                    "model has to be a ContinuousModel or ProbabilisticModel"
                )
                
        # self.model.train(self.dataset) # Why train at init? Will be trained at query..
        
        self.method = kwargs.pop('method', 'lc')
        if self.method not in ['lc', 'sm', 'entropy']:
            raise TypeError(
                "supported methods are ['lc', 'sm', 'entropy'], the given one "
                "is: " + self.method
            )
            
        if self.method == 'entropy' and \
            not isinstance(self.model, ProbabilisticModel):
                raise TypeError(
                    "method 'entropy' requires model to be a ProbabilisticModel"
                )
                
    def _get_scores(self):
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        
        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_prob(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)
            
        if self.method == 'lc': # least confident
            score = -np.max(dvalue)
        elif self.method == 'sm': # smallest margin
            if np.shape(dvalue)[1] > 2:
                # find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])
        elif self.method == 'entrop':
            score = np.sum(-dvalue * np.log(dvalue), axis=1)
        return zip(unlabeled_entry_ids, score)
    
    def make_query(self, top_n=1, return_score=False):
        """
        Return the index(es) of the sample to be queried and labeled. Optionally:
        a) query more than one example per training round; and b) return selection
        score(s) of for each sample(batch). Read-only.
        
        No modification to the internal states.
        
        
        Parameters
        ----------
        top_n: int, optional (default=1)
            The number of samples to query per training round. 
            
        return_score: bool, optional, (default=False)
            Return the associated uncertainty score for each query sample. 
        
        
        Returns
        -------
        ask_id: int
            The index of the next top_n unlabaled samples to be queried and labeled. 
            
        score: list of (index, score) tuples
            Selection score of unlabaled entries, the larger the better (more confident)
        """
        unlabeled_entry_ids, scores = zip(*self._get_scores())
        if top_n == 1:
            ask_id = np.argmax(scores)
        elif top_n > 1:
            ask_id = np.argpartition(scores, -top_n)[-top_n:][::-1]
            
        if return_score:
            return unlabeled_entry_ids[ask_id], list(zip(unlabeled_entry_ids, scores))
        else:
            return unlabeled_entry_ids[ask_id]
        
        
