'''
QUerying Informative and Representative Examples (QUIRE)

This module contains a class that implements an active learning algorithm (query_strategy): QUIRE
'''

import numpy as np 
import bisect
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from activelearner.interfaces import Query

            
class QUIRE(Query):
    """
    QUerying Informative and Representative Examples (QUIRE)
    
    Querying the most informative and representative examples where the metrics
    measuring and combining are executed using a min-max approach.
    
    Parameters
    ----------
    lambda: float, optional (default=1.0)
        A regularization parameter used in the learning framework.
        
    kernel: {'linear', 'poly', 'rbf', 'callable'} optional (default='rbf')
        Specifies the kernel type to be used in the algorithm. 
        It must be one of 'linear', 'poly', 'rbf', or a callable. 
        If a callable is given it is used to pre-compute the kernel matrix from
        data matrices; that matrix should be an array of shape ``(n_samples, n_samples)``
    
    degree: int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
        
    gamma: float, optional (default=1.0)
        Kernel coefficient for 'rbf', 'poly'
        
    coef0: float, optional (default=1.0)
        Independent term in kernel function. 
        It is only significiant in 'poly'. 
        
    
    Attributes
    ----------
    
    Examples
    --------
    How to delcare a QUIRE Query object:
    
    .. code-block:: python
    
        from Query import QUIRE
        
        qs = QUIRE(
                dataset,
        )
        
    References
    ----------
    .. [1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying
           informative and representative examples.
    """
    
    def __init__(self, *args, **kwargs):
        super(QUIRE, self).__init__(*args, **kwargs)
        self.Uindex = self.dataset.get_unlabeled_entries()[0].tolist()
        self.Lindex = np.where(self.dataset.grepl_labeled())[0].tolist()
        self.lmbda = kwargs.pop('lambda', 1.0)
        X, self.y = self.dataset.get_observations()
        self.y = list(self.y)
        self.kernel = kwargs.pop('kernel', 'rbf')
        
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.0))
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       degree=kwargs.pop('degree', 3),
                                       coef0=kwargs.pop('coef0', 1),
                                       gamma=kwargs.pop('gamma', 1.0))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError
        
        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self.K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel matrix should have size ({}, {})'.format(len(X), len(X))
            )
        
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))
        
    def update(self, entry_id, label):
        bisect.insort(a=self.Lindex, x=entry_id)
        self.Uindex.remove(entry_id)
        self.y[entry_id] = label
        
    def make_query(self):
        L = self.L 
        Lindex = self.Lindex
        Uindex = self.Uindex
        query_index = -1 
        min_eva = np.inf
        y_labeled = np.array([label for label in self.y if label is not None])
        
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        
        # efficient computuation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex)) + self.K[np.ix_(Lindex, Lindex)]))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0] 
        
        for i, each_index in enumerate(Uindex):
            # iterate all unlabled instances and compute their evaluations
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled
            )
            eva = L[each_index][each_index] - \
                det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)
            
            # TODO add way to store top_n in efficient manner without keep track of all results
            if eva < min_eva: 
                query_index = each_index
                min_eva = eva
        
        # TODO alter make_query() method such that it meets form return unlabaled_entry_ids[ask_id]    
        return query_index
