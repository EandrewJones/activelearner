'''
Batch Sampling

This module contains a class that implements ranked batch uncertainty sampling.
'''
import numpy as np
import psutil
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin_min
)
from activelearner.interfaces import (
    Query,
    ContinuousModel,
    ProbabilisticModel
)


class BatchUncertaintySampling(Query):
    """
    Batch Samping
    
    This class implements ranked batch uncertainity sampling strategies [1]_.
    
    Parameters
    ----------
    model: :py:class:`ContinuousModel` or :py:class:`ProbabilisticModel`
        The base model used for training
        
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        least confidence (lc), queries instance whose posterior probability
            of being postive is nearest 0.5 (for binary classification);
        smallest margin (sm), it queries the instance whose posterior
            probability gap between the most and second most probable
            labels is minimal;
        entropy, requires :py:class:`ProbabilisticModel` to be passed in as
            model parameter;

    metric: string, default = 'euclidean'
        Metric used to calculate similarity scores. One of [‘cityblock’,
        ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, braycurtis’,
        ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’,
        ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
        ‘sokalsneath’, ‘sqeuclidean’, ‘yule’].
        This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
    
    batch_size: int, default = 100
        Number of unlabeled observations to query.
        
    n_jobs: int, default = Number of cores - 2
        Number of threads to use for multiprocessing.
        This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.

    
    Attributes
    ----------
    model: :py:class:`ContinuousModel` or :py:class:`ProbabilisticModel` object instance.
        The model trained in last query.
        
    References
    ----------
    .. [1] Cardoso et al. 2017. "Ranked batch-mode active learning." 
            Information sciences. 379: 313-337.
    """
    
    def __init__(self, *args, **kwargs):
        super(BatchUncertaintySampling, self).__init__(*args, **kwargs)
        
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

        self.metric = kwargs.pop('metric', 'euclidean')
        allowed_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 
                           'l2', 'manhattan', 'braycurtis', 'canberra', 
                           'chebyshev', 'correlation', 'dice', 'hamming',
                           'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
                           'rogerstanimoto', 'russellrao', 'seuclidean', 
                           'sokalmichener', 'sokalsneath', 'sqeuclidean', 
                           'yule']
        if self.metric not in allowed_metrics:
            raise TypeError(
                "metric not allowed. See documentation for list of allowed"
                "metrics."
            )
        
        self.batch_size = kwargs.pop('batch_size', 100)
        self.n_jobs = kwargs.pop('n_jobs', psutil.cpu_count() - 2)
               
    def _get_scores(self):
        """
        Calculates uncertainty and similarity scores which
        are used in make_query() to calculate a new weighted score
        combining the dissimilarity in X_pool and X_training and our 
        uncertainty scores for the unlabeled data.
        """
        dataset = self.dataset
        self.model.train(dataset)
        
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_train, _ = dataset.get_labeled_entries()
        n_unlabeled = dataset.len_unlabeled()
        
        # Calculate uncertainty scores
        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_prob(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)
            
        if self.method == 'lc':  # least confident
            uncertainty_scores = -np.max(dvalue, axis=1)
        elif self.method == 'sm':  # smallest margin
            if np.shape(dvalue)[1] > 2:
                # find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            uncertainty_scores = -np.abs(dvalue[:, 0] - dvalue[:, 1])
        elif self.method == 'entropy':
            uncertainty_scores = np.sum(-dvalue * np.log(dvalue), axis=1)
        
        # Calculate similarity scores
        # TODO batch size needs to be tunable parameter that
        # is optimized for different data dimensions
        bs = 10000
        if n_unlabeled > bs:
            # compute batched pairwise distance
            distance_scores = None
            for i in range((n_unlabeled - 1) // bs + 1):
                start_i = i * bs
                end_i = start_i + bs
                X_pool_batch = X_pool[start_i:end_i, :]
                
                if self.n_jobs == 1 or self.n_jobs is None:
                    _, distance_scores_batch = pairwise_distances_argmin_min(
                        X_pool_batch,
                        X_train,
                        metric=self.metric
                    )
                else:
                    distance_scores_batch = pairwise_distances(
                        X_pool_batch,
                        X_train,
                        metric=self.metric,
                        n_jobs=self.n_jobs
                    ).min(axis=1)
                
                if distance_scores is None:
                    distance_scores = distance_scores_batch
                else:
                    distance_scores = np.append(distance_scores,
                                                distance_scores_batch)
        else:
            if self.n_jobs == 1 or self.n_jobs is None:
                _, distance_scores = pairwise_distances_argmin_min(
                    X_pool,
                    X_train,
                    metric=self.metric
                    )
            else:
                distance_scores = pairwise_distances(
                    X_pool,
                    X_train,
                    metric=self.metric,
                    n_jobs=self.n_jobs
                    ).min(axis=1)        
        similarity_scores = 1 / (1 + distance_scores)
        
        return uncertainty_scores, similarity_scores
                     
    def make_query(self):
        """
        Selects entry from unlabeled entries for labeling.
        
        Given labeled unlabeled entries, select_instance identifies
        the besst instance in unlabeled entries that best balances
        uncertainty with dissimilarity.
        
        Parameters
        ----------
        similarity_scores: array-like
            Similarity (minimum pairwise distance) between unlabeled
            (or already picked) observations and labeled observations.
        uncertainty_scores: array-like
            Uncerainty scores for unlabeled data calculated from
            _get_scores.    
        mask: array-like
            Array of previously selected observations indices to be masked
            from future selection.
        """
        
        # Extract number of labeled and unlabeled records
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        n_labeled = dataset.len_labeled()
        n_unlabeled = dataset.len_unlabeled()
        
        # The maxmimum number of records to sample
        ceiling = np.minimum(n_unlabeled, self.batch_size)
        
        # initialize transparent mask for unlabeled entries
        mask = np.ones(n_unlabeled, np.bool)
        
        # Get uncertainty and similarity scores
        uncertainty_scores, similarity_scores = self._get_scores()
        
        # Create batch list of ids to index unlabeled observations
        batch_ask_ids = []
        for _ in range(ceiling):
            
            # Determine our alpha parameter (weight for scores)
            alpha = n_unlabeled / (n_unlabeled + n_labeled)

            # calculate scores
            scores = alpha * (1 - similarity_scores[mask]) + \
                (1 - alpha) * uncertainty_scores[mask]

            # Find index of the best unlabeled sample to be queried
            ask_id = np.argmax(scores)
            
            # update the selected observation as 'labeled'
            ask_observation = X_pool[mask][[ask_id]]
            mask[ask_id] = False
            n_unlabeled -= 1
            n_labeled += 1
            
            # compute distance to queried sample
            distance_to_labeled = pairwise_distances(
                X_pool, 
                ask_observation, 
                metric=self.metric, 
                n_jobs=self.n_jobs
                )[:, 0]

            # Update similarity scores to account for queried observation
            # Because the distance to itself will be zero, we can ensure
            # next iteration won't redraw the same sample without having
            # to modify the dataset
            similarity_scores = np.max(
                [similarity_scores, 1 / (1 + distance_to_labeled)],
                axis=0
                )
            
            # Append queried sample to batch list
            batch_ask_ids.append(ask_id)
            
        return unlabeled_entry_ids[batch_ask_ids]