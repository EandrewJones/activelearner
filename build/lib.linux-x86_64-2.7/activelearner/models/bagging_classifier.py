'''
Bagging Classifier

A wrapper for scikit-learns Bagging Classifier
'''
import sklearn.ensemble

from activelearner.interfaces import ProbabilisticModel


class BaggingClassifier(ProbabilisticModel):
    """
    Bagged Ensemble Classifier
    
    Creates a classifier ensemble of arbitrary base models
    from the scitkit-learn library. Base model must be specified by
    the user.
    
    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    """
    def __init__(self, *args, **kwargs):
        self.model = sklearn.ensemble.BaggingClassifier(*args, **kwargs)
    
    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
    
    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)
        
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)
    
    def predict_prob(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
        
    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1: # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
        
