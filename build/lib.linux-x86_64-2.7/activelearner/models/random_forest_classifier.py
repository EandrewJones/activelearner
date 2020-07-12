'''
Random Forest Classifier

A wrapper for scikit-learns Random Forest Classifier Ensemble
'''
import sklearn.ensemble

from activelearner.interfaces import ProbabilisticModel


class RandomForestClassifier(ProbabilisticModel):
    """
    Random Forest Classifier
    
    Creates a random forest classifier ensemble using sklearn library.
    
    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    def __init__(self, *args, **kwargs):
        self.model = sklearn.ensemble.RandomForestClassifier(*args, **kwargs)
        
    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
    
    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)
        
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)
    
    def predict_prob(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
