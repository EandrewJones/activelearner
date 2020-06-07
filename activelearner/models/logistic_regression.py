'''
Logistic Regression

Interface to scikit-learn's logistic regression model 
'''
import sklearn.linear_model
import numpy as np 

from activelearner.interfaces import ProbabilisticModel

class LogisticRegression(ProbabilisticModel):
    """
    Logistic Regression Classifier
    
    Wrapper for scikit-learn logistic regression with added
    functions to integrate with the active learning module. 
    
    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    
    def __init__(self, *args, **kwargs):
        self.model = sklearn.linear_model.LogisticRegression(*args, **kwargs)
        
    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
    
    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)
    
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)
    
    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1: # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T 
        else:
            return dvalue
        
    def predict_prob(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
