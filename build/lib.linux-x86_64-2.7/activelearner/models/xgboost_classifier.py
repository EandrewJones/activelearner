
'''
eXtreme Gradient Boosting Classifier

A wrapper for XGBoost's gradient boosting classifier.
'''
import xgboost as xgb 

from activelearner.interfaces import ProbabilisticModel


class XGBClassifier(ProbabilisticModel):
    """
    XG Boost Classifier
    
    Creates an eXtreme Gradient Boosting Classifier using the XGBoost library. 
    
    References
    ----------
    https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
    """
    def __init__(self, *args, **kwargs):
        self.model = xgb.XGBClassifier(*args, **kwargs)
        
    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
    
    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)
    
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.eval_result(*(testing_dataset.format_sklearn() + args), **kwargs)
    
    def predict_prob(self,feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
