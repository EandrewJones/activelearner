"""
Concrete model classes
"""
from .logistic_regression import LogisticRegression
from .bagging_classifier import BaggingClassifier
from .random_forest_classifier import RandomForestClassifier
from .svm import SVM
from .xgboost_classifier import XGBClassifier

__all__ = [
    'LogisticRegression',
    'BaggingClassifier',
    'RandomForestClassifier',
    'SVM',
    'XGBClassifier'
]