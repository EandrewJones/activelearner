from __future__ import print_function
from __future__ import unicode_literals
import sys
import copy

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

from six import with_metaclass
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

'''
Base interfaces for active learner
The package works according to the (abstract) inferfaces defined below
'''

#=========#
# Classes #
#=========#

'''
Dataset class

Datasets consist of partially-labeled targets and 
associated features represent by a list of
(Feature: X, Label: y) tuples.add()

Options included for exporting to formats supported by
scikit-learn and scipy
'''
class Dataset(object): 
    """
    Dataset object for storing feature and labels
    
    This abstract factory class lays out required methods
    across all concrete
    
    Parameters
    ----------
    X : {array-like}, shape = (n_samples, n_features)
        Feature of sample set
        
    Y : list of {int, None}, shape = (n_samples)
        The ground truth (label) for corresponding sample. Unlabeled data
        should be given a label None.
        
    Attributes
    ----------
    data : list, shape = (n_samples)
        List of all sample feature, label tuples
        
    view: tuple, shape = (n_samples)
        Tuple of all raw sample features (text, image, etc.). Maintains
        original state of features for Oracle to view despite possible 
        feature engineering.
    
    """
    def __init__(self, X=None, y=None):
        if X is None: 
            X = np.array([])
        elif not isinstance(X, sp.csr_matrix):
            X = np.array(X)
            
        if y is None:
            y = [None]
        y = np.array(y)
        
        self._X = X
        self._y = y
        self._view = tuple(copy.copy(X))
        self.modified = True
        self._update_callback = set()
        
    def __len__(self):
        """
        Number of all sample entries in object.
        
        Returns
        -------
        n_samples: int
        """
        return self._X.shape[0]
    
    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]
    
    @property
    def data(self): return self
    
    @property
    def view(self): return self._view
    
    def grepl_labeled(self):
        """
        Get the mask of labeled entries
        
        Returns
        -------
        mask: numpy array of bool, shape = (n_sample, )
        """
        return ~np.fromiter((e is None for e in self._y), dtype=bool)
    
    def len_labeled(self):
        """
        Number of labeled data entries in dataset.
        
        Returns
        -------
        n_samples: int
        """
        return self.grepl_labeled().sum()
    
    def len_unlabeled(self):
        """
        Number of unlabled data entries in dataset.
        
        Returns
        -------
        n_samples: int
        """
        return (~self.grepl_labeled()).sum()
    
    def get_num_of_labels(self):
        """
        Number of distinct labels in this dataset. 
        
        Returns
        -------
        n_labels: int
        """
        return np.unique(self._y[self.grepl_labeled()]).size
    
    def append(self, feature, label=None):
        """
        Add a (feature, label) entry into the dataset/
        A None label indicates an unlabeled entry.
        
        Parameters
        ----------
        feature: {array-like}, shape = {n_features}
            Feature of the sample to append to dataset.
            
        label: {int, None}
            Label of the observation to append to dataset. None if unlabeled.
        
        Returns
        -------
        entry_id: {int}
            entry_id for the appended observation. 
        """
        if isinstance(self._X, np.ndarray):
            self._X = np.vstack([self._X, feature])
        else: 
            self._X = sp.vstack([self._X, feature])
        self._Y = np.append(self._y, label)
        
        self.modified = True
        return len(self) - 1
    
    def update(self, entry_id, new_label):
        """
        Assigns label to observation with given entry_id. 
        
        Parameters
        ----------
        entry_id: int
            entry index of the observation to update.
            
        new_label: {int, None}
            Label to be assigned to given observation.
        """
        self._y[entry_id] = new_label
        self.modified = True
        for callback in self._update_callback:
            callback(entry_id, new_label)
            
    def on_update(self, callback):
        """
        Add callback function to call when dataset updated. 
        
        Parameters
        ----------
        callback: callable
            The function to be called when dataset is updated.
        """
        self._update_callback.add(callback)
        
    def format_sklearn(self):
        """
        Returns dataset in (X, y) format for use in scikit-learn.
        Unlabled entries are ignored. 
        
        Returns
        -------
        X: numpy array, shape = (n_samples, n_features)
            Sample feature set.
            
        y: numpy array, shape = (n_samples)
            Target labels.
        """
        # becomes the same as get_labeled_entries
        X, y = self.get_labeled_entries()
        return X, np.array(y)
    
    def get_observations(self):
        """
        Return the list of all sample feature and ground truth tuples.
        
        Returns
        -------
        X: numpy array or scipy matrix, shape = (n_sample, n_features)
        y: numpy array, shape (n_samples),
        """
        return self._X, self._y
    
    def get_labeled_entries(self):
        """
        Returns list of labeled observations and their labels.
        
        Returns
        -------
        X: numpy array or scipy matrix, shape = (n_sample labeled, n_features)
        y: list, shape = (n_samples labeled)
        """
        return self._X[self.grepl_labeled()], self._y[self.grepl_labeled()].tolist()
    
    def get_unlabeled_entries(self):
        """
        Returns list of unlabeled features, along with their entry_ids
        
        Returns
        -------
        idx: numpy array, shape = (n_samples unlabeled)
        X: numpy array or scipy matrix, shape = (n_sample unlabeled, n_features)
        """
        return np.where(~self.grepl_labeled())[0], self._X[~self.grepl_labeled()]
    
    def labeled_uniform_sample(self, sample_size, replace=True):
        """
        Returns a Dataset objet with labeled data only which is
        resampled uniformly with given sample size. 
        
        Parameters
        ----------
        sample_size: int
            Number of samples to draw from labeled data.
        replace: bool
            Whether to sample with replacement or not, default is True. 
            
        Returns
        -------
        Data: list, shape (sample_size)
            Resampled list of (feature, target) tuple-like pairs
        """
        idx = np.random.choice(np.where(self.grepl_labeled())[0],
                               size=sample_size, replace=replace)
        return Dataset(self._X[idx], self._y[idx])
    
    def get_dataset_stats(self):
        """
        Returns a pandas dataframe with descriptive statistics on
        labeling progress.
        
        Returns
        -------
        Data: Pandas Series, shape (n_unique_labels, 3)
            A Pandas Series with three columns: unique target labels,
            value counts for each label, and relative frequencies of
            each label
            
        Progress bar:
            Prints total percentage of dataset with a label
        """
        # Descriptive stats data frame
        y_series = pd.Series(self._y)
        counts = y_series.value_counts(dropna=False).to_frame()
        freqs = y_series.value_counts(normalize=True, dropna=False).to_frame()
        stats_df = pd.concat([counts, freqs], axis=1)
        stats_df.columns = ['Count', 'Percentage']
        print(stats_df)
        
        # Progress bar
        non_null_labels = pd.unique(y_series.dropna())
        progress = stats_df.loc[non_null_labels]['Percentage'].sum()
        update_progress(progress=progress)



class Query(with_metaclass(ABCMeta, object)):
    """
    Pool-based query strategy
    
    A Query advises on which unlabeled data to be queried next from
    a pool of labeled and unlabeled data. 
    """
    
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        dataset.on_update(self.update)
        
    @property
    def dataset(self):
        """The dataset object associated with this Query"""
        return self._dataset
    
    def update(self, entry_id, label):
        """
        Update the internal states of the Querier after each queried
        sample is labeled.
        
        Parameters
        ----------
        entry_id: int
            The index of the newly labeled observation.
            
        label: float
            The label of the queried example.    
        """
        pass
    
    def _get_scores(self):
        """
        Return the score used for making query, the larger the better. Read-only.
        
        No modification to the internal states.
        
        Returns
        -------
        (ask_id, scores): list of tuple (int, float)
            The index of the next unlabeled observation to be queried and the score assigned.
        """
        pass
    
    @abstractmethod
    def make_query(self):
        """
        Return the index of the observation to be queried and labeled. Read-only.
        
        No modification to the internal states.
        
        Returns
        -------
        ask_id: int
            The index of the next unlabeled observation to be queried and labeled.
        """
        pass



class Labeler(with_metaclass(ABCMeta, object)):
    """
    Label the queries made by Query
    
    Assigns labels to the observations queried by Query
    """
    @abstractmethod
    def label(self, feature):
        """
        Returns the class labels for the input feature array.
        
        Parameters
        ----------
        feature: array-like, shape (n_features, )
            The feature vector whose label is to be quiried.
            
        Returns
        -------
        label: int
            The class of the queried feature
        """
        pass

class Model(with_metaclass(ABCMeta, object)):
    """
    Classification Model
    
    A Model returns a class-predicting function for unlabeled
    samples after training on the labeled set.
    """
    @abstractmethod
    def train(self, dataset, *args, **kwargs):
        """
        Train a model with given training dataset
        
        Parameters
        ----------
        dataset: Dataset objet
            a training dataset the model will be trained on.
            Contains target variable Y and feature matrix X. 
        
        Returns
        -------
        self: object
            Returns self.
        """
        pass
    
    @abstractmethod
    def predict(self, feature, *args, **kwargs):
        """
        Predicts class labels for input samples based on
        trained model. 
        
        Parameters
        ----------
        feature: array-like, shape (n_samples, n_features)
            Unlabeled samples for which we desire predictions. 
            
        Returns
        -------
        y_pred: array-like, shape (n_samples,)
            The predicted class labels for samples in feature array.
        """
        pass
    
    @abstractmethod
    def score(self, testing_dataset, *args, **kwargs):
        """
        Return the mean accuracy on the test dataset
        
        Parameters
        ----------
        testing_dataset: Dataset object
            The testing dataset used to measure the performance of the trained model. 
            
        Returns
        -------
        score: float
            Mean accuracy of self.predict(X) wrt. y. 
        """
        pass
    

class MultilabelModel(Model):
    """
    Multilabel Classification Model
    
    A Model that retuns a multilabel-prediction function for unlabeled samples
    after training on labeled data. 
    """
    pass


class ContinuousModel(Model):
    """
    Classification Model with intermediate continuous output
    
    A continuous classification model is able to output a real-valued
    vector (float) for each unlabeled sample.
    """
    @abstractmethod
    def predict_real(self, feature, *args, **kwargs):
        """
        Predict confidence scores for samples. 
        
        Returns the confidence score for each (sample, class) combination. 
        
        The larger the value for entry (sample=x, class=k) is, the more 
        confident the model is about the sample x belonging to class k. 
        
        E.G. For logistic regression, the returned value is the signed
        distance of that smaple to the hyperplane. 
        
        Parameters
        ----------
        feature: array-like, shape (n_samples, n_features)
            The samples whose confidence scores are to be predicted.
            
        Returns
        -------
        X: array-like, shape (n_samples, n_features)
            Each entry is the confidence scores per (sample, class)
            combination.
        """
        pass
    

class ProbabilisticModel(ContinuousModel):
    """
    Classification Model with probability output.
    
    A probabilistic classification model outputs a real-valued vector [0,1]
    for each feature provided representing probability of belonging to class k.
    """
    def predict_real(self, feature, *args, **kwargs):
        return self.predict_prob(feature, *args, **kwargs)
    
    @abstractmethod
    def predict_prob(self, feature, *args, **kwargs):
        """
        Predict probability estimate for sample
        
        Parameters
        ----------
        feature: array-like, shape (n_samples, n_features)
            The features of samples for which we would like to predict probability
            of belonging to a given class. 
        
        Returns
        -------
        X: array-like, shape (n_samples, n_classes)
            Each entry is the probability estimate for each class
        """
        pass
   
    
#===========#
# Functions #
#===========#


def import_libsvm_sparse(filename):
    """Imports datast file in libsvm sparse format"""
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(filename)
    return Dataset(X.toarray(), y)


def import_scipy_matrix(filename):
    """Imports dataset file in scipy matrix format"""
    from scipy.io import loadmat
    data = loadmat(filename)
    X = data['X']
    y = data['y']
    zipper = list(zip(X, y))
    np.random.shuffle(zipper)
    X, y = zip(*zipper)
    X, y = np.array(X), np.array(y).reshape(-1)
    return Dataset(X, y)


def update_progress(progress):
    '''
    Displays or updates a console progress bar
    
    Arguments
    ---------
    Progress: float
        A float value between 0 and 1. Any int will be converted
        to float. A value < 0 indicates a 'halt'. 1 or bigger
        indicates 'completion'.
        
    Returns
    -------
    command-line progress bar
    '''
    barLength = 40 # modify to change length of progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be a float\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress > 1:
        progress = 1
        status = 'Done...\r\n'
    block = int(round(barLength*progress))
    text = '\rProgress: [{0}] {1}% {2}\r\n'.format("="*block + " "*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
