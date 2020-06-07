# utils
from __future__ import print_function
from __future__ import unicode_literals
import time
import sys
import os
import copy
import re
import psutil
import warnings 

# building classes
from six import with_metaclass
from six.moves import input
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

# scientific computing
import random
import numpy as np
import pandas as pd

import scipy.sparse as sp
from scipy.spatial import ConvexHull

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Text analysis
from keras_preprocessing.text import text_to_word_sequence

# =========== #
#  FUNCTIONS  #
# =========== #

# Decorator function for classes to inherit docstrings from cls
def inherit_docstring_from(cls):
    """
    Decorator for class methods to inherit docstring from :code:`cls`
    """
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


# Function to instantiate random state
def seed_random_state(seed):
    """
    Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to generate numpy.random.RandomState"
                     "instance" % seed)
    

# Function to calculate cost/loss for label preditions
def calc_cost(y, yhat, cost_matrix):
    """
    Calculate the cost with given cost matrix
    
    y: ground truth
    
    yhat: prediction
    
    cost_matrix: array-like, shape=(n_classes, n_classes)
        The ith row, jth column represents the cost of the ground truth being
        ith class and prediction as jth class
    """
    return np.mean(cost_matrix[list(y), list(yhat)])


# Function to import data in libsvm format
def import_libsvm_sparse(filename):
    """Imports datast file in libsvm sparse format"""
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(filename)
    return Dataset(X.toarray(), y)


# Function to import data into scipy format
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


# Function to display progress bar
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
    

# Function to clean text
def preprocess(text, to_lower=True, remove_tags=True, digits=True, special_chars=True):
    """
    Takes a text object as input and performs a sequence of processing 
    operations: lowercases , removes html tags and special characters
    
    Parameters
    ----------
    text: str
        Text to be cleaned. 
        
    to_lower: bool (default=True)
        Whether to convert text to lower case.
    
    remove_tags: bool (default=True)
        Whether to remove content form within html <> tags. 
    
    digits: bool (default=True)
        Whether to remove numbers. 
    
    special_chars: bool (default=True)
        Whether to remove special characters.
        
    Returns
    -------
    text: str
        cleaned text
    """
    if to_lower:
        text = text.lower()

    if remove_tags:
        text = re.sub("</?.*?>", " <> ", text)
        
    if digits:
        text = re.sub('\\d+', '', text)
        
    if special_chars:
        text = re.sub('\\W+', ' ', text)

    text = text.strip()
    
    return text


# Function to calculate unique tokens and number of unique tokens in corpus
def unique_tokens(corpus):
    """Takes corpus and returns unique tokens and number of unique tokens."""
    uniq_tokens = set()
    
    for doc in corpus:
        doc = preprocess(doc)
        doc = text_to_word_sequence(doc)
        uniq_tokens.update(doc)
    
    return uniq_tokens, len(uniq_tokens)


# Function to calculate a gram matrix
def gram_rp(dtm, s=.05, p=3000, d_group_size=2000):
    """ 
    Copmutes gram.rp matrix from sparse dtm 
    
    Parameters
    ----------
    s: float, between [0, 1]
        The desired sparsity level. 
        
    p: int
        The number of projections. 
        
    d_group_sized: int
        The size of the groups of documents analyzed.
    """
    
    # Get projection matrix
    n_items = np.ceil(s*p).astype('int')
    triplet = [None] * dtm.shape[1]
    
    for i in range(dtm.shape[1]):
        col = np.random.choice(a=np.arange(0, p), size=n_items, replace=False)
        row = np.repeat(i, len(col))
        values = np.random.choice(a=[-1, 1], size=len(col), replace=True)
        triplet[i] = np.column_stack((row.ravel(), col.ravel(), values.ravel()))
    triplet = np.concatenate(triplet)
    row, col = triplet[:, 0], triplet[:, 1]
    data = triplet[:, 2]
    proj = sp.coo_matrix((data, (row, col))).toarray()
    
    # perform projection
    D = dtm.shape[0]
    groups = np.ceil(D / d_group_size).astype('int')
    Qnorm = np.zeros(dtm.shape[1])
    for i in range(groups):
        smat = dtm[(i * d_group_size):min((i+1)*d_group_size, D), :]
        rsums = np.array(np.sum(smat, axis=1))
        divisor = rsums * (rsums-1) + 1e-07
        smatd = np.array(smat / divisor)
        Qnorm = Qnorm + np.sum(smatd * (rsums - 1 - smatd), axis=0)
        Htilde = np.sum(smatd, axis=0)[:, np.newaxis] * proj
        smat = smat / np.sqrt(divisor)
        
        if i > 1:
            Q = Q + np.matmul(smat.T, np.matmul(smat, proj)) - Htilde
        else:
            Q = np.matmul(smat.T, np.matmul(smat, proj)) - Htilde
     
    return Q / Qnorm[:, np.newaxis]
 
 
# Function to solve convex hull using tSNE project
def tsne_anchor(qbar, init_dims=50, perplexity=30, **kwargs):
    """
    Calculates anchor words from gram matrix by projecting matrix onto
    lower-dimensional (2 or 3D) representation using tSNE. Then exactly
    solves for convex hull.
    
    Parameters
    ----------
    qbar: sparse-matrix, shape (n_documents, n_words)
        Gram matrix. 
        
    init_dims: int, default=50
        Number of dimensions to initialize PCA with. 
        
    perplexity: int, default=30
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significanlty different results. 
        
    n_jobs: int, default: number of physical cores - 2
        The number of cores to use for multiprocessing. Automatically defaults to
        run the algorithm in paralell with the max number of physical cores less 2. 
        
    Returns
    -------
    anchor: array-like, shape (n_anchors)
        A column-index of anchor words from the document-term matrix. 
    """
    n_components = min(init_dims, qbar.shape[1])
    n_jobs = kwargs.pop('njobs', psutil.cpu_count() - 2)
    
    # PCA init
    pca_model = PCA(n_components=n_components)
    xpca = pca_model.fit_transform(qbar)
    
    # tSNE
    # TODO add try-catch for errors
    tsne_model = TSNE(n_components=3, perplexity=perplexity, n_jobs=n_jobs)
    proj = tsne_model.fit_transform(xpca)
    

    # solve convexhull
    hull = ConvexHull(proj)
    return hull.vertices


# Function to read corpus into gensim format
def gensim_read_corpus(docs, tokens_only=False):
    for i, doc in enumerate(docs):
        doc = preprocess(doc)
        tokens = text_to_word_sequence(doc)
        yield TaggedDocument(tokens, [i])
    

# Function to save (pickle) python objects
def save_object(obj, filename):
    '''
    Saves python objects to specified filename. Will
    overwrite file
    
    Arguments
    ---------
    obj: python object
    
    filename: file path + name
    '''
    import pickle
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# function for performing ceiling division without importing math module
def ceildiv(a, b):
    return -(-a // b)

      
# Main function loop for active_learner algorithm        
def run_active_learner(X, y, feature_type, label_name, save_every=20,
                       print_progress=True, **kwargs):
    '''
    Runs main active learning algorithm loop, prompting Oracle for correct label and
    updating dataset. Currently only uses random sampling query strategy. Not yet implemented option for "active" updating via model-based query strategy.
    
    Parameters
    ----------
    X: {array-like}, shape (n_observations, n_features)
    
    y: vector, length n_observations
    
    feature_type: string
        Identifies the data structure of the feature. Must be either
        'text' or 'image'.
    
    label_name: list of strings
        Let the label space be from 0 to len(label_name)-1, this list
        corresponds to each label's name. If label_names are numeric, 
        e.g. 0,1,...,N must be entered as strings '0','1',...
    
    save_every: int (default=20)
        Number of iterations after which algorithm should save updated
        dataset and print labeling progress.
        
    print_progress: bool (default=True)
        Logical indicating whether to print labeling progress upon save.
    
    feature_name (optional): string
        The name of the feature being labeled. If provided, will be
        displayed to Oracle as a reminder when querying them.
    
    keywords (optional): list
            If feature_type is 'text', can a provide a list of 
            keywords to highlight when displayed in the console. 
            
    path (optional): string
        A character string specifying path location for saving dataset. Default
        is current working directory.
        
    file_name (optional): string
        A character string specifying filename for saved dataset object. Defaults to
        'dataset' if unspecified.
        
    seed (optional): int
        A random seed to instantite np.random.RandomState with. Defaults
        to 1 if unspecified.
    '''  
    # Argument checking
    assert X is not None
    assert y is not None
    
    # Extract optional arguments
    path = kwargs.pop('path', os.getcwd())
    file_name = kwargs.pop('file_name', 'dataset.pkl')
    feature_name = kwargs.pop('feature_name', None)
    keywords = kwargs.pop('keywords', None)
    seed = kwargs.pop('seed', 1)
    
    # Instantiate model
    # TODO choose which type of Dataset to instantiate
    # TODO choose which type of query strategy
    dataset = Dataset(X, y)
    querier = RandomSampling(dataset, random_state=seed)
    oracle = AskOracle(feature_type=feature_type, label_name=label_name,
                       feature_name=feature_name)
    
    # Model loop
    continue_cycle = True
    while continue_cycle:
        # active learning algorithm loop
        for _ in range(save_every):
            query_id = querier.make_query()
            label = oracle.label(dataset.view[query_id], # change change to dataset._view
                                 keywords=keywords)
            dataset.update(query_id, label)

            progress = (_ + 1) / save_every
            if progress % 5 == 0:
                update_progress(progress)
            
        # Print updated class information
        if print_progress:
            dataset.get_dataset_stats()
        
        # Save dataset object
        fname = os.path.join(path, '', file_name)
        save_object(obj=dataset, filename=fname)
        
        # ask user to input continue cycle 
        banner = f'Would you like to continue labeling another {save_every} examples? [(Y)es/(N)o]: '
        valid_input = set(['Yes', 'Y', 'y', 'yes', 'No', 'N', 'n', 'no'])
        continue_options = set(['Yes', 'Y', 'y', 'yes'])
        
        user_choice = input(banner)
        
        while user_choice not in valid_input:
            print(f'Invalid choice. Must be one of {valid_input}')
            user_choice = input(banner)
        continue_cycle = user_choice in continue_options
        
                
#==========================================#
# (ABSTRACT) FACTORY CLASSES FOR INTERFACE #
#==========================================#

'''
Base interface for active learner
The package works according to the (abstract) inferfaces defined below
'''

'''
Dataset class

Datasets consist of partially-labeled targets and 
associated features represent by a list of
(Feature: X, Label: y) tuples.add()

Options included for exporting to formats supported by
scikit-learn and scipy

# Q) Does this need to abstract?
# A) Seems not. None of its base methods will be modified by concrete classes. 
# Base class will only be extended upon.     
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
    
    @property # redundant?
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
   
    
#==========================#
# DATASET CONCRETE CLASSES #
#==========================#
  
'''
The dataset class

'''
from absl import logging
from tensorflow.python.client import device_lib
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import NMF

from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class TextDataset(Dataset):
    """
    Concrete Dataset class for text features
    
    Extends functionality of dataset factory class with additional
    methods for text feature engineering. Currently includes methods
    for:                                                      
        1) BoW                                                      
        2) doc2vec
        3) Universal Sentence Embeddings (DAN) (tensorflow)
        4) elmo (tensorflow)                                
    
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
        super().__init__(X, y)
        
        # Get basic corpus info
        uniq_tokens, n_uniq_tokens = unique_tokens(self._X)
        self._uniq_tokens = uniq_tokens
        self._n_uniq_tokens = n_uniq_tokens
        
        
    
    def bag_of_words(self, *args, **kwargs):
        """
        Takes corpus and converts to document term matrix according to one of
        three approaches: Hashing, Counts, or TF-IDF. All three methods
        wrap the sklearn library functions. 
        
        Parameters
        ----------
        vectorizer_type, str (default='tf-idf')
            One of 'hash', 'tf-idf', or 'count'.
            
        load_factor: float between [0, 1] (default=0.7)
            The hash loading factor used to determine number of features.
            Optional argument, only used if vectorizer='hash'.
            
        preprocessor: callable, default=self._preprocessor
            Override the preprocessing (string transformation) stage while preserving
            the tokenizing and n-grams generation step. 
            
        tokenizer: callable, default=self._keras_tokenizer
            Override the string tokenization step while preserving the preprocessing and
            n-gram generation steps. Only applies if analyzer == 'word'
            
        stop_words: {'english"}, list, default=spacy.lang.en.stop_words
            A list that is assumed to contain stop words which will be removed
            from the resulting tokens. Default implementation uses spaCy's english
            STOP_WORDS
            
        max_df: float or int, default=1.0
            When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float in range [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts.
            
        min_df: float or int, default=1
            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts.
        
        max_features: int, default=None
            If not None, build a vocabulary that only considers the top max_features ordered by term frequency across the corpus.
        
        Returns
        -------
        X: sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
            
        References
        ----------
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        """
        vectorizer_type = kwargs.pop('vectorizer_type', 'tf-idf')
        preprocessor = kwargs.pop('preprocessor', preprocess)
        tokenizer = kwargs.pop('tokenizer', text_to_word_sequence)
        load_factor = kwargs.pop('load_factor', 0.7)
        stop_words = kwargs.pop('stop_words', list(STOP_WORDS))
        max_df = kwargs.pop('max_df', 1.0)
        min_df = kwargs.pop('min_df', 1)
        max_features = kwargs.pop('min_df', None)
        
        if vectorizer_type == 'tf-idf':
            vectorizer = TfidfVectorizer(*args,
                                         preprocessor=preprocessor,
                                         tokenizer=tokenizer,
                                         stop_words=stop_words,
                                         max_df=max_df,
                                         min_df=min_df,
                                         max_features=max_features)
            dtm = vectorizer.fit_transform(self._X)
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(*args,
                                         preprocessor=preprocessor,
                                         tokenizer=tokenizer,
                                         stop_words=stop_words,
                                         max_df=max_df,
                                         min_df=min_df,
                                         max_features=max_features)
            dtm = vectorizer.fit_transform(self._X)
        elif vectorizer_type == 'hash':       
            vectorizer = HashingVectorizer(*args,
                                           preprocessor=preprocessor,
                                           tokenizer=tokenizer,
                                           stop_words=stop_words,
                                           max_df=max_df,
                                           min_df=min_df,
                                           max_features=max_features)
            n_features = ceildiv(self._n_uniq_tokens, load_factor)
            dtm = vectorizer.fit_transform(self._X, n_features=n_features)
        
        # replace dataset X with document-term matrix
        self._X = dtm
    
    def spectral_NMF(self, *args, **kwargs):
        """
        Transforms sample documents (X) from document-term matrix into a lower-dimensional
        representation using sklearn's non-negative matrix factorization implementation.
        
        Can only be performed after first transforming the corpus into document-term matrix format using 'bag_of_words' method first.
        
        Parameters
        ----------
        n_components: int, optional, default = self-determined
            Number of dimensions (K) of the data. If not specified, will
            be automatically determined using a spectral initialization approach [1] where the document-term matrix is projected into a low-dimensional 
            space (2 or 3D) using tSNE and an exact convex hull is calculated. The vertices of the hull provide a robust approximation of the dimensionality of the data.
        
        max_features: int, default=None
            If not None, build a vocabulary that only considers the top max_features ordered by term frequency across the corpus.
        
        Returns
        -------
        W: array, shape (n_samples, n_components)
            Transformed data with desired number of dimensions (components).
            
        References
        -----------
        .[1] Lee and Mimno, 2014, Low-dimensional Embeddings for Interpretable Anchor-based topic inference, https://mimno.infosci.cornell.edu/papers/EMNLP2014138.pdf
        """
        dtm = self._X
        n_components = kwargs.pop('n_components', None)
        max_features = kwargs.pop('max_features', None)
        
        # spectral initialization
        if n_components is None:
            # Get N = max_features most likely words
            if max_features is not None:
                wprob = np.sum(self._X, axis=0)
                wprob = np.array(wprob / np.sum(wprob)).flatten()
                keep = np.argsort(-wprob)[:max_features]
                dtm = dtm[:, keep]
                
            # perform spectral initialization via tSNE
            Q = gram_rp(dtm)
            anchor = tsne_anchor(qbar=Q)
            n_components = len(anchor)
            
        nmf_model = NMF(n_components=n_components)
        W = nmf_model.fit_transform(dtm)
        self._X = W
        
    def doc2vec(self, *args, **kwargs):
        """
        Transforms the sample documents (X) into document-level embeddings using gensim's Doc2vec algorithm.
        
        Calculates document embeddings from corpus using either distributed bag-of-words (PV-DBOW) approach or a distributed memory (PV-DM) approach. Also entails option
        to train word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training. 
        
        Parameters
        ----------
        vector_size: int, optional, (default=300)
            Dimensionality of the feature vectors.
            
        window: int, optional (default=15)
            The maximum distance between the current and predicted wod within a sentence. 
            
        min_count: int, optional (default=1)
            Ignores all words with total frequency lower than this. 
            
        dm: {1, 0}, optional, (default=0)
            Defines the training algorithm. If dm=1, 'distributed memory' (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is used. 
            
        sample: float, optional (default=1e-5)
            The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
            
        negative: int, optional, (default=5)
            If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
        
        epochs: int, optional, (default=100)
            Number of iterations (epochs) over the corpus. 
        
        workers: int, optional (default=Number of physical cores - 2)
            Use these many worker threads to train the model (=faster training with multicore machines).
        
        Returns
        -------
        X: array, shape (n_samples, vector_size)
            
        """
        # Set parameters
        vector_size = kwargs.pop('vector_size', 300)
        window = kwargs.pop('window', 15)
        min_count = kwargs.pop('min_count', 1)
        dm = kwargs.pop('dm', 0)
        sample = kwargs.pop('sample', 1e-5)
        negative = kwargs.pop('negative', 5)
        epochs = kwargs.pop('epochs', 100)
        workers = kwargs.pop('workers', psutil.cpu_count() - 2)
        
        # transform documents into gensim-readable corpus
        docs = list(gensim_read_corpus(self._X))
        
        # train model
        model = Doc2Vec(docs, vector_size=vector_size, window=window, min_count=min_count,
                        dm=dm, sample=sample, negative=negative, epochs=epochs, workers=workers)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        
        # extract doc_vectors into numpy array format
        doc_vecs = np.zeros([len(model.docvecs), vector_size], dtype='float32')
        for i in range(len(model.docvecs)): # normal for loop iterator approach returns error
            doc_vecs[i, ] = model.docvecs[i]

        self._X = doc_vecs
        
    def elmo(self, *args, **kwargs):
        """
        """
        # Check if tensorflow finds GPU
        if 'GPU' not in str(device_lib.list_local_devices()):
            # Warn user
            warnings.warn(
                'GPU device not found. The is a very computationally expensive model. The use of an accelerator is recommended to avoid reaching resource limits.', 
                ResourceWarning
                )
            
            # ask user to contiue 
            banner = 'Would you like to continue anyways? [(Y)es/(N)o]: '
            valid_input = set(['Yes', 'Y', 'y', 'yes', 'No', 'N', 'n', 'no'])
            continue_options = set(['Yes', 'Y', 'y', 'yes'])
            
            user_choice = input(banner)
            
            while user_choice not in valid_input:
                print(f'Invalid choice. Must be one of {valid_input}')
                user_choice = input(banner)
            if user_choice not in continue_options: return
            
        # configure tensorflow options
        tf.disable_eager_execution()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
                
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        # download model from tensorflow hub
        elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        
        docs = []
        for doc in self._X:
            doc = preprocess(doc)
            docs.append(doc)
            
        embeddings = elmo(
            docs,
            signature='default',
            as_dict=True
        )['elmo']
        
        
    
#====================================================#
# MODELS CONCRETE CLASSES                            #
# - different models for predicting dataset labels   #
#                                                    #
# Includes:                                          #
#   1) Logistic Regression                           #
#   2) SVM                                           #
#   3) Random Forest Ensemble                        #
#   4) Bagging Ensemble                              #
#   5) XGBoost                                       #
#   6) Bert-based transfer learning NN (for text)    #
#       4a) DistilBert                               #
#       4b) LadaBert                                 #
#       4c) FastBert                                 #
#   7) Multiclass Output Conv NN (for images)        #
#====================================================#
'''
Logistic Regression

Interface to scikit-learn's logistic regression model 
'''
import sklearn.linear_model


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
    
'''
SVM

An interrace for scikit-learn's C-Support Vector Classifier Model
'''
import logging
LOGGER = logging.getLogger(__name__)

from sklearn.multiclass import OneVsRestClassifier


class SVM(ContinuousModel):
    """
    C-Support Vector Machine Classifier
    
    When decision_function_shape == 'ovr', uses OneVsRestClassifier(SVC)
    from sklearn.multiclass instead of the output from the SVC directory
    since it is not exactly the implementation of One Vs Rest. 
    
    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    
    def __init__(self, *args, **kwargs):
        self.model = sklearn.svm.SVC(*args, **kwargs)
        if self.model.decision_function_shape == 'ovr':
            self.decision_function_shape = 'ovr'
            # sklearn's ovr isn't real ovr
            self.model = OneVsRestClassifier(self.model)
            
    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
    
    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)
        
    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), 
                                **kwargs)
    
    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1: # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            if self.decision_function_shape != 'ovr':
                LOGGER.warn("SVM model support only 'ovr' for multiclass"
                            "predict_real.")
            return dvalue
        

'''
Bagging Classifier

A wrapper for scikit-learns Bagging Classifier
'''
import sklearn.ensemble


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
        

'''
Random Forest Classifier

A wrapper for scikit-learns Random Forest Classifier Ensemble
'''
import sklearn.ensemble


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
    

'''
eXtreme Gradient Boosting Classifier

A wrapper for XGBoost's gradient boosting classifier.
'''
import xgboost as xgb 


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
    
#================================================================#
# QUERYING TECHNIQUES CONCRETE CLASSES                           #
# - different techniques based on model outputs                  #
#                                                                #
# Includes:                                                      #
#   1) Random Sampling                                           #
#   2) UncertaintySampling                                       #
#       a) Least Confidence Method                               #
#       b) Margin sampling                                       #
#   3) QUerying Informative and Representative Examples (QUIRE)  #
#   4) Query by Committee                                        #
#   5) Variance Reduction    X                                   #
#   6) Density Weighted Uncertainty Sampling      X              #
#================================================================#

'''
Random Sampling
'''


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
        

'''
Uncertainty Sampling

This module contains a class that implements two of the most well-known 
uncertainty sampling query strategies: the least confidence method and
the smallest margin method (margin sampling).
'''


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
        
        
'''
QUerying Informative and Representative Examples (QUIRE)

This module contains a class that implements an active learning algorithm (query_strategy): QUIRE
'''

import bisect
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

# TODO alter make_query() method such that it meets form return unlabaled_entry_ids[ask_id]
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
                
            if eva < min_eva: # TODO add way to store top_n in efficient manner without keep track of all results
                query_index = each_index
                min_eva = eva
                
        return query_index


'''
Query by Committee

This module contains a class that implements Query by committee active learning algorithm. 
'''
import logging
import math

LOGGER = logging.getLogger(__name__)


class QueryByCommittee(Query):
    r"""
    Query by commitee
    
    
    Parameters
    ----------
    models: list of :py:mod:`models` instances or str
        This parameter accepts a list of initialized Model instances,
        or class names of Model classes to determine the Models to be included
        in the commitee to vote for each unlabeled instance.
        
    disagreement: ['vote', 'kl_divergence'], optional (default='vote')
        Sets the method for meaursing disagreement between models.
        'vote' represents vote entropy
        kl_divergence requries models being ProbabilisticModel
        
    random_state: {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance if np.random.RandomState instance, random_state is
        the random number generated. 
        
    
    Attributes
    ----------
    students: list, shape = (len(models))
        A list of the model instances used in this algorithm
        
    random_states\_: np.random.RandomState instance
        The random number generator being used. 
        
    Examples
    --------
    How to declare a QueryByCommittee object:
    
    .. code-block:: python
    
    from Query import QueryByCommittee
    from models import LogisticRegression
    
    qs = QueryByCommittee(
        dataset,
        models=[
            LogisticRegression(C=1.0),
            LogisticRegression(C=0.1)
        ],
    )
    
    References
    ----------
    .. [1] Seung, H. Sebastian, Manfred Opper, and Haim Sompolinsky. "Query by
           committee." Proceedings of the fifth annual workshop on
           Computational learning theory. ACM, 1992.
    """
    
    def __init__(self, *args, **kwargs):
        super(QueryByCommittee, self).__init__(*args, **kwargs)
        
        self.disagreement = kwargs.pop('disagreement', 'vote')
        
        models = kwargs.pop('models', None)
        if models is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
            )
        elif not models:
            raise ValueError("models list is empty")
        
        if self.disagreement == 'kl_divergence':
            for model in models:
                if not isinstance(model, 'ProbabilisticModel'):
                    raise TypeError(
                        "Given disagreemetn set as 'kl_divergence', all models"
                        "must be ProbabilisticModel"
                    )
                    
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = np.random.RandomState(random_state)
        
        self.students = list()
        for model in models:
            if isinstance(model, str):
                self.students.append(getattr(activeLearner.models, model)())
            else:
                self.students.append(model)
        self.n_students = len(self.students)
        self.teach_students()
        
    def _vote_disagreement(self, votes):
        """
        Returns the disagreement measurement of the given number of votes.
        It uses the vote to measure the disagreement. 
        
        Parameters
        ----------
        votes: list of int, shape=(n_samples. n_students)
            The predictions that each student gives to each sample
        
        
        Returns
        -------
        disagreement: list of float, shape=(n_samples)
            The vote entropy of the given votes. 
        """
        disagreement = []
        for candidate in votes:
            disagreement.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1
                
            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                disagreement[-1] -= lab_count[lab] / self.n_students * \
                    math.log(float(lab_count[lab]) / self.n_students)
                    
            return disagreement
        
    def _kl_divergence_disagreement(self, proba):
        """
        Calculates the Kullback-Leibler (KL) divergence disagreement measure.
        
        Parameters
        ----------
        proba: array-like, shape=(n_samples, n_students, n_class)
        
        
        Returns
        -------
        disagreement: list of floats, shape=(n_samples)
            The kl-divergence of the given probability. 
        """
        n_students = np.shape(proba)[1]
        consensus = np.mean(proba, axis=1) # shape=(n_samples, n_students)
        consensus = np.tile(consensus, (n_students, 1, 1)).transpose(1, 0 , 2)
        kl = np.sum(proba * np.log(proba / consensus), axis=2)
        return np.mean(kl, axis=1)
    
    def _labeled_uniform_sample(self, sample_size):
        """Uniformly sample labeled entries"""
        X, y = self.dataset.get_labeled_entries()
        samples_idx = [self.random_state_.randint(0, X.shape[0]) for _ in range(sample_size)]
        return Dataset(X[samples_idx], np.array(y)[samples_idx])
    
    def teach_students(self):
        """
        Train each model (student) with the labeled data using bagging. 
        """
        dataset = self.dataset
        for student in self.students:
            bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
                LOGGER.warning('There is a student receiving only one label,'
                               're-sample the bag.')
            student.train(bag)
            
    @inherit_docstring_from(Query)
    def update(self, entry_id, label):
        # Train each model with newly updated label.
        self.teach_students()
        
    @inherit_docstring_from(Query)
    def make_query(self, top_n=1):
        # TODO Why isn't the model updated when make_query() called
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        
        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            votes = np.zeroes((len(X_pool), len(self.students)))
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)
                
            vote_entropy = self._vote_disagreement(votes)
            ask_idx = self.random_state_.choice(
                np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0],
                size=top_n
            )
            
        elif self.disagreement == 'kl_divergence':
            proba = []
            for student in self.students:
                proba.append(student.predict_proba(X_pool))
            proba = np.array(proba).transpose(1, 0, 2).astype(float)
            
            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = self.random_state_.choice(
                np.where(np.isclose(avg_kl, np.max(avg_kl)))[0],
                size=top_n
            )
            
        return unlabeled_entry_ids[ask_idx]
                              
        
            
#==========================# 
# LABELER CONCRETE CLASSES #
#==========================#

'''
Interactive Labeler

This module includes a means of asking the Oracle for the
correct label.
'''


class AskOracle(Labeler):
    """
    Ask Oracle
    
    AskOracle is a Labeler object that displays the feature which
    needs to be labeled by the Orcle. 
    
    If the feature is an image, it will be shown using matplotlib 
    and lets the Oracle label it through command line interface. 
    If it is a text image, it will be printed via the command line
    and lets the Oracle label it via the command line as well.
    
    Parameters
    ----------
    feature_type: string
        Identifies the data structure of the feature. Must be either
        'text' or 'image'.
    
    label_name: list of strings
        Let the label space be from 0 to len(label_name)-1, this list
        corresponds to each label's name. If label_names are numeric, 
        e.g. 0,1,...,N must be entered as strings '0','1',...
        
    feature_name (optional): string
        The name of the feature being labeled. If provided, will be
        displayed to Oracle ask a reminder when querying them.
    """
    
    def __init__(self, feature_type=None, label_name=None, **kwargs):
        self.feature_type = feature_type
        if label_name is None:
            raise ValueError('Must provided a list of label names of \
                len(label_name)-1')
        self.label_name = label_name
        self.feature_name = kwargs.pop('feature_name', None)
        
    @inherit_docstring_from(Labeler)
    def label(self, feature, **kwargs):
        """
        Queries oracle to provide correct label for given unlabeled example.
        
        Parameters
        ----------
        feature: {array-like} string or rbg channel matrix
            Unlabeled feature from the dataset.
            
        (optional) keywords: list
            If feature_type is 'text', can a provide a list of 
            keywords to highlight when displayed in the console. 
        """
        # Display feature
        if self.feature_type == 'image':
            plt.imshow(feature, cmap=plt.cm.gray, interpolation='nearest')
            plt.draw()
        elif self.feature_type == 'text':
            # Highlight keywords if provided
            keywords = kwargs.pop('keywords', None)
            if keywords is not None:
                for word in keywords:
                    feature = feature.replace(word, '\033[49;32m{}\033[m'.format(word))
            # print
            print('\n' + feature + '\n')
        else:
            raise ValueError("Feature cannot be display. Feature_type must be \
                either 'image' or 'text'")
        
        # Oracle prompt 
        if self.feature_name is not None:
            banner = f"Enter the label name for {self.feature_name} "
        else:
            banner = "Enter the label for the associated the feature "
        banner += str(self.label_name) + ': '    
           
        lbl = input(banner)
        
        while (self.label_name is not None) and (lbl not in self.label_name):
            print('Invalid label, please re-enter the associated label.')
            lbl = input(banner)
            
        return self.label_name.index(lbl)
