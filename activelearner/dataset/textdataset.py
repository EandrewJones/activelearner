'''
Text Dataset
(concrete class)

This module includes an implementation of the dataset class
for text-type datasets with additional methods for feature engineering.
'''
import numpy as np
import re
import psutil
import scipy.sparse as sp

from scipy.spatial import ConvexHull
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras_preprocessing.text import text_to_word_sequence
from sentence_transformers import SentenceTransformer
from torch import 
from activelearner.interfaces import Dataset


#=========#
# Classes #
#=========#

class TextDataset(Dataset):
    """
    Concrete Dataset class for text features

    Extends functionality of dataset factory class with additional
    methods for text feature engineering. Currently includes methods
    for:
        1) BoW
        2) doc2vec
        3) Spectral NMF
        4) Any embedding model included in the sentence-transformers package

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

        tokenized_docs = self._X.apply(lambda x: text_to_word_sequence(x))
        self._optimal_seq_length = optimal_seq_length(tokenized_docs)

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
        
    def transformer(self, model, **kwargs):
        """
        Transform sample documents (X) into document-level embeddings using pre-trained
        models from the SentenceTransformer module or your own custom sentence-transformer
        model.
        
        Parameters
        ----------
        model: str
            Either the name of a pre-trained sentence transformer model or the path to a local
            pretrained sentence transformer model. Available pre-trained models are listed at
            https://www.sbert.net/docs/pretrained_models.html. The 'best' model will depend
            on use case, and the desired trade off between computational cost and performance.
            
        device: str, (default='cuda')
            By default, the function will search for Cuda-enabled GPUs and use the most powerful
            device available for computation. Otherwise, cpu will be used. If the dataset is 
            large, reaching CPU/memory limits is likely, and the user will be warned.
            
        seq_length: int, (default=128)
            The number of allowed word pieces used for each embedding. Transformer models like 
            BERT / RoBERTa / DistilBERT etc. the runtime and the memory requirement grows quadratic 
            with the input length. A common value for BERT models is 512, which roughly equates to
            300-400 (English) words.
        
        batch_size: int, (default=32)
            Number of documents to process per batch. Small batches use less memory, but will take
            longer. Conversely, larger batches will run faster, but consume larger memory.
            
        show_progress_bar: bool, (default=False)
            Whether to print the progess of encoder.
    
        """
        
        # Argument checks
        if model is None:
            ValueError('model name or path must be provided.')
        device = kwargs.pop('device', None)
        seq_length = kwargs.pop('seq_length', 128)
        batch_size = kwargs.pop('batch_size', 32)
        show_progress_bar = kwargs.pop('show_progress_bar', False)
        
        # Create model
        print('# ================================ #')
        print('Instantiating ' + model + ' model.')
        model = SentenceTransformer(model_name_or_path=model, device=device)
        
        
        # set optimal sequence length
        print('Setting max sequence length to: ' + self._optimal_seq_length)
        model.max_seq_length = self._optimal_seq_length
        
        # Create embeddings
        print('Starting encoder with batch size: ' + batch_size)
        doc_list = self._X.tolist()
        embeddings = model.encode(sentences=doc_list,
                                  batch_size=batch_size,
                                  show_progress_bar=show_progress_bar)
        print('Complete!')
        print('# ================================ #')
        self._X = embeddings


#===========#
# Functions #
#===========#


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


def unique_tokens(corpus):
    """Takes corpus and returns unique tokens and number of unique tokens."""
    uniq_tokens = set()

    for doc in corpus:
        doc = preprocess(doc)
        doc = text_to_word_sequence(doc)
        uniq_tokens.update(doc)

    return uniq_tokens, len(uniq_tokens)


def optimal_seq_length(tokenized_list):
    """Takes tokenized list of documents and returns optimal sequence length."""
    
    # Calculate lengths
    lengths = [len(doc) for doc in tokenized_list]
    quants = np.quantile(lengths, q=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])
    
    # Calculate optimal sequence length
    seq_lenths = [32, 64, 128, 256, 512]
    distances = [[[abs(q - l) for l in seq_lengths] for q in quants]]
    optimal = np.mean(distances, axis=1).argmin()
    
    return(seq_length[optimal])


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


def gensim_read_corpus(docs, tokens_only=False):
    '''reads corpus into gensim format'''
    for i, doc in enumerate(docs):
        doc = preprocess(doc)
        tokens = text_to_word_sequence(doc)
        yield TaggedDocument(tokens, [i])


