"""Main module."""
import warnings
import os
import sys
import re

from six.moves import input

from activelearner import dataset, models, strategies, labeler, utils


# Define active learning labeling modes
def interactive_mode(data, keywords, querier, oracle, save_every, 
                     path, file_name, print_progress):
    '''Implements loop for interactive labeling.'''
    # Start interactive loop that retrains model every iteration
    continue_cycle = True
    while continue_cycle:
        for _ in range(save_every):
            query_id = querier.make_query()
            label = oracle.label(data.view[query_id],
                                 keywords=keywords)
            data.update(query_id, label)

            progress = (_ + 1) / save_every
            if progress % 5 == 0:
                update_progress(progress)

        # Print updated class information
        if print_progress:
            data.get_dataset_stats()

        # Save dataset object
        fname = os.path.join(path, '', file_name)
        utils.save_object(obj=data, filename=fname)

        # ask user to input continue cycle
        banner = f'Would you like to continue labeling another {save_every} examples? [(Y)es/(N)o]: '
        valid_input = set(['Yes', 'Y', 'y', 'yes', 'No', 'N', 'n', 'no'])
        continue_options = set(['Yes', 'Y', 'y', 'yes'])

        user_choice = input(banner)

        while user_choice not in valid_input:
            print(f'Invalid choice. Must be one of {valid_input}')
            user_choice = input(banner)
        continue_cycle = user_choice in continue_options


def batch_mode(data, keywords, querier, oracle, save_every, 
               path, file_name, print_progress):
    '''Implements loop for batch labeling.'''
    # Get batch of ids to query
    query_ids = querier.make_query()
    
    # Query oracle for labels
    for _, query_id in enumerate(query_ids):
        label = oracle.label(data.view[query_id],
                                 keywords=keywords)
        data.update(query_id, label)

        progress = (_ + 1) / len(query_ids)
        if print_progress and progress % save_every == 0:
            # show progress
            update_progress(progress)
            data.get_dataset_stats()
            
            # save progress
            print('Saving progress...')
            fname = os.path.join(path, '', file_name)
            utils.save_object(obj=data, filename=fname)     
    

# Main function for active_learner algorithm
def run_active_learner(mode, data, querier, feature_type, label_name,
                       save_every=20, print_progress=True, **kwargs):
    '''
    Runs main active learning algorithm loop, prompting Oracle for correct label and
    updating dataset. Currently only uses random sampling query strategy. Not yet implemented option for "active" updating via model-based query strategy.

    Parameters
    ----------
    mode: string
        Sets the labeling mode. Currently support 'batch' or 'interactive'.
    
    dataset: dataset object
        Must be activelearner dataset object containing features X and labels Y.
        Current verision only supports TextDataset class.

    querier: query strategy object
        Must be activelearner query strategy object of type 'QueryByCommittee',
        'QUIRE', 'RandomSampling', 'UncertaintySampling', or 'BatchUncertaintySampling.
        `BatchUncertaintySampling` only works with `batch` mode and vice versa.

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
    # Extract optional arguments
    path = kwargs.pop('path', os.getcwd())
    file_name = kwargs.pop('file_name', 'dataset.pkl')
    feature_name = kwargs.pop('feature_name', None)
    keywords = kwargs.pop('keywords', None)
    seed = kwargs.pop('seed', 1)

    # Argument checking
    # TODO implement type checking version of entire packed
    assert mode in ['batch', 'interactive'], "Mode must be one of ['batch', 'interactive']"
    if not isinstance(data, dataset.textdataset.TextDataset) and \
        not isinstance(data, dataset.imagedataset.ImageDataset):
            raise TypeError("data must be of class TextDataset or ImageDataset from dataset submodule.")
    strategy_class = re.compile("activelearner.strategies")
    assert re.search(strategy_class, str(type(querier))), "querier must be of class 'BatchUncertaintySampling', 'UncertaintySampling', 'QUIRE', 'QueryByCommittee' or 'RandomSampling'"
    assert isinstance(save_every, int), "save_every must be a positive integer."
    assert save_every > 0, "save_every must be a positive integer."
    assert isinstance(print_progress, bool), "print_progress must be a boolean."
    assert isinstance(path, str), "path must be a string."
    assert isinstance(seed, int), "seed must be a integer."
    assert isinstance(file_name, str), "file_name must be a string."
    if feature_name is not None:
        assert isinstance(feature_name, str), "feature_name must be a string."
    assert feature_type in ['text', 'image']
    if feature_type != 'text' and keywords is not None:
        warnings.warn("feature_type is not 'text', keywords will be ignored.")
    if feature_type == 'text' and keywords is not None:
        for word in keywords:
            assert isinstance(word, str)

    for name in label_name:
        assert isinstance(name, str), "label_name must be a string or list of strings."

    # Instantiate oracle
    oracle = labeler.AskOracle(feature_type=feature_type,
                               label_name=label_name,
                               feature_name=feature_name)

    # Run interactive model
    if mode == 'interactive':
        interactive_mode(data=data, keywords=keywords, querier=querier,
                         oracle=oracle, save_every=save_every, path=path,
                         file_name=file_name, print_progress=print_progress)
    if mode == 'batch':
        batch_mode(data=data, keywords=keywords, querier=querier,
                   oracle=oracle, save_every=save_every, path=path,
                   file_name=file_name, print_progress=print_progress)