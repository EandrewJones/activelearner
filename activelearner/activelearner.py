"""Main module."""
import warnings

# Main function loop for active_learner algorithm
def run_active_learner(dataset, querier, feature_type, label_name,
                       save_every=20, print_progress=True, **kwargs):
    '''
    Runs main active learning algorithm loop, prompting Oracle for correct label and
    updating dataset. Currently only uses random sampling query strategy. Not yet implemented option for "active" updating via model-based query strategy.

    Parameters
    ----------
    dataset: dataset object
        Must be activelearner dataset object containing features X and labels Y.
        Current verision only supports TextDataset class.

    querier: query strategy object
        Must be activelearner query strategy object of type 'QueryByCommittee',
        'QUIRE', 'RandomSampling', or 'UncertaintySampling'.

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
    assert dataset is not None
    assert model is not None
    assert querier is not None
    assert feature_type in ['text', 'image']
    if feature_type != 'text' and keywords is not None:
        warnings.warn("feature_type is not 'text', keywords will be ignored.")
    if feature_type == 'text' and keywords is not None:
        for word in keywords:
            assert isinstance(word, str)

    for name in label_name:
        assert isinstance(name, str)

    assert isinstance(save_every, int) and save_every > 0
    assert isinstance(print_progress, bool)
    assert isinstance(path, str)
    assert isinstance(seed, int)
    assert isinstance(file_name, str)
    if feature_name is not None:
        assert isinstance(feature_name, str)

    # Instantiate oracle
    oracle = AskOracle(feature_type=feature_type,
                       label_name=label_name,
                       feature_name=feature_name)

    # Model loop
    continue_cycle = True
    while continue_cycle:
        # active learning algorithm loop
        for _ in range(save_every):
            query_id = querier.make_query()
            label = oracle.label(dataset.view[query_id],
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

