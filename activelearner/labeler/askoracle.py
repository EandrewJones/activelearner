"""
Labeler

This module incluedes a concrete class interface for asking the
oracle for the correct label
"""

import matplotlib.pyplot as plt
from six.moves import input

from activelearner.interfaces import Labeler
from activelearner.utils import inherit_docstring_from

'''
Interactive Labeler
(Concrete class)

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
