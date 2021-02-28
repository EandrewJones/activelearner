======================================================
ActiveLearner: An Extensible Active Learning Framework
======================================================

.. image:: https://img.shields.io/pypi/v/activelearner.svg
        :target: https://pypi.python.org/pypi/activelearner

.. 
   image:: https://img.shields.io/travis/EandrewJones/activelearner.svg
        :target: https://travis-ci.com/EandrewJones/activelearner

.. image:: https://readthedocs.org/projects/activelearner/badge/?version=latest
        :target: https://activelearner.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Introduction
------------

ActiveLearner is a python library for active learning written with the aim of balancing extensibility and an opiniated API. 
The intent is for inexperienced users to be able to rely heavily on sensible defaults, yet savvy users can implement
bespoke models and feature design. The package implements popular active learning strategies such as query by committee, 
uncertainity sampling, and batch sampling. It is currently tailored toward text data, with built-in defaults for 
multilingual sentence embeddings (BERT / RoBERTa / XLM-RoBERTa / etc.) from the UKP Lab's 
sentence-transformer_ library. However, a framework is in place
to extend to image / video data. Contributions on this front are welcome.

The library was inspired heavily by the libact_ package from National Taiwan University's Computational Learning Lab.
A deep study and re-implementaiton of this code base exposed me to software design patterns and helped me grok certain active-learning algorithms.
Many thanks to Yao-Yuan Yang, Shao-Chuan Lee, Yu-An Chung, Tung-En Wu, Si-An Chen, and Hsuan-Tien Lin.

* Free software: Apache Software License 2.0
* Documentation: https://activelearner.readthedocs.io.


Installation
------------

Dependencies:

* Python (>= 3.7)
* **PyTorch 1.6.0** or higher
* **transformers v3.1.0** or higher
* The code **does not work with Python 2.7**

The deep learning models, and especially the transformers, can be very computationally expensive. 
*GPU-enabled compute is recommended*. If you want to use a GPU / CUDA, you must install PyTorch with 
the matching CUDA Version. `Follow PyTorch - Get Started`_ for further details how to install 
cuda-enabled PyTorch.

To install:

1. Create a virtual environment with python >= 3.7
2. Run the following commands

.. code-block:: console

    $ git clone https://github.com/EandrewJones/activelearner.git
    $ cd activelearner
    $ pip install -r requirements.txt


Features
--------

* Pool-based active learning
* popular strategies such as query-by-committee, uncertainty sampling, and batch uncertainty sampling
* Built-in text pre-processing including transformer embeddings


Credits
-------

:Author:
        Evan Jones

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _sentence-transformer: https://github.com/UKPLab/sentence-transformers
.. _libact: https://github.com/ntucllab/libact
.. _`Follow PyTorch - Get Started`: https://pytorch.org/get-started/locally/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
