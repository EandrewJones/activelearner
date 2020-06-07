#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements_dev.txt') as f:
    requirements = f.read().splitlines()
     
install_requirements = requirements

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Evan Andrew Jones",
    author_email='evan.a.jones3@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Extensible Active Learning Framework",
    entry_points={
        'console_scripts': [
            'activelearner=activelearner.cli:main',
        ],
    },
    install_requires=_install_requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='activelearner',
    name='activelearner',
    packages=['activelearner', 
              'activelearner.dataset', 
              'activelearner.interfaces', 
              'activelearner.labeler', 
              'activelearner.models', 
              'activelearner.strategies', 
              'activelearner.utils'
              ],
    package_dir={
        'activelearner': 'activelearner',
        'activelearner.dataset': 'activelearner/dataset',
        'activelearner.interfaces': 'activelearner/interfaces',
        'activelearner.labeler': 'activelearner/labeler',
        'activelearner.models': 'activelearner/models',
        'activelearner.strategies': 'activelearner/strategies',
        'activelearner.utils': 'activelearner/utils'
    }
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/EandrewJones/activelearner',
    version='0.1.0',
    zip_safe=False,
)
