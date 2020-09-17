import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='tmma',
    version='0.1.0',
    description='MA plots and TMM for python',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',

    url='https://github.com/lukauskas/tmma',

    author='Saulius Lukauskas',


    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    python_requires='>=3.6, <4',

    install_requires=['numpy>=1.19.2',
                      'pandas>=1.1.2',
                      'matplotlib>=3.3.1',
                      'scipy>=1.5.2'],

    extras_require={  # Optional
        'test': ['hypothesis>=5.35.2',
                 'rpy2>=3.3.5',
                 'sinfo>=0.3.1'],
    },
)