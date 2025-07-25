=====================
Cell Maps CoEmbedder
=====================
The Cell Maps CoEmbedding is part of the Cell Mapping Toolkit

.. image:: https://img.shields.io/pypi/v/cellmaps_coembedding.svg
        :target: https://pypi.python.org/pypi/cellmaps_coembedding

.. image:: https://app.travis-ci.com/idekerlab/cellmaps_coembedding.svg?branch=main
        :target: https://app.travis-ci.com/idekerlab/cellmaps_coembedding

.. image:: https://readthedocs.org/projects/cellmaps-coembedding/badge/?version=latest
        :target: https://cellmaps-coembedding.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/620523316.svg
        :target: https://zenodo.org/doi/10.5281/zenodo.10651873
        :alt: Zenodo DOI badge

Creates Coembedding from `Cell Maps ImmunoFluorscent Image Embedder <https://cellmaps-image-embedding.readthedocs.io>`__
and `Cell Maps PPI Embedder <https://cellmaps-ppi-embedding.readthedocs.io>`__ using an implementation of `MUSE <https://github.com/AltschulerWu-Lab/MUSE>`__

* Free software: MIT license
* Documentation: https://cellmaps-coembedding.readthedocs.io.


Dependencies
------------

* `cellmaps_utils <https://pypi.org/project/cellmaps-utils>`__
* `phenograph <https://pypi.org/project/phenograph>`__
* `numpy <https://pypi.org/project/numpy>`__
* `torch <https://pypi.org/project/torch>`__
* `pandas <https://pypi.org/project/pandas>`__
* `matplotlib <https://pypi.org/project/matplotlib>`__
* `dill <https://pypi.org/project/dill>`__
* `tqdm <https://pypi.org/project/tqdm>`__
* `scipy <https://pypi.org/project/scipy/>`__


Compatibility
-------------

* Python 3.8 - 3.11

Installation
------------

.. code-block::

   git clone https://github.com/idekerlab/cellmaps_coembedding
   cd cellmaps_coembedding
   pip install -r requirements_dev.txt
   make dist
   pip install dist/cellmaps_coembedding*whl


Run **make** command with no arguments to see other build/deploy options including creation of Docker image

.. code-block::

   make

Output:

.. code-block::

   clean                remove all build, test, coverage and Python artifacts
   clean-build          remove build artifacts
   clean-pyc            remove Python file artifacts
   clean-test           remove test and coverage artifacts
   lint                 check style with flake8
   test                 run tests quickly with the default Python
   test-all             run tests on every Python version with tox
   coverage             check code coverage quickly with the default Python
   docs                 generate Sphinx HTML documentation, including API docs
   servedocs            compile the docs watching for changes
   testrelease          package and upload a TEST release
   release              package and upload a release
   dist                 builds source and wheel package
   install              install the package to the active Python's site-packages
   dockerbuild          build docker image and store in local repository
   dockerpush           push image to dockerhub

Before running tests, please install ``pip install -r requirements_dev.txt``.

For developers
-------------------------------------------

To deploy development versions of this package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are steps to make changes to this code base, deploy, and then run
against those changes.

#. Make changes

   Modify code in this repo as desired

#. Build and deploy

.. code-block::

    # From base directory of this repo cellmaps_coembedding
    pip uninstall cellmaps_coembedding -y ; make clean dist; pip install dist/cellmaps_coembedding*whl



Needed files
------------

The output directories for the image embeddings (see `Cell Maps Image Embedding <https://github.com/idekerlab/cellmaps_image_embedding/>`__) and protein-protein interaction network embeddings (see `Cell Maps PPI Embedding <https://github.com/idekerlab/cellmaps_ppi_embedding/>`__) are required.


Usage
-----

For information invoke :code:`cellmaps_coembeddingcmd.py -h`

**Example usage**

.. code-block::

   cellmaps_coembeddingcmd.py ./cellmaps_coembedding_outdir --embeddings ./cellmaps_image_embedding_outdir ./cellmaps_ppi_embedding_outdir



Via Docker
~~~~~~~~~~~~~~~~~~~~~~

**Example usage**


.. code-block::

   Coming soon...

Cite
-------

If you find this tool useful, please cite:

Lenkiewicz, J., Churas, C., Hu, M., Qian, G., Jain, M., Levinson, M. A., ... & Schaffer, L. V. (2025). Cell Mapping Toolkit: An end-to-end pipeline for mapping subcellular organization. Bioinformatics, 41(6), btaf205.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _NDEx: http://www.ndexbio.org
