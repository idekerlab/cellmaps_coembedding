=====
Usage
=====

The **cellmaps_coembedding** tool takes image and Protein-Protein Interaction (PPI)
embeddings and generates co-embedding. The embeddings can be generated by
`cellmaps_image_embedding <https://cellmaps-image-embedding.readthedocs.io>`__ and
`cellmaps_ppi_embedding <https://cellmaps-ppi-embedding.readthedocs.io>`__ packages.

In a project
--------------

To use cellmaps_coembedding in a project::

    import cellmaps_coembedding


Needed files
------------

The output directories for the image embeddings (see `Cell Maps Image Embedding <https://github.com/idekerlab/cellmaps_image_embedding/>`__) and protein-protein interaction network embeddings (see `Cell Maps PPI Embedding <https://github.com/idekerlab/cellmaps_ppi_embedding/>`__) are required.


On the command line
---------------------

For information invoke :code:`cellmaps_coembeddingcmd.py -h`

**Usage**

.. code-block::

  cellmaps_coembeddingcmd.py [outdir] [--embeddings EMBEDDING_DIR [EMBEDDING_DIR2 ...]] [OPTIONS]

**Arguments**

- ``outdir``
    The directory where the output will be written to.

*Required (choose one)*

- ``--embeddings EMBEDDINGS_DIR``
    Paths to directories containing image and/or PPI embeddings. The directory should have a TSV file, named `image_emd.tsv` or `ppi_emd.tsv`.
    Second option is to provide paths to specific TSV files.

    **Deprecated Flags (still functional but no longer required):**

        - ``--ppi_embeddingdir``
            The directory path created by `cellmaps_ppi_embedding` which has a TSV file containing the embeddings of the PPI network. For each row, the first value is assumed to be the gene symbol followed by the embeddings.

        - ``--image_embeddingdir``
            The directory path created by `cellmaps_image_embedding` which has a TSV file containing the embeddings of the IF images. For each row, the first value is assumed to be the sample ID followed by the embeddings.

*Optional*

- ``--embedding_names``
    Names corresponding to each filepath input in --embeddings.

- ``--algorithm``
    Algorithm to use for coembedding. Choices: 'auto', 'muse', 'proteingps'. Defaults to 'muse'.
    'auto' is deprecated, and 'proteingps' should be used instead.

- ``--latent_dimension``
    Output dimension of the embedding. Default is 128.

- ``--n_epochs_init``
    Number of initial training epochs. Default is 200.

- ``--n_epochs``
    Number of training epochs. Default is 500.

- ``--jackknife_percent``
    Percentage of data to withhold from training. For example, a value of 0.1 means to withhold 10 percent of the data.

- ``--mean_losses``
    If set, use the mean of losses; otherwise, sum the losses.

- ``--dropout``
    Percentage to use for dropout layers in the neural network.

- ``--l2_norm``
    If set, L2 normalize coembeddings.

- ``--fake_embedding``
    If set, generates fake co-embeddings.

- ``--logconf``
    Path to the Python logging configuration file in the specified format.

- ``--verbose`` or ``-v``
    Increases verbosity of the logger to standard error for log messages in this module. Logging levels: `-v` = ERROR, `-vv` = WARNING, `-vvv` = INFO, `-vvvv` = DEBUG, `-vvvvv` = NOTSET.

- ``--version``
    Shows the version of the program.

**Example usage**

.. code-block::

   cellmaps_coembeddingcmd.py ./cellmaps_coembedding_outdir --embeddings ./cellmaps_image_embedding_outdir ./cellmaps_ppi_embedding_outdir

Via Docker
---------------

**Example usage**


.. code-block::

   Coming soon...

