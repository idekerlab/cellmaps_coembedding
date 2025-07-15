=====
Usage
=====

This tool generates a co-embedding from input embeddings using either the `MuSE` algorithm or `ProteinGPS`.
The co-embedding is constructed from two or more input embedding datasets, such as protein-protein interaction (PPI)
embeddings (`cellmaps_ppi_embedding <https://cellmaps-ppi-embedding.readthedocs.io>`__) or image embeddings
(`cellmaps_image_embedding <https://cellmaps-image-embedding.readthedocs.io>`__).

By default, it uses the `muse` algorithm unless specified otherwise.

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
    Output directory where all results and intermediate files will be saved.

*Required*

- ``--embeddings EMBEDDINGS_DIR``
    Paths to embedding files. Either directories containing image and/or PPI embeddings that contain a TSV file,
    named `image_emd.tsv` or `ppi_emd.tsv` or paths to specific TSV files.

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
    If set, performs L2 normalization on coembeddings (ProteinGPS only).

- ``--lambda_triplet``
    Weight for triplet loss (ProteinGPS only). (Default: 1.0)

- ``--mean_losses``
    If set, uses the mean of the loss functions instead of the sum (ProteinGPS only).

- ``--batch_size``
    Batch size for training (ProteinGPS only). (Default: 16)

- ``--triplet_margin``
    Margin for triplet loss (ProteinGPS only). (Default: 1.0)

- ``--learn_rate``
    Learning rate for the optimizer (ProteinGPS only). (Default: 1e-4)

- ``--hidden_size_1``
    Size of the first hidden layer in the neural network (ProteinGPS only). (Default: 512)

- ``--hidden_size_2``
    Size of the second hidden layer in the neural network (ProteinGPS only). (Default: 256)

- ``--save_update_epochs``
    If set, saves the model state at specified epoch intervals (ProteinGPS only).

- ``--negative_from_batch``
    If set, uses negative samples from the same batch for triplet loss (ProteinGPS only).

- ``--fake_embedding``
    If set, generates fake co-embeddings.

- ``--provenance``
    Path to a JSON file containing provenance information for the input files. Required if the embedding directories do not contain `ro-crate-metadata.json`.

- ``--name``
    Name of the run (used for FAIRSCAPE). If unset, value is inferred from the embedding directory or provenance.

- ``--organization_name``
    Name of the organization running this tool (used for FAIRSCAPE).

- ``--project_name``
    Name of the project running this tool (used for FAIRSCAPE).

- ``--logconf``
    Path to a Python logging configuration file. Overrides `-v`. (Default: None)

- ``--skip_logging``
    If set, disables creation of `output.log` and `error.log`.

- ``--verbose``, ``-v``
    Increases logging verbosity:
    - `-v` = WARNING
    - `-vv` = INFO
    - `-vvv` = DEBUG
    - `-vvvv` = NOTSET

- ``--version``
    Displays the version of the tool and exits.

**Example usage**

CM4AI Data Example:

.. code-block::

   cellmaps_coembeddingcmd.py ./cellmaps_coembedding_outdir --embeddings ./cellmaps_image_embedding_outdir ./cellmaps_ppi_embedding_outdir


Another example:

.. code-block:: bash

    cellmaps_coembeddingcmd.py my_output_dir \
        --embeddings ppi_emd.tsv image_emd.tsv \
        --embedding_names ppi image \
        --algorithm proteingps \
        --latent_dimension 128 \
        --n_epochs 300 \
        --jackknife_percent 0.2 \
        --l2_norm \
        --provenance metadata.json

Via Docker
---------------

**Example usage**


.. code-block::

   Coming soon...

Embedding Evaluation (additional functionality)
------------------------------------------------

The `cellmaps_coembedding.utils` module provides functions for evaluating embeddings. It is not part of the standard workflow,
but an additional functionality. It includes statistical analysis of similarity scores and
visualization of embedding performance using enrichment tests.

The `get_embedding_eval_data` function computes enrichment effect sizes for various embeddings using a reference
adjacency matrix (CORUM). It also saves KDE data for the MUSE embedding. The `generate_embedding_evaluation_figures`
automates the evaluation process by loading embeddings, computing effect sizes, and generating figures.

**Returns:**

- `sim_muse_data.csv`: MUSE similarity scores.

- `embedding_eval.csv`: Enrichment effect sizes for each embedding.

- `sim_muse.png`: KDE plot for similarity scores.

- `embedding_eval.png`: Enrichment comparison plot.

**Usage Example**

.. code-block::

    from cellmaps_coembedding.utils import generate_embedding_evaluation_figures

    generate_embedding_evaluation_figures(
        coembedding='/path/to/coembedding',
        ppi='/path/to/ppiembedding',
        image='/path/to/imageembedding',
        outdir='/output/directory',
        num_samplings=1000,
        num_edges=1000
    )


**UMAP Generation**

Optionally, you can create UMAP visualizations of the generated embeddings by using the ``cellmaps_coembedding.utils``
helpers. These plots allow you to see how samples cluster in a 2D projection based on their embedding similarity.

.. note::
   To generate UMAP plots, you need to have the ``umap-learn`` (often installed as ``umap`` or ``umap-learn``) and ``seaborn`` Python packages installed. For example, you can install them via::

     pip install umap-learn seaborn


.. code-block::

    from cellmaps_coembedding.utils import generate_umap_of_embedding

    generate_umap_of_embedding(emb_file='/path/to/embedding', outdir='/output/directory')

If you want to color the UMAP based on label (for example localization of the protein in the cell), you can pass a
directory that contains label to protein mapping in ``label_map`` argument.

.. code-block::

    from cellmaps_coembedding.utils import generate_umap_of_embedding

    generate_umap_of_embedding(emb_file='/path/to/embedding', outdir='/output/directory', label_map=location_dict)
