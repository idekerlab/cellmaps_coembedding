=======
History
=======

1.3.1 (2025-07-03)
--------------------

* Fix for UD-3116, missing header 0 row

1.3.0 (2025-05-29)
-------------------

* Create flags for parameters of proteingps coembedding algorithm

1.2.2 (2025-05-15)
-------------------

* Updated to PEP 517 compliant build system
* Bug fixes: update constants, add back L2 normalization and fix separator issue for proteinGPS

1.2.1 (2025-04-14)
-------------------

* Fix scipy package version

1.2.0 (2025-03-19)
-------------------

* Added functionality to generate umap of embeddings (in cellmaps_coembedding.utils)

1.1.0 (2025-03-05)
-------------------

* Added functionality to evaluate embeddings using statistical analysis and visualization (functions
  `get_embedding_eval_data` and `generate_embedding_evaluation_figures` in cellmaps_coembedding.utils).

* Update defauls (EPOCHS and DROPOUT)

1.0.0 (2025-01-28)
-------------------

* Rename auto coembedding name and proteinGPS. `--algorithm auto` option is depreacted and `--algorithm proteingps`
  should be used. The coembedding implementation was moved to `ProteinGPSCoEmbeddingGenerator` class and
  `AutoCoEmbeddingGenerator` is deprecated and calls proteingps. The package name was renamed from `autoembed_sc`
  to `proteingps`.

* Removed l2 normalization

* Constants updated in `ProteinGPSCoEmbeddingGenerator` (triplet_margin=0.2) and in proteingps's fit_predict
  (triplet_margin=0.2, lambda_reconstruction=5.0, lambda_triplet=5.0)

* Bug fix: add missing a `.to(device)` call to ensure tensors are correctly moved to the appropriate device.

* Update version bounds of required packages

0.4.0 (2024-12-02)
-------------------

* Added README generation.

* Refactor code.

* Update defaults (EPOCHS)

0.3.1 (2024-09-13)
------------------

* Bug fix: raise more informative error when no embeddings overlap.

0.3.0 (2024-09-06)
------------------

* Added ``--provenance`` flag to pass a path to json file with provenance information. This removes the
  necessity of input directory to be an RO-Crate.

0.2.0 (2024-07-17)
------------------

* Added a new coembedding algorithm accessible via flag ``--algorithm auto``. This algorithm utilizes neural networks
  to generate latent embeddings, optimizing both reconstruction and triplet losses to improve embedding accuracy
  by learning intra- and inter-modality relationships.

0.1.0 (2024-02-12)
------------------

* First release on PyPI.
