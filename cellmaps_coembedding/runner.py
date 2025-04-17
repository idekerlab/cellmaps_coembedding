#! /usr/bin/env python

import os
import csv
import random
import logging
import time
from datetime import date
import pandas as pd
import numpy as np
import dill
import sys
from tqdm import tqdm
from cellmaps_utils import constants
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil
import cellmaps_coembedding
import cellmaps_coembedding.muse_sc as muse
import cellmaps_coembedding.protein_gps as proteingps
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingGenerator(object):
    """
    Base class for implementations that generate
    network embeddings
    """
    LATENT_DIMENSIONS = 128
    N_EPOCHS = 100
    JACKKNIFE_PERCENT = 0.0
    DROPOUT = 0.5

    def __init__(self, dimensions=LATENT_DIMENSIONS,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 embeddings=None,
                 embedding_names=None):
        """
        Constructor
        """
        self._dimensions = dimensions
        self._embedding_names = embedding_names
        self._initialize_embeddings(embeddings, ppi_embeddingdir, image_embeddingdir)

    def _initialize_embeddings(self, embeddings, ppi_embeddingdir, image_embeddingdir):
        """
        Initializes the embedding locations based on the provided inputs.

        :param embeddings: A list of paths to embedding files or directories with tsv files.
        :type embeddings: list[str] or None
        :param ppi_embeddingdir: The directory path where PPI (Protein-Protein Interaction) embeddings are stored.
        :type ppi_embeddingdir: str or None
        :param image_embeddingdir: The directory path where image embeddings are stored.
        :type image_embeddingdir: str or None
        :raises CellmapsCoEmbeddingError: If both embeddings and flags ppi_embeddingdir or image_embeddingdir
                                          are provided, an error is raised to prevent ambiguity.
        """
        if embeddings is not None and len(embeddings) < 2:
            raise CellmapsCoEmbeddingError(f'Coembedding generator requires at least two embeddings. '
                                           f'Provide at least two files or directories in embedding parameter or '
                                           f'both ppi_embeddingdir and image_embeddingdir')
        if (ppi_embeddingdir or image_embeddingdir) and embeddings:
            raise CellmapsCoEmbeddingError('Use either ppi_embeddingdir and image_embeddingdir or embeddings, '
                                           'not both')
        self._embeddings = embeddings if embeddings is not None else [ppi_embeddingdir, image_embeddingdir]

    def _get_embedding_file_and_name(self, embedding_path):
        """
        Get the embedding file path and its default name based on the given path. If the path is a file,
        it extracts the name from the file name. If the path is a directory, it looks for predefined PPI or image
        embedding file names within this directory.

        :param embedding_path: The path to the embedding file or directory containing the embedding file.
        :type embedding_path: str
        :return: A tuple containing the path to the embedding file and a default name.
        :rtype: tuple[str, str]
        :raises CellmapsCoEmbeddingError: If no embedding file is found in the provided directory path.
        """
        if os.path.isfile(embedding_path):
            name = os.path.basename(embedding_path).split('.')[0]
            return embedding_path, name

        path_ppi = os.path.join(embedding_path,
                                constants.PPI_EMBEDDING_FILE)
        if os.path.exists(path_ppi):
            return path_ppi, 'PPI'
        path_image = os.path.join(embedding_path,
                                  constants.IMAGE_EMBEDDING_FILE)
        if os.path.exists(path_image):
            return path_image, 'image'
        raise CellmapsCoEmbeddingError(f'Embedding file not found in {embedding_path}')

    def _get_embedding_files_and_names(self, embedding_paths, embedding_names=None):
        """
        Retrieves the embedding file paths and their corresponding names based on the provided list of filenames or
        directories. If user supplies names, these replace the default names derived from the files.

        :param embedding_paths: A list of file paths or directories from which to retrieve embedding file paths.
        :type embedding_paths: list
        :param embedding_names: Optional. A list of names supplied by the user.
        :type embedding_names: list or None
        :return: A tuple of two lists: the first containing embedding file paths,
                                        and the second containing corresponding unique names.
        :rtype: (list, list)
        :raises CellmapsCoEmbeddingError: If the number of user-supplied names does not match
                                        the number of embedding file paths.
        """
        embeddings = []
        names = []

        for filepath in embedding_paths:
            embedding_file, embedding_name = self._get_embedding_file_and_name(filepath)
            embeddings.append(embedding_file)
            names.append(embedding_name)

        if embedding_names:  # if user supplied names, replace default
            names = embedding_names
        if len(names) != len(embeddings):
            raise CellmapsCoEmbeddingError('Input list of embedding names does not match number of embeddings.')

        names = self._fix_duplicate_names(names)

        return embeddings, names

    def _fix_duplicate_names(self, names):
        """
        Ensures that each name in the provided list is unique by appending a sequential number to duplicate names.

        :param names: A list of names.
        :type names: list
        :return: unique_names: A list of names where duplicates have been made unique by appending a sequential number.
        :rtype: list
        """
        counts = {}
        unique_names = []
        for name in names:
            if name in counts:
                counts[name] += 1
                unique_names.append('{}_{}'.format(name, counts[name]))
            else:
                counts[name] = 0
                unique_names.append(name)
        return unique_names

    def get_embedding_inputdirs(self):
        """
        Determines the input directories for embeddings by extracting the directory path from each embedding file path.
        If the path is already a directory, it's returned as is.

        :return: A list of directory paths for each embedding, derived from the embedding file paths.
        :rtype: list
        """
        return [os.path.dirname(file) if not os.path.isdir(file) else file for file in self._embeddings]

    def _get_set_of_gene_names(self, embedding):
        """
        Get a set of gene names from **embedding**

        :param embedding:
        :return:
        """
        name_set = set()
        for entry in embedding:
            name_set.add(entry[0])
        return name_set

    def _get_embeddings_from_file(self, embedding_file):
        """
        Gets embedding as a list or lists

        :param embedding_file: Path to embedding file
        :type embedding_file: str
        :return: embeddings
        :rtype: list
        """
        embeddings = []
        with open(embedding_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                embeddings.append(row)
        return embeddings

    def _get_embeddings_and_names(self):
        """
        Gets a list of embeddings and a list of their names. It retrieves the file paths and names,
        and then loads the actual embedding data from those files.

        :return: A tuple where the first element is a list of embeddings, and the second element is a list of names.
        :rtype: list, list
        """
        embeddings = []
        embedding_files, names = self._get_embedding_files_and_names(self._embeddings, self._embedding_names)
        for file in embedding_files:
            embeddings.append(self._get_embeddings_from_file(file))

        return embeddings, names

    def get_dimensions(self):
        """
        Gets number of dimensions this embedding will generate

        :return: number of dimensions aka vector length
        :rtype: int
        """
        return self._dimensions

    def get_next_embedding(self):
        """
        Generator method for getting next embedding.
        Caller should implement with ``yield`` operator

        :raises: NotImplementedError: Subclasses should implement this
        :return: Embedding
        :rtype: list
        """
        raise NotImplementedError('Subclasses should implement')


class ProteinGPSCoEmbeddingGenerator(EmbeddingGenerator):
    """
    Generates co-embedding using proteingps
    """

    def __init__(self, dimensions=EmbeddingGenerator.LATENT_DIMENSIONS,
                 outdir=None,
                 embeddings=None,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 embedding_names=None,
                 lambda_reconstruction=5.0,
                 lambda_triplet=5.0,
                 lambda_l2=0.001,
                 jackknife_percent=EmbeddingGenerator.JACKKNIFE_PERCENT,
                 n_epochs=EmbeddingGenerator.N_EPOCHS,
                 save_update_epochs=True,
                 batch_size=16,
                 triplet_margin=0.2,
                 dropout=EmbeddingGenerator.DROPOUT,
                 l2_norm=False,
                 mean_losses=False
                 ):
        """
        Initializes the ProteinGPSCoEmbeddingGenerator.

        :param dimensions: The dimensionality of the embedding space (default: 128).
        :param outdir: The output directory where embeddings should be saved.
        :param embeddings: Embedding data.
        :param ppi_embeddingdir: Directory containing protein-protein interaction embeddings.
        :param image_embeddingdir: Directory containing image embeddings.
        :param embedding_names: List of names corresponding to each type of embedding provided.
        :param jackknife_percent: Percentage of data to withhold from training as a method of resampling (default: 0).
        :param n_epochs: Number of epochs for which the model trains (default: 250).
        :param save_update_epochs: Boolean indicating whether to save embeddings at regular epoch intervals.
        :param batch_size: Number of samples per batch during training (default: 16).
        :param triplet_margin: The margin value for the triplet loss during training (default: 1.0).
        :param dropout: The dropout rate between layers in the neural network (default: 0).
        :param l2_norm: If true, L2 normalize coembeddings
        """
        super().__init__(dimensions=dimensions, embeddings=embeddings,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir,
                         embedding_names=embedding_names
                         )
        self._outdir = outdir
        self.triplet_margin = triplet_margin
        self._dropout = dropout
        self._l2_norm = l2_norm
        self._n_epochs = n_epochs
        self._save_update_epochs = save_update_epochs
        self._batch_size = batch_size
        self._jackknife_percent = jackknife_percent
        self._mean_losses = mean_losses

    def get_next_embedding(self):
        """
        Iteratively generates embeddings by fitting the proteingps to the current data set.

        :return: Yields the next embedding, produced by the proteingps embedder's fit_predict method.
        """
        embeddings, embedding_names = self._get_embeddings_and_names()

        for index in np.arange(len(embeddings)):
            e = embeddings[index]
            e.sort(key=lambda x: x[0])
            logger.info('There are ' + str(len(e)) + ' ' + embedding_names[index] + ' embeddings')

        embedding_gene_names = [self._get_set_of_gene_names(x) for x in embeddings]
        unique_name_set = np.unique([item for sublist in embedding_gene_names for item in sublist])

        logger.info('There are ' +
                    str(len(unique_name_set)) +
                    ' total proteins')

        resultsdir = os.path.join(self._outdir, 'proteingps')

        for embedding in proteingps.fit_predict(resultsdir=resultsdir,
                                                modality_data=embeddings,
                                                modality_names=embedding_names,
                                                lambda_reconstruction=self.lambda_reconstruction,
                                                lambda_triplet=self.lambda_triplet,
                                                lambda_l2=self.lambda_l2,
                                                latent_dim=self.get_dimensions(),
                                                n_epochs=self._n_epochs,
                                                batch_size=self._batch_size,
                                                save_update_epochs=self._save_update_epochs,
                                                dropout=self._dropout,
                                                l2_norm=self._l2_norm,
                                                mean_losses=self._mean_losses):
            yield embedding


class MuseCoEmbeddingGenerator(EmbeddingGenerator):
    """
    Generats co-embedding using MUSE
    """
    N_EPOCHS_INIT = 100

    def __init__(self, dimensions=EmbeddingGenerator.LATENT_DIMENSIONS,
                 k=10, triplet_margin=0.1,
                 dropout=EmbeddingGenerator.DROPOUT, n_epochs=EmbeddingGenerator.N_EPOCHS,
                 n_epochs_init=N_EPOCHS_INIT,
                 outdir=None,
                 embeddings=None,
                 ppi_embeddingdir=None,
                 image_embeddingdir=None,
                 embedding_names=None,
                 jackknife_percent=EmbeddingGenerator.JACKKNIFE_PERCENT,
                 ):
        """

        :param dimensions:
        :param k: k nearest neighbors value used for clustering - clustering used for triplet loss
        :param triplet_margin: margin for triplet loss
        :param dropout: dropout between neural net layers
        :param n_epochs: training epochs
        :param n_epochs_init: initialization training epochs
        :param outdir:
        :param ppi_embeddingdir:
        :param image_embeddingdir:
        :param jackknife_percent: percent of data to withhold from training
        """
        super().__init__(dimensions=dimensions, embeddings=embeddings,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir,
                         embedding_names=embedding_names
                         )
        self._outdir = outdir
        self._k = k
        self.triplet_margin = triplet_margin
        self._dropout = dropout
        self._n_epochs = n_epochs
        self._n_epochs_init = n_epochs_init
        self._jackknife_percent = jackknife_percent

    def get_next_embedding(self):
        """

        :return:
        """
        embeddings, embedding_names = self._get_embeddings_and_names()
        if len(embeddings) > 2:
            raise CellmapsCoEmbeddingError('Currently, only two embeddings are supported with MUSE coembedding option')

        for index in np.arange(len(embeddings)):
            e = embeddings[index]
            e.sort(key=lambda x: x[0])
            logger.info('There are ' + str(len(e)) + ' ' + embedding_names[index] + ' embeddings')

        embedding_name_sets = [self._get_set_of_gene_names(x) for x in embeddings]
        intersection_name_set = embedding_name_sets[0].intersection(embedding_name_sets[1])

        logger.info('There are ' +
                    str(len(intersection_name_set)) +
                    ' overlapping embeddings')

        if len(intersection_name_set) == 0:
            logger.error('There are no overlapping embeddings. Cannot perform coembedding.')
            raise CellmapsCoEmbeddingError('There are no overlapping embeddings. Cannot perform coembedding.')

        name_index = [x[0] for x in embeddings[0] if x[0] in intersection_name_set]

        embedding_data = []
        for e in embeddings:
            embedding_data.append(
                np.array([np.array([float(v) for v in xi[1:]]) for xi in e if xi[0] in intersection_name_set]))

        resultsdir = os.path.join(self._outdir, 'muse')

        test_subset = random.sample(list(np.arange(len(name_index))), int(self._jackknife_percent * len(name_index)))
        if self._jackknife_percent > 0:
            with open('{}_test_genes.txt'.format(resultsdir), 'w') as file:
                file.write('\n'.join(np.array(name_index)[test_subset]))

        model, res_embedings = muse.muse_fit_predict(resultsdir=resultsdir,
                                                     modality_data=embedding_data,
                                                     modality_names=embedding_names,
                                                     name_index=name_index,
                                                     test_subset=test_subset,
                                                     latent_dim=self.get_dimensions(),
                                                     n_epochs=self._n_epochs,
                                                     n_epochs_init=self._n_epochs_init,
                                                     triplet_margin=self.triplet_margin,
                                                     k=self._k, dropout=self._dropout)
        for index, embedding in enumerate(res_embedings):
            row = [name_index[index]]
            row.extend(embedding)
            yield row


class AutoCoEmbeddingGenerator(ProteinGPSCoEmbeddingGenerator):
    """
    Generates co-embedding using proteingps

    .. deprecated:: 1.0.0
       The embedding was renamed to proteingps. This class is now called ProteinGPSCoEmbeddingGenerator.
    """

    def __init__(self, dimensions=EmbeddingGenerator.LATENT_DIMENSIONS, outdir=None, embeddings=None,
                 ppi_embeddingdir=None, image_embeddingdir=None, embedding_names=None,
                 jackknife_percent=EmbeddingGenerator.JACKKNIFE_PERCENT, n_epochs=EmbeddingGenerator.N_EPOCHS,
                 save_update_epochs=True, batch_size=16, triplet_margin=0.2, dropout=EmbeddingGenerator.DROPOUT,
                 l2_norm=False, mean_losses=False):
        super().__init__(dimensions, outdir, embeddings, ppi_embeddingdir, image_embeddingdir, embedding_names,
                         jackknife_percent, n_epochs, save_update_epochs, batch_size, triplet_margin, dropout, l2_norm,
                         mean_losses)


class FakeCoEmbeddingGenerator(EmbeddingGenerator):
    """
    Generates a fake coembedding for intersection of embedding dirs
    """

    def __init__(self, dimensions=EmbeddingGenerator.LATENT_DIMENSIONS, ppi_embeddingdir=None,
                 image_embeddingdir=None, embeddings=None, embedding_names=None):
        """
        Constructor
        :param dimensions:
        """
        super().__init__(dimensions=dimensions,
                         ppi_embeddingdir=ppi_embeddingdir,
                         image_embeddingdir=image_embeddingdir,
                         embeddings=embeddings,
                         embedding_names=embedding_names)

    def get_next_embedding(self):
        """
        Gets next embedding

        :return:
        """
        embeddings, embedding_names = self._get_embeddings_and_names()
        for index in np.arange(len(embeddings)):
            e = embeddings[index]
            e.sort(key=lambda x: x[0])
            logger.info('There are ' + str(len(e)) + ' ' + embedding_names[index] + ' embeddings')

        name_sets = [self._get_set_of_gene_names(x) for x in embeddings]
        intersection_name_set = name_sets[0].intersection(name_sets[1])

        logger.info('There are ' +
                    str(len(intersection_name_set)) +
                    ' overlapping embeddings')

        for embed_name in intersection_name_set:
            row = [embed_name]
            row.extend([random.random() for x in range(0, self.get_dimensions())])
            yield row


class CellmapsCoEmbedder(object):
    """
    Class to run algorithm
    """

    def __init__(self, outdir=None,
                 inputdirs=None,
                 embedding_generator=None,
                 name=None,
                 organization_name=None,
                 project_name=None,
                 provenance_utils=ProvenanceUtil(),
                 skip_logging=True,
                 input_data_dict=None,
                 provenance=None):
        """
        Constructor

        :param outdir: Directory to write the results of this tool
        :type outdir: str
        :param inputdir: Output directory where embeddings to be coembedded are located
                         (output of cellmaps_image_embedding and cellmaps_ppi_embedding)
        :type inputdir: str
        :param embedding_generator:
        :param skip_logging: If ``True`` skip logging, if ``None`` or ``False`` do NOT skip logging
        :type skip_logging: bool
        :param name:
        :type name: str
        :param organization_name:
        :type organization_name: str
        :param project_name:
        :type project_name: str
        :param input_data_dict:
        :type input_data_dict: dict
        """
        if outdir is None:
            raise CellmapsCoEmbeddingError('outdir is None')
        self._outdir = os.path.abspath(outdir)
        self._start_time = int(time.time())
        self._end_time = -1
        self._name = name
        self._project_name = project_name
        self._organization_name = organization_name
        self._provenance_utils = provenance_utils
        self._provenance = provenance
        self._keywords = None
        self._description = None
        self._embedding_generator = embedding_generator
        self._inputdirs = inputdirs
        self._input_data_dict = input_data_dict
        self._softwareid = None
        self._coembedding_id = None
        self._inputdir_is_rocrate = None

        if skip_logging is None:
            self._skip_logging = False
        else:
            self._skip_logging = skip_logging

        if self._input_data_dict is None:
            self._input_data_dict = {'outdir': self._outdir,
                                     'inputdirs': self._inputdirs,
                                     'embedding_generator': str(self._embedding_generator),
                                     'name': self._name,
                                     'project_name': self._project_name,
                                     'organization_name': self._organization_name,
                                     'skip_logging': self._skip_logging,
                                     'provenance': str(self._provenance)
                                     }

        logger.debug('In constructor')

    def _get_embedding_dirs(self, embeddings):
        dirs = []
        for embed in embeddings:
            if os.path.isfile(embed):
                dirs.append(os.path.dirname(embed))
            else:
                dirs.append(embed)

        return dirs

    def _update_provenance_fields(self):
        """

        :return:
        """
        rocrate_dirs = []
        if self._inputdirs is not None:
            for embeddind_dir in self._inputdirs:
                if os.path.exists(os.path.join(embeddind_dir, constants.RO_CRATE_METADATA_FILE)):
                    rocrate_dirs.append(embeddind_dir)
        if len(rocrate_dirs) > 0:
            prov_attrs = self._provenance_utils.get_merged_rocrate_provenance_attrs(rocrate_dirs,
                                                                                    override_name=self._name,
                                                                                    override_project_name=
                                                                                    self._project_name,
                                                                                    override_organization_name=
                                                                                    self._organization_name,
                                                                                    extra_keywords=['merged embedding'])

            self._name = prov_attrs.get_name()
            self._organization_name = prov_attrs.get_organization_name()
            self._project_name = prov_attrs.get_project_name()
            self._keywords = prov_attrs.get_keywords()
            self._description = prov_attrs.get_description()
        elif self._provenance is not None:
            self._name = self._provenance['name'] if 'name' in self._provenance else 'Coembedding'
            self._organization_name = self._provenance['organization-name'] \
                if 'organization-name' in self._provenance else 'NA'
            self._project_name = self._provenance['project-name'] \
                if 'project-name' in self._provenance else 'NA'
            self._keywords = self._provenance['keywords'] if 'keywords' in self._provenance else ['coembedding']
            self._description = self._provenance['description'] if 'description' in self._provenance else \
                'Coembedding of multiple embeddings'
        else:
            self._name = 'Coembedding tool'
            self._organization_name = 'NA'
            self._project_name = 'NA'
            self._keywords = ['coembedding']
            self._description = 'Coembedding of multiple embeddings'
            logger.warning("One of input directories should be ro-crate, or provenance file should be provided.")

    def _write_task_start_json(self):
        """
        Writes task_start.json file with information about
        what is to be run

        """
        data = {}

        if self._input_data_dict is not None:
            data['commandlineargs'] = self._input_data_dict

        logutils.write_task_start_json(outdir=self._outdir,
                                       start_time=self._start_time,
                                       version=cellmaps_coembedding.__version__,
                                       data=data)

    def _create_rocrate(self):
        """
        Creates rocrate for output directory

        :raises CellMapsProvenanceError: If there is an error
        """
        try:
            self._provenance_utils.register_rocrate(self._outdir,
                                                    name=self._name,
                                                    organization_name=self._organization_name,
                                                    project_name=self._project_name,
                                                    description=self._description,
                                                    keywords=self._keywords)
        except TypeError as te:
            raise CellmapsCoEmbeddingError('Invalid provenance: ' + str(te))
        except KeyError as ke:
            raise CellmapsCoEmbeddingError('Key missing in provenance: ' + str(ke))

    def _register_software(self):
        """
        Registers this tool

        :raises CellMapsImageEmbeddingError: If fairscape call fails
        """
        software_keywords = self._keywords
        software_keywords.extend(['tools', cellmaps_coembedding.__name__])
        software_description = self._description + ' ' + \
                               cellmaps_coembedding.__description__
        self._softwareid = self._provenance_utils.register_software(self._outdir,
                                                                    name=cellmaps_coembedding.__name__,
                                                                    description=software_description,
                                                                    author=cellmaps_coembedding.__author__,
                                                                    version=cellmaps_coembedding.__version__,
                                                                    file_format='py',
                                                                    keywords=software_keywords,
                                                                    url=cellmaps_coembedding.__repo_url__)

    def _register_computation(self):
        """
        # Todo: added inused dataset, software and what is being generated
        :return:
        """
        logger.debug('Getting id of input rocrate')
        used_dataset = []
        for entry in self._inputdirs:
            if os.path.exists(os.path.join(entry, constants.RO_CRATE_METADATA_FILE)):
                used_dataset.append(self._provenance_utils.get_id_of_rocrate(entry))

        keywords = self._keywords
        keywords.extend(['computation'])
        description = self._description + ' run of ' + cellmaps_coembedding.__name__

        self._provenance_utils.register_computation(self._outdir,
                                                    name=cellmaps_coembedding.__computation_name__,
                                                    run_by=str(self._provenance_utils.get_login()),
                                                    command=str(self._input_data_dict),
                                                    description=description,
                                                    keywords=keywords,
                                                    used_software=[self._softwareid],
                                                    used_dataset=used_dataset,
                                                    generated=[self._coembedding_id])

    def _register_image_coembedding_file(self):
        """
        Registers coembedding file with create as a dataset

        """
        description = self._description
        description += ' Co-Embedding file'
        keywords = self._keywords
        keywords.extend(['file'])
        data_dict = {'name': os.path.basename(self.get_coembedding_file()) + ' coembedding output file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'tsv',
                     'author': cellmaps_coembedding.__name__,
                     'version': cellmaps_coembedding.__version__,
                     'schema': 'https://raw.githubusercontent.com/fairscape/cm4ai-schemas/main/v0.1.0/cm4ai_schema_coembedding.json',
                     'date-published': date.today().strftime(self._provenance_utils.get_default_date_format_str())}
        self._coembedding_id = self._provenance_utils.register_dataset(self._outdir,
                                                                       source_file=self.get_coembedding_file(),
                                                                       data_dict=data_dict,
                                                                       skip_copy=True)

    def get_coembedding_file(self):
        """
        Gets image embedding file
        :return:
        """
        return os.path.join(self._outdir, constants.CO_EMBEDDING_FILE)

    def generate_readme(self):
        description = getattr(cellmaps_coembedding, '__description__', 'No description provided.')
        version = getattr(cellmaps_coembedding, '__version__', '0.0.0')

        with open(os.path.join(os.path.dirname(__file__), 'readme_outputs.txt'), 'r') as f:
            readme_outputs = f.read()

        readme = readme_outputs.format(DESCRIPTION=description, VERSION=version)
        with open(os.path.join(self._outdir, 'README.txt'), 'w') as f:
            f.write(readme)

    def run(self):
        """
        Runs CM4AI Generate COEMBEDDINGS


        :return:
        """
        logger.debug('In run method')
        exitcode = 99
        try:
            if self._outdir is None:
                raise CellmapsCoEmbeddingError('outdir must be set')

            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir, mode=0o755)

            if self._skip_logging is False:
                logutils.setup_filelogger(outdir=self._outdir,
                                          handlerprefix='cellmaps_coembedding')
            self._write_task_start_json()

            self.generate_readme()

            if self._inputdirs is None:
                raise CellmapsCoEmbeddingError('No embeddings provided')

            self._update_provenance_fields()
            self._create_rocrate()
            self._register_software()

            # generate result
            with open(os.path.join(self._outdir, constants.CO_EMBEDDING_FILE), 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                header_line = ['']
                header_line.extend([x for x in range(1, self._embedding_generator.get_dimensions())])
                writer.writerow(header_line)
                for row in tqdm(self._embedding_generator.get_next_embedding(), desc='Saving embedding'):
                    writer.writerow(row)

            self._register_image_coembedding_file()

            self._register_computation()

            exitcode = 0
        finally:
            self._end_time = int(time.time())
            # write a task finish file
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            end_time=self._end_time,
                                            status=exitcode)
        logger.debug('Exit code: ' + str(exitcode))
        return exitcode
