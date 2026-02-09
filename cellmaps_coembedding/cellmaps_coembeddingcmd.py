#! /usr/bin/env python

import argparse
import json
import os
import sys
import logging
import logging.config

from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_coembedding
from cellmaps_coembedding.runner import (
    CellmapsCoEmbedder,
    EmbeddingGenerator,
    FakeCoEmbeddingGenerator,
    MuseCoEmbeddingGenerator,
    ProteinGPSCoEmbeddingGenerator,
    ProteinProjectorCoEmbeddingGenerator,
    ProMERGECoEmbeddingGenerator
)

logger = logging.getLogger(__name__)

PPI_EMBEDDINGDIR = '--ppi_embeddingdir'
IMAGE_EMBEDDINGDIR = '--image_embeddingdir'


def _parse_arguments(desc, args):
    """
    Parses command line arguments

    :param desc: description to display on command line
    :type desc: str
    :param args: command line arguments usually :py:func:`sys.argv[1:]`
    :type args: list
    :return: arguments parsed by :py:mod:`argparse`
    :rtype: :py:class:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=constants.ArgParseFormatter)
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--embeddings', nargs='+',
                        help='Filepath to .tsv with embeddings. Requires two or more paths.')
    parser.add_argument('--embedding_names', nargs='+',
                        help='Name corresponding to each filepath input in --embeddings. ')
    parser.add_argument('--algorithm', choices=['auto', 'muse', 'proteingps', 'proteinprojector'], default='muse',
                        help='Algorithm to use for coembedding. Defaults to MUSE. "auto" and "proteingps" are '
                             'deprecated; use "proteinprojector" instead.'
                        )
    parser.add_argument(PPI_EMBEDDINGDIR,
                        help='Directory aka rocrate where ppi '
                             'embedding file resides (Deprecated: use --embeddings flag)')
    parser.add_argument(IMAGE_EMBEDDINGDIR,
                        help='Directory aka rocrate image embedding '
                             'file resides (Deprecated: use --embeddings flag)')
    parser.add_argument('--latent_dimension', type=int, default=EmbeddingGenerator.LATENT_DIMENSIONS,
                        help='Output dimension of embedding')
    parser.add_argument('--n_epochs_init', default=MuseCoEmbeddingGenerator.N_EPOCHS_INIT, type=int,
                        help='# of init training epochs (for muse algorithm)')
    parser.add_argument('--n_epochs', default=EmbeddingGenerator.N_EPOCHS, type=int,
                        help='# of training epochs')
    parser.add_argument('--jackknife_percent', default=EmbeddingGenerator.JACKKNIFE_PERCENT, type=float,
                        help='Percentage of data to withhold from training'
                             'a value of 0.1 means to withhold 10 percent of the data')
    parser.add_argument('--dropout', default=EmbeddingGenerator.DROPOUT, type=float,
                        help='Percentage to use fo dropout layers in neural network')
    parser.add_argument('--k', default=EmbeddingGenerator.K, type=int,
                        help='Number of neighbors for clustering.')
    parser.add_argument('--l2_norm', action='store_true',
                        help='If set, L2 normalize coembeddings (for proteinprojector algorithm, formerly proteingps)')
    parser.add_argument('--lambda_reconstruction', type=float, default=1.0,
                        help='Weight for reconstruction loss (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 1.0)')
    parser.add_argument('--lambda_l2', type=float, default=0.001,
                        help='Weight for L2 regularization (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 0.001)')
    parser.add_argument('--lambda_triplet', type=float, default=1.0,
                        help='Weight for triplet loss (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 1.0)')
    parser.add_argument('--mean_losses', action='store_true',
                        help='If set, use mean of losses otherwise sum (for proteinprojector algorithm, '
                             'formerly proteingps)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 16)')
    parser.add_argument('--triplet_margin', type=float, default=1.0,
                        help='Margin for triplet loss (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 1.0)')
    parser.add_argument('--learn_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 1e-4)')
    parser.add_argument('--hidden_size_1', type=int, default=512,
                        help='Size of first hidden layer (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 512)')
    parser.add_argument('--hidden_size_2', type=int, default=256,
                        help='Size of second hidden layer (for proteinprojector algorithm, formerly proteingps) '
                             '(default: 256)')
    parser.add_argument('--save_update_epochs', action='store_true',
                        help='If set, save model state at specified epoch intervals (for proteinprojector algorithm, '
                             'formerly proteingps)')
    parser.add_argument('--negative_from_batch', action='store_true',
                        help='If set, use negative samples from same batch for triplet loss '
                             '(for proteinprojector algorithm, formerly proteingps)')
    parser.add_argument('--lambda_disentangle', type=float, default=1.0,
                        help='Weight for disentanglement loss '
                        '(for ProMERGE algorithm) (default: 1.0)')
    parser.add_argument('--lambda_l2_disentangle', type=float, default=0.0,
                        help='Weight for L2 regularization on disentanglement layers '
                        '(for ProMERGE algorithm) (default: 0.0)')
    parser.add_argument('--lambda_var', type=float, default=0.1,
                        help='Weight for variance loss regularization '
                        '(for ProMERGE algorithm) (default: 0.1)')
    parser.add_argument('--fake_embedding', action='store_true',
                        help='If set, generate fake coembeddings')
    parser.add_argument('--provenance',
                        help='Path to file containing provenance '
                             'information about input files in JSON format. '
                             'This is required if none of embeddings directory contains '
                             'ro-crate-metadata.json file.')
    parser.add_argument('--name',
                        help='Name of this run, needed for FAIRSCAPE. If '
                             'unset, name value from specified '
                             'by --embeddings directories or provenance file will be used')
    parser.add_argument('--organization_name',
                        help='Name of organization running this tool, needed '
                             'for FAIRSCAPE. If unset, organization name specified '
                             'in --embedding directories or provenance file will be used')
    parser.add_argument('--project_name',
                        help='Name of project running this tool, needed for '
                             'FAIRSCAPE. If unset, project name specified '
                             'in --embedding directories or provenance file will be used')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--skip_logging', action='store_true',
                        help='If set, output.log, error.log '
                             'files will not be created')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = WARNING, -vv = INFO, '
                             '-vvv = DEBUG, -vvvv = NOTSET (default ERROR '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_coembedding.__version__))

    return parser.parse_args(args)


def main(args):
    """
    Main entry point for program

    :param args: arguments passed to command line usually :py:func:`sys.argv[1:]`
    :type args: list

    :return: return value of :py:meth:`cellmaps_coembedding.runner.CellmapsCoEmbedder.run`
             or ``2`` if an exception is raised
    :rtype: int
    """
    desc = """
    Version {version}

    Given input embeddings, this tool generates a co-embedding using either a UniEmbed algorith or
    a variant of MuSE algorithm within this code base from
    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022
    that is under MIT License.

    To run this tool requires that an output directory be specified and two embeddings
    be set via --embeddings flag. The values passed to --embeddings can be an ro-crate
    containing either a ppi_emd.tsv or image_emd.tsv file or a path to a TSV file.

    It is assumed these files are tab delimited embeddings and for each row,
    first value is assumed to be sample ID followed by the embeddings separated by
    tabs. The first row is assumed to be a header.



    """.format(version=cellmaps_coembedding.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_coembedding.__version__

    if theargs.algorithm == 'auto':
        logging.warning('"auto" is deprecated and will default to "proteinprojector". '
                        'Please use "--algorithm proteinprojector" instead.')
        theargs.algorithm = 'proteinprojector'

    if theargs.algorithm == 'proteingps':
        logging.warning('"proteingps" algorithm is deprecated and will be removed in a future release. '
                        'Use "--algorithm proteinprojector" instead.')
        theargs.algorithm = 'proteinprojector'

    if (theargs.ppi_embeddingdir or theargs.image_embeddingdir) and theargs.embeddings:
        raise CellmapsCoEmbeddingError('Use either --ppi_embeddingdir and --image_embeddingdir or --embeddings, '
                                       'not both')
    if theargs.embeddings:
        if len(theargs.embeddings) > 2 and theargs.algorithm == 'muse':
            raise CellmapsCoEmbeddingError('Only two embeddings are supported with --embeddings for MUSE algorithm')

    if not (theargs.ppi_embeddingdir and theargs.image_embeddingdir) and not theargs.embeddings:
        raise CellmapsCoEmbeddingError('Either --ppi_embeddingdir and --image_embeddingdir, '
                                       'or --embeddings are required')

    if theargs.provenance is not None:
        with open(theargs.provenance, 'r') as f:
            json_prov = json.load(f)
    else:
        json_prov = None

    try:
        logutils.setup_cmd_logging(theargs)
        gen = EmbeddingGenerator()
        if theargs.fake_embedding:
            gen = FakeCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                           ppi_embeddingdir=theargs.ppi_embeddingdir,
                                           image_embeddingdir=theargs.image_embeddingdir,
                                           embeddings=theargs.embeddings,
                                           embedding_names=theargs.embedding_names)
        else:
            if theargs.algorithm == 'muse':
                gen = MuseCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                               ppi_embeddingdir=theargs.ppi_embeddingdir,
                                               image_embeddingdir=theargs.image_embeddingdir,
                                               n_epochs=theargs.n_epochs,
                                               n_epochs_init=theargs.n_epochs_init,
                                               jackknife_percent=theargs.jackknife_percent,
                                               dropout=theargs.dropout,
                                               k=theargs.k,
                                               outdir=os.path.abspath(theargs.outdir),
                                               embeddings=theargs.embeddings,
                                               embedding_names=theargs.embedding_names)
            elif theargs.algorithm == 'proteinprojector':
                gen = ProteinProjectorCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                                           ppi_embeddingdir=theargs.ppi_embeddingdir,
                                                           image_embeddingdir=theargs.image_embeddingdir,
                                                           n_epochs=theargs.n_epochs,
                                                           dropout=theargs.dropout,
                                                           l2_norm=theargs.l2_norm,
                                                           jackknife_percent=theargs.jackknife_percent,
                                                           outdir=os.path.abspath(theargs.outdir),
                                                           embeddings=theargs.embeddings,
                                                           embedding_names=theargs.embedding_names,
                                                           mean_losses=theargs.mean_losses,
                                                           lambda_reconstruction=theargs.lambda_reconstruction,
                                                           lambda_l2=theargs.lambda_l2,
                                                           lambda_triplet=theargs.lambda_triplet,
                                                           batch_size=theargs.batch_size,
                                                           triplet_margin=theargs.triplet_margin,
                                                           learn_rate=theargs.learn_rate,
                                                           hidden_size_1=theargs.hidden_size_1,
                                                           hidden_size_2=theargs.hidden_size_2,
                                                           save_update_epochs=theargs.save_update_epochs,
                                                           negative_from_batch=theargs.negative_from_batch)
            elif theargs.algorithm == 'proteingps':
                gen = ProteinGPSCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                                     ppi_embeddingdir=theargs.ppi_embeddingdir,
                                                     image_embeddingdir=theargs.image_embeddingdir,
                                                     n_epochs=theargs.n_epochs,
                                                     dropout=theargs.dropout,
                                                     l2_norm=theargs.l2_norm,
                                                     jackknife_percent=theargs.jackknife_percent,
                                                     outdir=os.path.abspath(theargs.outdir),
                                                     embeddings=theargs.embeddings,
                                                     embedding_names=theargs.embedding_names,
                                                     mean_losses=theargs.mean_losses,
                                                     lambda_reconstruction=theargs.lambda_reconstruction,
                                                     lambda_l2=theargs.lambda_l2,
                                                     lambda_triplet=theargs.lambda_triplet,
                                                     batch_size=theargs.batch_size,
                                                     triplet_margin=theargs.triplet_margin,
                                                     learn_rate=theargs.learn_rate,
                                                     hidden_size_1=theargs.hidden_size_1,
                                                     hidden_size_2=theargs.hidden_size_2,
                                                     save_update_epochs=theargs.save_update_epochs,
                                                     negative_from_batch=theargs.negative_from_batch)
            elif theargs.algorithm == 'promerge':
                gen = ProMERGECoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                                   ppi_embeddingdir=theargs.ppi_embeddingdir,
                                                   image_embeddingdir=theargs.image_embeddingdir,
                                                   n_epochs=theargs.n_epochs,
                                                   dropout=theargs.dropout,
                                                   l2_norm=theargs.l2_norm,
                                                   outdir=os.path.abspath(theargs.outdir),
                                                   embeddings=theargs.embeddings,
                                                   embedding_names=theargs.embedding_names,
                                                   batch_size=theargs.batch_size,
                                                   triplet_margin=theargs.triplet_margin,
                                                   mean_losses=theargs.mean_losses,
                                                   learn_rate=theargs.learn_rate,
                                                   hidden_size_1=theargs.hidden_size_1,
                                                   hidden_size_2=theargs.hidden_size_2,
                                                   save_update_epochs=theargs.save_update_epochs,
                                                   negative_from_batch=theargs.negative_from_batch,

                                                   lambda_reconstruction=theargs.lambda_reconstruction,
                                                   lambda_triplet_disentangle=theargs.lambda_triplet,
                                                   lambda_disentangle=theargs.lambda_disentangle,
                                                   lambda_l2_disentangle=theargs.lambda_l2_disentangle,
                                                   lambda_l2_latent=theargs.lambda_l2,
                                                   lambda_var=theargs.lambda_var)

        inputdirs = gen.get_embedding_inputdirs()
        return CellmapsCoEmbedder(outdir=theargs.outdir,
                                  inputdirs=inputdirs,
                                  embedding_generator=gen,
                                  name=theargs.name,
                                  organization_name=theargs.organization_name,
                                  project_name=theargs.project_name,
                                  skip_logging=theargs.skip_logging,
                                  input_data_dict=theargs.__dict__,
                                  provenance=json_prov).run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
