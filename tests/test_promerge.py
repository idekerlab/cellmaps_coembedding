#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the ProMERGE co-embedding feature.

Covers the dev-branch addition of the ``promerge`` algorithm: the
``ProMERGECoEmbeddingGenerator`` in ``runner.py``, its ``--lambda_*`` CLI
wiring in ``cellmaps_coembeddingcmd``, and the pure helpers in the
``promerge`` package. The heavy training path (``promerge.fit_predict`` /
the torch model) is mocked, following the convention in
``test_cellmaps_coembeddingrunner.py``.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import torch

from cellmaps_coembedding.runner import ProMERGECoEmbeddingGenerator
from cellmaps_coembedding import cellmaps_coembeddingcmd
from cellmaps_coembedding.promerge import (add_gaussian_noise,
                                           variance_regularizer,
                                           fit_predict, save_results)
from cellmaps_coembedding.promerge.model import CoEmbed
from cellmaps_coembedding.proteinprojector.architecture import (
    TrainingDataWrapper, Protein_Dataset)


def _write_embedding_file(path, rows, dim=2):
    """Writes a minimal tab-delimited embedding file (header + rows)."""
    with open(path, 'w') as f:
        f.write('\t'.join([''] + [str(i) for i in range(dim)]) + '\n')
        for gene, vec in rows:
            f.write('\t'.join([gene] + [str(v) for v in vec]) + '\n')


class TestProMERGEGenerator(unittest.TestCase):

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp()
        self._f1 = os.path.join(self._temp_dir, 'ppi.tsv')
        self._f2 = os.path.join(self._temp_dir, 'image.tsv')
        _write_embedding_file(self._f1, [('geneB', [0.1, 0.2]),
                                         ('geneA', [0.3, 0.4])])
        _write_embedding_file(self._f2, [('geneA', [0.5, 0.6]),
                                         ('geneC', [0.7, 0.8])])

    def tearDown(self):
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def test_constructor_stores_parameters(self):
        gen = ProMERGECoEmbeddingGenerator(
            dimensions=64,
            outdir=self._temp_dir,
            embeddings=[self._f1, self._f2],
            embedding_names=['ppi_base', 'image_base'],
            n_epochs=42,
            lambda_reconstruction=2.0,
            lambda_disentangle=3.0,
            lambda_triplet_disentangle=4.0,
            lambda_l2_disentangle=5.0,
            lambda_l2_latent=6.0,
            lambda_var=7.0,
            disentangle_method='subtract')
        self.assertEqual(64, gen.get_dimensions())
        self.assertEqual(self._temp_dir, gen._outdir)
        self.assertEqual(42, gen._n_epochs)
        self.assertEqual(2.0, gen._lambda_reconstruction)
        self.assertEqual(3.0, gen._lambda_disentangle)
        self.assertEqual(4.0, gen._lambda_triplet_disentangle)
        self.assertEqual(5.0, gen._lambda_l2_disentangle)
        self.assertEqual(6.0, gen._lambda_l2_latent)
        self.assertEqual(7.0, gen._lambda_var)
        self.assertEqual('subtract', gen._disentangle_method)

    def test_constructor_lambda_defaults(self):
        gen = ProMERGECoEmbeddingGenerator(embeddings=[self._f1, self._f2])
        self.assertEqual(1.0, gen._lambda_reconstruction)
        self.assertEqual(1.0, gen._lambda_disentangle)
        self.assertEqual(1.0, gen._lambda_triplet_disentangle)
        self.assertEqual(0, gen._lambda_l2_disentangle)
        self.assertEqual(0, gen._lambda_l2_latent)
        self.assertEqual(0.1, gen._lambda_var)
        self.assertEqual('MINE', gen._disentangle_method)

    @patch('cellmaps_coembedding.runner.promerge.fit_predict')
    def test_get_next_embedding_wires_fit_predict(self, mock_fit_predict):
        fake_rows = [['geneA', 0.11, 0.22], ['geneB', 0.33, 0.44]]
        mock_fit_predict.return_value = iter(fake_rows)

        gen = ProMERGECoEmbeddingGenerator(
            dimensions=64,
            outdir=self._temp_dir,
            embeddings=[self._f1, self._f2],
            embedding_names=['ppi_base', 'image_base'],
            n_epochs=7,
            lambda_var=0.9,
            disentangle_method='subtract')

        result = list(gen.get_next_embedding())
        self.assertEqual(fake_rows, result)

        mock_fit_predict.assert_called_once()
        kwargs = mock_fit_predict.call_args.kwargs
        # a representative slice of the ~25 params, confirming they thread through
        self.assertEqual(self._temp_dir, kwargs['resultsdir'])
        self.assertEqual(64, kwargs['latent_dim'])
        self.assertEqual(7, kwargs['n_epochs'])
        self.assertEqual(0.9, kwargs['lambda_var'])
        self.assertEqual('subtract', kwargs['disentangle_method'])
        self.assertEqual(['ppi_base', 'image_base'], kwargs['modality_names'])


class TestPromergeHelpers(unittest.TestCase):

    def _frame(self):
        return pd.DataFrame(
            {'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 20.0, 30.0, 40.0]},
            index=['g1', 'g2', 'g3', 'g4'])

    def test_add_gaussian_noise_zero_frac_is_noop(self):
        df = self._frame()
        out = add_gaussian_noise(df, frac=0.0, seed=0)
        self.assertTrue(np.allclose(df.to_numpy(), out.to_numpy()))

    def test_add_gaussian_noise_preserves_shape_and_labels(self):
        df = self._frame()
        out = add_gaussian_noise(df, frac=0.1, seed=1)
        self.assertEqual(df.shape, out.shape)
        self.assertEqual(list(df.index), list(out.index))
        self.assertEqual(list(df.columns), list(out.columns))

    def test_add_gaussian_noise_seed_reproducible(self):
        df = self._frame()
        a = add_gaussian_noise(df, frac=0.2, seed=123)
        b = add_gaussian_noise(df, frac=0.2, seed=123)
        self.assertTrue(np.allclose(a.to_numpy(), b.to_numpy()))

    def test_variance_regularizer_zero_when_std_matches_target(self):
        # unbiased std of [-s, s] is 1 when s = 1/sqrt(2)
        s = 1.0 / np.sqrt(2)
        z = torch.tensor([[-s, -s], [s, s]], dtype=torch.float32)
        reg = variance_regularizer(z, target=1.0)
        self.assertAlmostEqual(0.0, reg.item(), places=5)

    def test_variance_regularizer_penalizes_deviation(self):
        z_far = torch.zeros((4, 2), dtype=torch.float32)  # std 0, target 1
        reg = variance_regularizer(z_far, target=1.0)
        self.assertGreater(reg.item(), 0.0)


class TestProMERGECmd(unittest.TestCase):

    def test_parse_arguments_lambda_defaults(self):
        res = cellmaps_coembeddingcmd._parse_arguments(
            'desc', ['outdir', '--ppi_embeddingdir', 'p',
                     '--image_embeddingdir', 'i'])
        self.assertEqual(1.0, res.lambda_disentangle)
        self.assertEqual(0.0, res.lambda_l2_disentangle)
        self.assertEqual(0.1, res.lambda_var)

    def test_parse_arguments_lambda_overrides(self):
        res = cellmaps_coembeddingcmd._parse_arguments(
            'desc', ['outdir', '--ppi_embeddingdir', 'p',
                     '--image_embeddingdir', 'i',
                     '--lambda_disentangle', '2.5',
                     '--lambda_l2_disentangle', '0.3',
                     '--lambda_var', '0.05'])
        self.assertEqual(2.5, res.lambda_disentangle)
        self.assertEqual(0.3, res.lambda_l2_disentangle)
        self.assertEqual(0.05, res.lambda_var)

    @patch('cellmaps_coembedding.cellmaps_coembeddingcmd.CellmapsCoEmbedder')
    @patch('cellmaps_coembedding.cellmaps_coembeddingcmd.'
           'ProMERGECoEmbeddingGenerator')
    def test_main_promerge_branch_wires_generator(self, mock_gen, mock_embedder):
        mock_embedder.return_value.run.return_value = 0
        temp_dir = tempfile.mkdtemp()
        try:
            rc = cellmaps_coembeddingcmd.main(
                ['prog', temp_dir,
                 '--algorithm', 'promerge',
                 '--embeddings', 'd1', 'd2',
                 '--latent_dimension', '77',
                 '--lambda_disentangle', '2.5',
                 '--lambda_var', '0.05'])
            self.assertEqual(0, rc)
            mock_gen.assert_called_once()
            kwargs = mock_gen.call_args.kwargs
            self.assertEqual(77, kwargs['dimensions'])
            self.assertEqual(2.5, kwargs['lambda_disentangle'])
            self.assertEqual(0.05, kwargs['lambda_var'])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCoEmbedModel(unittest.TestCase):
    """Exercises the CoEmbed model's forward pass directly (no training)."""

    LAT = 4

    def _wrapper(self, disentangle_method):
        import torch
        proteins = ['P%d' % i for i in range(6)]
        rng = np.random.default_rng(0)
        m1 = [[p] + list(rng.normal(size=3)) for p in proteins]
        m2 = [[p] + list(rng.normal(size=5)) for p in proteins]
        dw = TrainingDataWrapper([m1, m2], ['mod1-query', 'mod2-query'],
                                 torch.device('cpu'), False, 0.0, self.LAT,
                                 8, 4, os.path.join(tempfile.mkdtemp(), 'm'))
        dw.disentangle_method = disentangle_method
        return dw, proteins

    def _inputs(self, dw):
        return {name: mod.train_features
                for name, mod in dw.modalities_dict.items()}

    def test_forward_mine_shapes_and_keys(self):
        dw, proteins = self._wrapper('MINE')
        model = CoEmbed(dw)
        # MINE path does not consult anchor_emb in forward
        latents, outputs, disentangles = model(self._inputs(dw), proteins, {})
        self.assertEqual({'mod1-query', 'mod2-query'}, set(latents))
        for v in latents.values():
            self.assertEqual((6, self.LAT), tuple(v.shape))
        # every (latent_modality, input_modality) pair gets a decoded output
        self.assertEqual(
            {'mod1-query___mod1-query', 'mod1-query___mod2-query',
             'mod2-query___mod1-query', 'mod2-query___mod2-query'},
            set(outputs))
        self.assertEqual({'mod1-query', 'mod2-query'}, set(disentangles))

    def test_forward_subtract_uses_anchor(self):
        dw, proteins = self._wrapper('subtract')
        model = CoEmbed(dw)
        anchor = {m: pd.DataFrame(np.random.randn(len(proteins), self.LAT),
                                  index=proteins)
                  for m in dw.modalities_dict}
        latents, _, disentangles = model(self._inputs(dw), proteins, anchor)
        for v in latents.values():
            self.assertEqual((6, self.LAT), tuple(v.shape))
        for v in disentangles.values():
            self.assertEqual((6, self.LAT), tuple(v.shape))

    def test_forward_subtract_missing_anchor_fills_zero(self):
        # a protein absent from the anchor frame must not crash (reindex fills 0)
        dw, proteins = self._wrapper('subtract')
        model = CoEmbed(dw)
        anchor = {m: pd.DataFrame(np.random.randn(3, self.LAT),
                                  index=proteins[:3])
                  for m in dw.modalities_dict}
        latents, _, _ = model(self._inputs(dw), proteins, anchor)
        self.assertEqual((6, self.LAT), tuple(latents['mod1-query'].shape))

    def test_invalid_disentangle_method_raises(self):
        dw, _ = self._wrapper('bogus')
        with self.assertRaises(Exception):
            CoEmbed(dw)


class TestSaveResults(unittest.TestCase):

    def test_save_results_writes_files_and_returns_dict(self):
        import torch
        LAT = 4
        proteins = ['P%d' % i for i in range(5)]
        rng = np.random.default_rng(0)
        m1 = [[p] + list(rng.normal(size=3)) for p in proteins]
        m2 = [[p] + list(rng.normal(size=5)) for p in proteins]
        temp_dir = tempfile.mkdtemp()
        try:
            resultsdir = os.path.join(temp_dir, 'promerge')
            dw = TrainingDataWrapper([m1, m2], ['mod1-query', 'mod2-query'],
                                     torch.device('cpu'), False, 0.0, LAT,
                                     8, 4, resultsdir)
            dw.disentangle_method = 'MINE'  # forward ignores anchor for MINE
            model = CoEmbed(dw)
            dataset = Protein_Dataset(dw.modalities_dict)

            embeddings = save_results(model, dataset, dw, {}, '_suffix')
            self.assertIsInstance(embeddings, dict)
            self.assertEqual(len(proteins), len(embeddings))

            base = resultsdir + '_suffix'
            expected = [base + '_model.pth', base + '_latent.tsv',
                        base + '_disentangle.tsv']
            for mod in ['mod1-query', 'mod2-query']:
                expected.append('%s_%s_latent.tsv' % (base, mod))
                expected.append('%s_%s_disentangle.tsv' % (base, mod))
            for i in ['mod1-query', 'mod2-query']:
                for o in ['mod1-query', 'mod2-query']:
                    expected.append('%s_%s___%s_reconstructed.tsv' % (base, i, o))
            for path in expected:
                self.assertTrue(os.path.exists(path),
                                'missing output: %s' % path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFitPredictSmoke(unittest.TestCase):
    """End-to-end smoke tests of the training loop on tiny data.

    Base-context anchor latent files are pre-seeded so ProMERGE skips the
    ProteinProjector base pass, isolating ProMERGE's own code.
    """

    LAT = 4

    def _setup(self, temp_dir):
        resultsdir = os.path.join(temp_dir, 'promerge')
        base_dir = os.path.join(resultsdir, 'base')
        os.makedirs(base_dir)
        proteins = ['P%d' % i for i in range(8)]
        rng = np.random.default_rng(0)

        def rows(dim):
            return [[p] + list(rng.normal(size=dim)) for p in proteins]

        for mod in ['mod1', 'mod2']:
            fn = os.path.join(base_dir, 'base_%s-base_latent.tsv' % mod)
            with open(fn, 'w') as f:
                f.write('\t' + '\t'.join(str(i) for i in range(self.LAT)) + '\n')
                for p in proteins:
                    f.write(p + '\t' +
                            '\t'.join('%.4f' % v for v in rng.normal(size=self.LAT))
                            + '\n')
        modality_data = [rows(3), rows(5), rows(3), rows(5)]
        modality_names = ['mod1-base', 'mod2-base', 'mod1-query', 'mod2-query']
        return resultsdir, modality_data, modality_names

    def _run(self, disentangle_method, n_epochs):
        temp_dir = tempfile.mkdtemp()
        try:
            resultsdir, data, names = self._setup(temp_dir)
            gen = fit_predict(
                resultsdir=resultsdir, modality_data=data, modality_names=names,
                latent_dim=self.LAT, n_epochs=n_epochs, batch_size=2,
                hidden_size_1=8, hidden_size_2=4,
                save_update_epochs=False, mean_losses=True,
                cond_str_list=['base', 'query'], mod_str_list=['mod1', 'mod2'],
                disentangle_method=disentangle_method)
            rows_out = list(gen)
            # one averaged row per protein, each latent_dim wide
            self.assertEqual(8, len(rows_out))
            for row in rows_out:
                self.assertEqual(self.LAT, len(row) - 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_fit_predict_mine(self):
        # 2 epochs: epoch 0 is MINE warmup, epoch 1 updates the main model
        self._run('MINE', n_epochs=2)

    def test_fit_predict_subtract(self):
        self._run('subtract', n_epochs=1)


if __name__ == '__main__':
    unittest.main()
