import csv
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

import cellmaps_coembedding.protein_gps as legacy_proteingps
import cellmaps_coembedding.proteinprojector as proteinprojector
from cellmaps_coembedding.proteinprojector import (
    Modality,
    Protein_Dataset,
    ToTensor,
    TrainingDataWrapper,
    fit_predict,
    save_results,
    uniembed_nn,
    write_embedding_dictionary_to_file,
)


class TestToTensor(unittest.TestCase):
    def test_conversion(self):
        to_tensor = ToTensor()
        numpy_array = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(numpy_array)
        self.assertTrue(torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0])))


class TestModality(unittest.TestCase):
    def setUp(self):
        self.training_data = [[0, 1.0, 2.0, 3.0], [1, 4.0, 5.0, 6.0]]
        self.transform = ToTensor()
        self.device = torch.device("cpu")

    def test_initialization(self):
        modality = Modality(self.training_data, 'test_modality', self.transform, self.device)
        self.assertEqual(modality.name, 'test_modality')
        self.assertTrue(torch.equal(modality.train_features, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))
        self.assertEqual(modality.input_dim, 3)


class TestProteinDataset(unittest.TestCase):
    def setUp(self):
        training_data = [[0, 1.0, 2.0, 3.0], [1, 4.0, 5.0, 6.0]]
        transform = ToTensor()
        device = torch.device("cpu")
        modality = Modality(training_data, 'test_modality', transform, device)
        self.dataset = Protein_Dataset({'test_modality': modality})

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        features, mask, index = self.dataset[0]
        self.assertIn('test_modality', features)
        self.assertIn('test_modality', mask)
        self.assertEqual(index, 0)


class TestTrainingDataWrapper(unittest.TestCase):
    def setUp(self):
        self.training_data = [[[0, 1.0, 2.0, 3.0], [1, 4.0, 5.0, 6.0]]]
        self.modality_names = ['test_modality']
        self.device = torch.device("cpu")
        self.wrapper = TrainingDataWrapper(self.training_data, self.modality_names, self.device, False, 0.5, 2, 10, 5,
                                           '/tmp')

    def test_initialization(self):
        self.assertEqual(len(self.wrapper.modalities_dict), 1)
        self.assertIn('test_modality', self.wrapper.modalities_dict)


class Testuniembednn(unittest.TestCase):
    def setUp(self):
        training_data = [[[0, 1.0, 2.0, 3.0], [1, 4.0, 5.0, 6.0]]]
        modality_names = ['test_modality']
        device = torch.device("cpu")
        wrapper = TrainingDataWrapper(training_data, modality_names, device, False, 0.5, 2, 10, 5, '/tmp')
        self.model = uniembed_nn(wrapper)

    def test_forward(self):
        test_input = {'test_modality': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        latents, outputs = self.model.forward(test_input)
        self.assertIn('test_modality', latents)
        self.assertIn('test_modality___test_modality', outputs)


class TestProteinProjector(unittest.TestCase):

    def test_write_embedding_dictionary_to_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(temp_dir, "embeddings.tsv")
            dictionary = {'protein1': np.array([1.0, 2.0]), 'protein2': np.array([3.0, 4.0])}
            dims = 2

            write_embedding_dictionary_to_file(filepath, dictionary, dims)

            expected_header = [''] + [str(x) for x in range(1, dims)]
            expected_rows = [
                ['protein1', '1.0', '2.0'],
                ['protein2', '3.0', '4.0']
            ]

            with open(filepath, 'r', newline='') as file:
                reader = csv.reader(file, delimiter='\t')
                header = next(reader)
                self.assertEqual(header, expected_header)

                for row, expected_row in zip(reader, expected_rows):
                    self.assertEqual(row, expected_row)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_results(self):
        temp_dir = tempfile.mkdtemp()
        try:
            modality_data = [
                [[1, 1.0, 2.0, 3.0], [2, 4.0, 5.0, 6.0]],
                [[1, 1.0, 2.0, 3.0], [2, 4.0, 5.0, 6.0]]
            ]
            modality_names = ['testmod1', 'testmod2']
            device = torch.device("cpu")
            result_dir = os.path.join(temp_dir, 'proteinprojector')
            data_wrapper = TrainingDataWrapper(modality_data, modality_names, device, False, 0.5, 2, 10, 5, result_dir)
            protein_dataset = Protein_Dataset(data_wrapper.modalities_dict)
            model = uniembed_nn(data_wrapper)

            embeddings = save_results(model, protein_dataset, data_wrapper, "_suffix")
            self.assertIsInstance(embeddings, dict)

            expected_files = ([f"{result_dir}_suffix_{mod}_latent.tsv" for mod in modality_names] +
                              [f"{result_dir}_suffix_{mod}___{mod}_reconstructed.tsv" for mod in modality_names])

            for file_path in expected_files:
                self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist.")

        finally:
            shutil.rmtree(temp_dir)


class TestFitPredict(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.modality_data = [torch.randn(10, 3), torch.randn(10, 3)]
        self.modality_names = ['mod1', 'mod2']

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_fit_predict_simple_case(self):
        config = {
            'resultsdir': self.results_dir,
            'modality_data': self.modality_data,
            'modality_names': self.modality_names,
            'batch_size': 2,
            'latent_dim': 2,
            'n_epochs': 1,
            'triplet_margin': 0.5,
            'lambda_reconstruction': 1.0,
            'lambda_triplet': 1.0,
            'l2_norm': False,
            'dropout': 0.1,
            'save_epoch': 1,
            'learn_rate': 0.001,
            'hidden_size_1': 10,
            'hidden_size_2': 5,
            'save_update_epochs': False,
            'mean_losses': True,
            'negative_from_batch': True
        }

        results_generator = fit_predict(**config)
        results = list(results_generator)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], list)


class TestDeprecatedAlias(unittest.TestCase):
    def test_fit_predict_alias(self):
        self.assertIs(legacy_proteingps.fit_predict, proteinprojector.fit_predict)


if __name__ == '__main__':
    unittest.main()
