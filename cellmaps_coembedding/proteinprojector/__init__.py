"""
ProteinProjector co-embedding algorithm.

This module provides the ProteinProjector implementation, formerly known as
ProteinGPS. It exposes utilities for training the neural network, saving
results, and yielding co-embeddings.
"""

import collections
import csv
import random
from typing import Dict, Generator, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .architecture import (MODALITY_SEP, Modality, Protein_Dataset, ToTensor,
                           TrainingDataWrapper, uniembed_nn)

__all__ = [
    "MODALITY_SEP",
    "Modality",
    "Protein_Dataset",
    "ToTensor",
    "TrainingDataWrapper",
    "uniembed_nn",
    "write_embedding_dictionary_to_file",
    "save_results",
    "fit_predict",
]


def write_embedding_dictionary_to_file(filepath: str,
                                       dictionary: Dict[str, np.ndarray],
                                       dims: int) -> None:
    """
    Writes a dictionary of embeddings to a tab-separated file with headers.

    :param filepath: Path to the file where embeddings will be saved.
    :param dictionary: Dictionary of embeddings with keys as names and values as embedding vectors.
    :param dims: Dimension of embedding vectors.
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        header_line = ['']
        header_line.extend([x for x in range(1, dims)])
        writer.writerow(header_line)
        for key, value in dictionary.items():
            row = [key]
            if isinstance(value, dict):
                averaged_values = np.mean(list(value.values()), axis=0)
                row.extend(averaged_values)
            else:
                row.extend(value)
            writer.writerow(row)


def save_results(model: torch.nn.Module,
                 protein_dataset: Protein_Dataset,
                 data_wrapper: TrainingDataWrapper,
                 results_suffix: str = '') -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluates the model, saves the state, and exports embeddings for each protein.

    :param model: The neural network model.
    :param protein_dataset: The dataset containing protein data.
    :param data_wrapper: Data handling and configurations as an object.
    :param results_suffix: Suffix to append to results directory for saving.
    :return: Dictionary keyed by protein containing modality embeddings.
    """
    resultsdir = data_wrapper.resultsdir + results_suffix
    model.eval()
    torch.save(model.state_dict(), f'{resultsdir}_model.pth')

    all_latents: Dict[str, Dict[str, np.ndarray]] = dict()
    all_outputs: Dict[str, Dict[str, np.ndarray]] = dict()
    for input_modality in data_wrapper.modalities_dict.keys():
        all_latents[input_modality] = dict()
        for output_modality in data_wrapper.modalities_dict.keys():
            output_key = input_modality + MODALITY_SEP + output_modality
            all_outputs[output_key] = dict()

    embeddings_by_protein: Dict[str, Dict[str, np.ndarray]] = dict()
    with torch.no_grad():
        for i in np.arange(len(protein_dataset)):
            protein, mask, protein_index = protein_dataset[i]
            protein_name = protein_dataset.protein_ids[protein_index]
            embeddings_by_protein[protein_name] = dict()
            latents, outputs = model(protein)
            for modality, latent in latents.items():
                if mask[modality] > 0:
                    protein_embedding = latent.detach().cpu().numpy()
                    all_latents[modality][protein_name] = protein_embedding
                    embeddings_by_protein[protein_name][modality] = protein_embedding
            for modality, output in outputs.items():
                input_modality = modality.split(MODALITY_SEP)[0]
                output_modality = modality.split(MODALITY_SEP)[1]
                if mask[input_modality] > 0:
                    all_outputs[modality][protein_name] = output.detach().cpu().numpy()

    # save latent embeddings
    for modality, latents in all_latents.items():
        filepath = f'{resultsdir}_{modality}_latent.tsv'
        write_embedding_dictionary_to_file(filepath, latents, data_wrapper.latent_dim)

    # save averaged coembedding
    filepath = f'{resultsdir}_latent.tsv'
    write_embedding_dictionary_to_file(filepath, embeddings_by_protein, data_wrapper.latent_dim)

    # save reconstructed embeddings
    for modality, outputs in all_outputs.items():
        filepath = f'{resultsdir}_{modality}_reconstructed.tsv'
        output_modality = modality.split(MODALITY_SEP)[1]
        output_modality_dim = data_wrapper.modalities_dict[output_modality].input_dim
        write_embedding_dictionary_to_file(filepath, outputs, output_modality_dim)

    return embeddings_by_protein


def fit_predict(resultsdir: str,
                modality_data: Iterable[Iterable[Iterable[float]]],
                modality_names: Iterable[str] = (),
                batch_size: int = 16,
                latent_dim: int = 128,
                n_epochs: int = 250,
                triplet_margin: float = 1.0,
                lambda_reconstruction: float = 1.0,
                lambda_triplet: float = 1.0,
                lambda_l2: float = 0.001,
                l2_norm: bool = False,
                dropout: float = 0.0,
                save_epoch: int = 50,
                learn_rate: float = 1e-4,
                hidden_size_1: int = 512,
                hidden_size_2: int = 256,
                save_update_epochs: bool = False,
                mean_losses: bool = False,
                negative_from_batch: bool = False) -> Generator[List[float], None, None]:
    """
    Trains and predicts using a deep learning model with the given configuration and data.

    :param resultsdir: Directory to save training results and models.
    :param modality_data: Input data for the model.
    :param modality_names: Names of modalities; autogenerated if not provided.
    :param batch_size: Batch size for training.
    :param latent_dim: Dimensionality of the latent embeddings.
    :param n_epochs: Number of training epochs.
    :param triplet_margin: Margin for triplet loss.
    :param lambda_reconstruction: Weight for reconstruction loss.
    :param lambda_triplet: Weight for triplet loss.
    :param lambda_l2: Weight for L2 regularization.
    :param l2_norm: Whether to use L2 normalization.
    :param dropout: Dropout rate.
    :param save_epoch: Epoch interval at which to save the model.
    :param learn_rate: Learning rate for the optimizer.
    :param hidden_size_1: Size of the first hidden layer.
    :param hidden_size_2: Size of the second hidden layer.
    :param save_update_epochs: Flag to save model state at specified epoch intervals.
    :param mean_losses: Whether to average losses or not.
    :param negative_from_batch: Whether to use negative samples from the same batch for triplet loss.
    :returns: Generator of average embeddings for each protein.
    """
    source_file = open(f'{resultsdir}.txt', 'w')

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name()

    # if modality names doesn't match data size, create names with index
    modality_names = list(modality_names)
    num_data_modalities = len(modality_data)
    if len(modality_names) != num_data_modalities:
        modality_names = [f'modality_{x}' for x in np.arange(num_data_modalities)]

    data_wrapper = TrainingDataWrapper(modality_data, modality_names, device, l2_norm, dropout,
                                       latent_dim, hidden_size_1, hidden_size_2, resultsdir)

    # create models, optimizer, trainloader
    ae_model = uniembed_nn(data_wrapper).to(device)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=learn_rate)

    protein_dataset = Protein_Dataset(data_wrapper.modalities_dict)
    train_loader = DataLoader(protein_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):

        # train
        total_loss: List[float] = []
        total_reconstruction_loss: List[float] = []
        total_triplet_loss: List[float] = []
        total_l2_loss: List[float] = []
        total_reconstruction_loss_by_modality: Dict[str, List[float]] = collections.defaultdict(list)
        total_triplet_loss_by_modality: Dict[str, List[float]] = collections.defaultdict(list)

        ae_model.train()

        # loop over all batches
        for step, (batch_data, batch_mask, batch_proteins) in enumerate(train_loader):

            # pass through model
            latents, outputs = ae_model(batch_data)

            batch_reconstruction_losses = torch.tensor([]).to(device)
            batch_triplet_losses = torch.tensor([]).to(device)
            batch_l2_losses = torch.tensor([]).to(device)

            for input_modality in batch_data.keys():

                # get l2 loss
                l2_loss = torch.norm(latents[input_modality], p=2, dim=1)
                batch_l2_losses = torch.cat((batch_l2_losses, l2_loss))

                # get reconstruction losses
                for output_modality in batch_data.keys():

                    # protein_present in both modalities mask
                    mask = (batch_mask[input_modality].bool()) & (batch_mask[output_modality].bool())
                    if torch.sum(mask) == 0:
                        continue  # no overlap

                    output_key = input_modality + MODALITY_SEP + output_modality

                    pairwise_dist_input_output = 1 - F.cosine_similarity(batch_data[output_modality],
                                                                         outputs[output_key], dim=1)
                    reconstruction_loss = pairwise_dist_input_output[mask]
                    batch_reconstruction_losses = torch.cat((batch_reconstruction_losses, reconstruction_loss))
                    total_reconstruction_loss_by_modality[output_key].append(
                        torch.mean(reconstruction_loss).detach().cpu().numpy())

            for anchor_modality in batch_data.keys():
                posneg_modality = random.choice([x for x in batch_data.keys() if x != anchor_modality])

                mask = (batch_mask[anchor_modality].bool()) & (batch_mask[posneg_modality].bool())
                if batch_mask[posneg_modality].sum() < 2:
                    continue
                if torch.sum(mask) == 0:
                    continue

                anchor_latents = latents[anchor_modality]
                positive_latents = latents[posneg_modality]
                positive_dist = 1 - F.cosine_similarity(anchor_latents, positive_latents, dim=1)

                positive_mask = torch.eye(len(mask))
                if negative_from_batch:
                    negative_mask = (torch.logical_not(positive_mask) & (batch_mask[posneg_modality].bool()))
                    negative_indices = [x.nonzero().flatten() for x in negative_mask]
                    negative_index = [int(x[torch.randperm(len(x))[0]]) for x in negative_indices]
                    negative_latents = latents[posneg_modality][negative_index]
                else:
                    posneg_modality_indices = np.arange(len(data_wrapper.modalities_dict[posneg_modality].train_labels))
                    protein_indexes_not_in_batch = list(set(posneg_modality_indices) - set(batch_proteins))
                    negative_indices = random.sample(protein_indexes_not_in_batch, len(positive_dist))
                    negative_data = {posneg_modality:
                                     data_wrapper.modalities_dict[posneg_modality].train_features[negative_indices]}
                    negative_latents_dict, _ = ae_model(negative_data)
                    negative_latents = negative_latents_dict[posneg_modality]

                negative_dist = 1 - F.cosine_similarity(anchor_latents, negative_latents, dim=1)

                triplet_loss = torch.maximum(positive_dist - negative_dist + triplet_margin,
                                             torch.zeros(len(positive_dist)).to(device))
                triplet_loss = triplet_loss[mask]

                batch_triplet_losses = torch.cat((batch_triplet_losses, triplet_loss))
                total_triplet_loss_by_modality[anchor_modality + MODALITY_SEP +
                                               posneg_modality].append(torch.mean(triplet_loss).detach().cpu().numpy())

            if (len(batch_reconstruction_losses) == 0) | (len(batch_triplet_losses) == 0):
                continue

            if mean_losses:
                reconstruction_loss = torch.mean(batch_reconstruction_losses)
                triplet_loss = torch.mean(batch_triplet_losses)
                l2_loss = torch.mean(batch_l2_losses)
            else:
                reconstruction_loss = torch.sum(batch_reconstruction_losses)
                triplet_loss = torch.sum(batch_triplet_losses)
                l2_loss = torch.sum(batch_l2_losses)

            batch_total = (lambda_reconstruction * reconstruction_loss +
                           lambda_triplet * triplet_loss + lambda_l2 * l2_loss)

            ae_optimizer.zero_grad()
            batch_total.backward()
            ae_optimizer.step()

            total_loss.append(batch_total.detach().cpu().numpy())
            total_reconstruction_loss.append(reconstruction_loss.detach().cpu().numpy())
            total_triplet_loss.append(triplet_loss.detach().cpu().numpy())
            total_l2_loss.append(l2_loss.detach().cpu().numpy())

        result_string = (
            f'epoch:{epoch}\ttotal_loss:{np.mean(total_loss):03.5f}'
            f'\treconstruction_loss:{np.mean(total_reconstruction_loss):03.5f}'
            f'\ttriplet_loss:{np.mean(total_triplet_loss):03.5f}'
            f'\tl2_loss:{np.mean(total_l2_loss):03.5f}\t'
        )
        for modality, loss in total_reconstruction_loss_by_modality.items():
            result_string += f'{modality}_reconstruction_loss:{np.mean(loss):03.5f}\t'
        for modality, loss in total_triplet_loss_by_modality.items():
            result_string += f'{modality}_triplet_loss:{np.mean(loss):03.5f}\t'
        print(result_string, file=source_file)

        if save_update_epochs and (epoch % save_epoch == 0):
            save_results(ae_model, protein_dataset, data_wrapper, results_suffix=f'_epoch{epoch}')

    embeddings_by_protein = save_results(ae_model, protein_dataset, data_wrapper)
    source_file.close()

    for protein, embeddings in embeddings_by_protein.items():
        average_embedding = np.mean(list(embeddings.values()), axis=0)
        row = [protein]
        row.extend(average_embedding)
        yield row
