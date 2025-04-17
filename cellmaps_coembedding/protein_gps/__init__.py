import torch.optim as optim
from torch.utils.data import DataLoader
from .architecture import *
import collections
import torch.nn.functional as F
import csv
import random
import itertools


def write_embedding_dictionary_to_file(filepath, dictionary, dims):
    """
    Writes a dictionary of embeddings to a tab-separated file with headers.

    :param filepath: Path to the file where embeddings will be saved.
    :type filepath: str
    :param dictionary: Dictionary of embeddings with keys as names and values as embedding vectors.
    :type dictionary: dict
    :param dims: Dimension of embedding vectors.
    :type dims: int
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


def save_results(model, protein_dataset, data_wrapper, results_suffix=''):
    """
    Evaluates the model, saves the state, and exports embeddings for each protein.

    :param model: The neural network model.
    :type model: torch.nn.Module
    :param protein_dataset: The dataset containing protein data.
    :type protein_dataset: cellmaps_coembedding.autoembed_sc.architecture.Protein_Dataset
    :param data_wrapper: Data handling and configurations as an object.
    :type data_wrapper: TrainingDataWrapper
    :param results_suffix: Suffix to append to results directory for saving.
    :type results_suffix: str
    """
    resultsdir = data_wrapper.resultsdir + results_suffix
    model.eval()
    torch.save(model.state_dict(), '{}_model.pth'.format(resultsdir))

    all_latents = dict()
    all_outputs = dict()
    for input_modality in data_wrapper.modalities_dict.keys():
        all_latents[input_modality] = dict()
        for output_modality in data_wrapper.modalities_dict.keys():
            output_key = input_modality + '_' + output_modality
            all_outputs[output_key] = dict()

    embeddings_by_protein = dict()
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
                input_modality = modality.split('_')[0]
                output_modality = modality.split('_')[1]
                if (mask[
                    input_modality] > 0):  # just need input modality to be there... #& (mask[output_modality] > 0):
                    all_outputs[modality][protein_name] = output.detach().cpu().numpy()

    # save latent embeddings
    for modality, latents in all_latents.items():
        filepath = '{}_{}_latent.tsv'.format(resultsdir, modality)
        write_embedding_dictionary_to_file(filepath, latents, data_wrapper.latent_dim)

    # save averaged coembedding
    filepath = '{}_latent.tsv'.format(resultsdir)
    write_embedding_dictionary_to_file(filepath, embeddings_by_protein, data_wrapper.latent_dim)

    # save reconstructed embeddings
    for modality, outputs in all_outputs.items():
        filepath = '{}_{}_reconstructed.tsv'.format(resultsdir, modality)
        output_modality = modality.split('_')[1]
        output_modality_dim = data_wrapper.modalities_dict[output_modality].input_dim
        write_embedding_dictionary_to_file(filepath, outputs, output_modality_dim)

    return embeddings_by_protein


def fit_predict(resultsdir, modality_data,
                modality_names=[],
                batch_size=16,
                latent_dim=128,
                n_epochs=250,
                triplet_margin=0.2,
                lambda_reconstruction=5.0,
                lambda_triplet=5.0,
                lambda_l2=0.001,
                l2_norm=False,
                dropout=0,
                save_epoch=50,
                learn_rate=1e-4,
                hidden_size_1=512,
                hidden_size_2=256,
                save_update_epochs=False,
                mean_losses=False,
                negative_from_batch=False):
    """
    Trains and predicts using a deep learning model with the given configuration and data.

    :param resultsdir: Directory to save training results and models.
    :type resultsdir: str
    :param modality_data: Input data for the model.
    :type modality_data: dict
    :param modality_names: Names of modalities; autogenerated if not provided.
    :type modality_names: list, optional
    :param batch_size: Batch size for training.
    :type batch_size: int
    :param latent_dim: Dimensionality of the latent embeddings.
    :type latent_dim: int
    :param n_epochs: Number of training epochs.
    :type n_epochs: int
    :param triplet_margin: Margin for triplet loss.
    :type triplet_margin: float
    :param lambda_reconstruction: Weight for reconstruction loss.
    :type lambda_reconstruction: float
    :param lambda_triplet: Weight for triplet loss.
    :type lambda_triplet: float
    :param l2_norm: Whether to use L2 normalization.
    :type l2_norm: bool
    :param dropout: Dropout rate.
    :type dropout: float
    :param save_epoch: Epoch interval at which to save the model.
    :type save_epoch: int
    :param learn_rate: Learning rate for the optimizer.
    :type learn_rate: float
    :param hidden_size_1: Size of the first hidden layer.
    :type hidden_size_1: int
    :param hidden_size_2: Size of the second hidden layer.
    :type hidden_size_2: int
    :param save_update_epochs: Flag to save model state at specified epoch intervals.
    :type save_update_epochs: bool
    :param mean_losses: Whether to average losses or not.
    :type mean_losses: bool
    :param negative_from_batch: Whether to use negative samples from the same batch for triplet loss.
    :type negative_from_batch: bool

    :returns: Generator of average embeddings for each protein.
    :rtype: generator
    """
    source_file = open('{}.txt'.format(resultsdir), 'w')
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name()

    # if modality names doesn't match data size, create names with index
    num_data_modalities = len(modality_data)
    if len(modality_names) != num_data_modalities:
        modality_names = ['modality_'.format(x) for x in np.arange(num_data_modalities)]

    data_wrapper = TrainingDataWrapper(modality_data, modality_names, device, l2_norm, dropout,
                                       latent_dim, hidden_size_1, hidden_size_2, resultsdir)

    # create models, optimizer, trainloader
    AE_model = uniembed_nn(data_wrapper).to(device)

    AE_optimizer = optim.Adam(AE_model.parameters(), lr=learn_rate)

    protein_dataset = Protein_Dataset(data_wrapper.modalities_dict)
    train_loader = DataLoader(protein_dataset, batch_size=batch_size, shuffle=True)
    print(f'lambda L2: {lambda_l2}')
    print(f'lambda triplet loss: {lambda_triplet}')

    for epoch in range(n_epochs):

        # train
        total_loss = []
        total_reconstruction_loss = []
        total_triplet_loss = []
        total_l2_loss = []
        
        total_reconstruction_loss_by_modality = collections.defaultdict(list)  # key: inputmodality_outputmodality
        total_triplet_loss_by_modality = collections.defaultdict(list)  # key: modality
        AE_model.train()

        # loop over all batches
        for step, (batch_data, batch_mask, batch_proteins) in enumerate(train_loader):

            # pass through model
            latents, outputs = AE_model(batch_data)

            batch_reconstruction_losses = torch.tensor([]).to(device)
            batch_triplet_losses = torch.tensor([]).to(device)
            batch_l2_losses = torch.tensor([])

            for input_modality in batch_data.keys():
                l2_loss = torch.norm(latents[input_modality], p=2, dim=1)
                batch_l2_losses = torch.cat((batch_l2_losses, l2_loss))
                
                # get reconstruction loss
                for output_modality in batch_data.keys():

                    # protein_present in both modalities mask
                    mask = (batch_mask[input_modality].bool()) & (batch_mask[output_modality].bool())
                    if torch.sum(mask) == 0:
                        continue  # no overlap

                    output_key = input_modality + '_' + output_modality

                    # compare OUTPUT modality original embedding to output embedding
                    pairwise_dist_input_output = 1 - F.cosine_similarity(batch_data[output_modality],
                                                                         outputs[output_key], dim=1)
                    reconstruction_loss = pairwise_dist_input_output[mask]
                    batch_reconstruction_losses = torch.cat((batch_reconstruction_losses, reconstruction_loss))
                    total_reconstruction_loss_by_modality[output_key].append(
                        torch.mean(reconstruction_loss).detach().cpu().numpy())

            # get triplet losses
            for anchor_modality in batch_data.keys():
                posneg_modality = random.choice(list([x for x in batch_data.keys() if x != anchor_modality]))

                mask = (batch_mask[anchor_modality].bool()) & (batch_mask[posneg_modality].bool())
                if batch_mask[posneg_modality].sum() < 2:
                    continue  # need at least 2 proteins in batch to make triplet with negative (if only one, th)
                if torch.sum(mask) == 0:
                    continue  # no overlap, need at least 1

                anchor_latents = latents[anchor_modality]
                positive_latents = latents[posneg_modality]
                positive_dist = 1 - F.cosine_similarity(anchor_latents, positive_latents, dim=1)

                positive_mask = torch.eye(len(mask))
                # pick random negative each anchor protein (criteria = not same protein, and negative that exists in
                # negative modality(not masked))
                if negative_from_batch:
                    # within same batch
                    negative_mask = (torch.logical_not(positive_mask) & (batch_mask[posneg_modality].bool()))
                    negative_indices = [x.nonzero().flatten() for x in negative_mask]
                    negative_index = [int(x[torch.randperm(len(x))[0]]) for x in negative_indices]
                    negative_latents = latents[posneg_modality][negative_index]
                else:
                    # any protein
                    posneg_modality_indices = np.arange(len(data_wrapper.modalities_dict[posneg_modality].train_labels))
                    protein_indexes_not_in_batch = list(set(posneg_modality_indices) - set(batch_proteins))
                    negative_indices = random.sample(protein_indexes_not_in_batch, len(positive_dist))
                    negative_data = {posneg_modality:
                                         data_wrapper.modalities_dict[posneg_modality].train_features[negative_indices]}
                    negative_latents_dict, _ = AE_model(negative_data)
                    negative_latents = negative_latents_dict[posneg_modality]

                negative_dist = 1 - F.cosine_similarity(anchor_latents, negative_latents, dim=1)

                # triplet is max of 0 or positive - negative
                triplet_loss = torch.maximum(positive_dist - negative_dist + triplet_margin,
                                             torch.zeros(len(positive_dist)).to(device))
                triplet_loss = triplet_loss[mask]

                batch_triplet_losses = torch.cat((batch_triplet_losses, triplet_loss))
                total_triplet_loss_by_modality[anchor_modality + '_' +
                                               posneg_modality].append(torch.mean(triplet_loss).detach().cpu().numpy())

            if (len(batch_reconstruction_losses) == 0) | (len(batch_triplet_losses) == 0):
                continue  # didn't have any overlapping proteins in any modalities

            # calc losses
            if mean_losses:
                reconstruction_loss = torch.mean(batch_reconstruction_losses)
                triplet_loss = torch.mean(batch_triplet_losses)
                l2_loss = torch.mean(batch_l2_losses)

            else:
                reconstruction_loss = torch.sum(batch_reconstruction_losses)
                triplet_loss = torch.sum(batch_triplet_losses)
                l2_loss = torch.sum(batch_l2_losses)

            batch_total = lambda_reconstruction * reconstruction_loss + lambda_triplet * triplet_loss + lambda_l2 * l2_loss

            AE_optimizer.zero_grad()
            batch_total.backward()
            AE_optimizer.step()

            total_loss.append(batch_total.detach().cpu().numpy())
            total_reconstruction_loss.append(reconstruction_loss.detach().cpu().numpy())
            total_triplet_loss.append(triplet_loss.detach().cpu().numpy())
            total_l2_loss.append(l2_loss.detach().cpu().numpy())

        # get result string wtith losses
        result_string = 'epoch:%d\ttotal_loss:%03.5f\treconstruction_loss:%03.5f\ttriplet_loss:%03.5f\tl2_loss:%03.5f\t' % (
            epoch,
            np.mean(total_loss),
            np.mean(total_reconstruction_loss),
            np.mean(total_triplet_loss),
            np.mean(total_l2_loss),
            )
        for modality, loss in total_reconstruction_loss_by_modality.items():
            result_string += '%s_reconstruction_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_triplet_loss_by_modality.items():
            result_string += '%s_triplet_loss:%03.5f\t' % (modality, np.mean(loss))
        
        print(result_string, file=source_file)

        # save results at update epochs
        if (save_update_epochs) & (epoch % save_epoch == 0):
            save_results(AE_model, protein_dataset, data_wrapper, results_suffix='_epoch{}'.format(epoch))

    # save final results
    embeddings_by_protein = save_results(AE_model, protein_dataset, data_wrapper)
    source_file.close()

    #    average embeddings for each protein and return as coemembedding
    for protein, embeddings in embeddings_by_protein.items():
        average_embedding = np.mean(list(embeddings.values()), axis=0)
        row = [protein]
        row.extend(average_embedding)
        yield row
